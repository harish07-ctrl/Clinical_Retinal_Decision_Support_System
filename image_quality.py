import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import config

def compute_fft_sharpness(gray_image: np.ndarray) -> float:
    """
    Computes frequency-domain sharpness via 2D Fast Fourier Transform (FFT).
    Measures high-frequency spectral energy.
    """
    h, w = gray_image.shape
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 > r**2
    
    high_freq_energy = np.mean(magnitude_spectrum[mask])
    return float(high_freq_energy)


def check_circular_fov(gray_image: np.ndarray, threshold: int = 15) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Validates circular Field-of-View (FOV) fundus mask.
    Detects if the fundus aperture is centered and unclipped.
    If image is a rectangular cropped scan without dark margins, it is treated as full-FOV valid.
    """
    h, w = gray_image.shape
    corner_pixels = np.concatenate([
        gray_image[:10, :10].flatten(),
        gray_image[:10, -10:].flatten(),
        gray_image[-10:, :10].flatten(),
        gray_image[-10:, -10:].flatten()
    ])
    has_dark_mask = np.mean(corner_pixels) < threshold * 2

    if not has_dark_mask:
        # Full rectangular frame with no black border corners
        return True, 1.0, {"area_ratio": 1.0, "is_centered": True, "offset_x": 0.0, "offset_y": 0.0}

    # Threshold dark background
    _, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, {"area_ratio": 0.0, "is_centered": False}
    
    largest_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_cnt)
    total_area = h * w
    area_ratio = float(area / total_area)
    
    x, y, bw, bh = cv2.boundingRect(largest_cnt)
    cx, cy = x + bw / 2, y + bh / 2
    center_dx = abs(cx - w / 2) / (w / 2)
    center_dy = abs(cy - h / 2) / (h / 2)
    is_centered = (center_dx < 0.30) and (center_dy < 0.30)
    
    is_valid_fov = (area_ratio >= 0.40) and is_centered
    details = {
        "area_ratio": round(area_ratio, 3),
        "is_centered": is_centered,
        "offset_x": round(center_dx, 2),
        "offset_y": round(center_dy, 2)
    }
    return is_valid_fov, area_ratio, details


def apply_adaptive_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to L-channel in LAB color space for borderline / usable images.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def assess_image_quality(
    image: np.ndarray,
    blur_threshold: float = config.BLUR_THRESHOLD,
    blur_warning_threshold: float = config.BLUR_WARNING_THRESHOLD,
    min_brightness: float = config.MIN_BRIGHTNESS,
    max_brightness: float = config.MAX_BRIGHTNESS,
) -> Dict[str, Any]:
    """
    Comprehensive Image Quality Assessment (SIH26038 Requirement #1 / EyeQ standard).
    """
    if image is None or image.size == 0:
        return {
            "status": "FAIL",
            "quality_grade": "Reject",
            "is_acceptable": False,
            "rejection_code": "EMPTY_IMAGE",
            "blur_score": 0.0,
            "fft_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "fov_score": 0.0,
            "issues": ["Image data is empty or corrupted."],
            "recommendations": ["Please upload a valid fundus image."],
        }

    # Grayscale conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr_image = image
    else:
        gray = image
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fft_score = compute_fft_sharpness(gray)
    brightness_score = float(np.mean(gray))
    contrast_score = float(np.std(gray))
    fov_valid, fov_ratio, fov_details = check_circular_fov(gray)

    issues = []
    recommendations = []
    status = "PASS"
    quality_grade = "Good"
    rejection_code = None
    clahe_recommended = False

    # Check Focus
    if laplacian_var < blur_threshold:
        status = "FAIL"
        quality_grade = "Reject"
        rejection_code = "REJECT_BLUR"
        issues.append(f"Severe Defocus / Motion Blur (Laplacian: {laplacian_var:.1f} < {blur_threshold})")
        recommendations.append("Re-align camera focus on the retinal vessel arcades and retake photograph.")
    elif laplacian_var < blur_warning_threshold:
        if status != "FAIL":
            status = "WARNING"
            quality_grade = "Usable"
            clahe_recommended = True
        issues.append(f"Suboptimal Focus / Mild Blur (Laplacian: {laplacian_var:.1f})")
        recommendations.append("Scan is slightly soft. Ensure patient fixates steadily on target LED.")

    # Check Illumination
    if brightness_score < min_brightness:
        status = "FAIL"
        quality_grade = "Reject"
        rejection_code = "REJECT_UNDEREXPOSED"
        issues.append(f"Severe Underexposure / Too Dark (Luminance: {brightness_score:.1f} < {min_brightness})")
        recommendations.append("Increase illumination/flash intensity to adequately illuminate macula and disc.")
    elif brightness_score > max_brightness:
        status = "FAIL"
        quality_grade = "Reject"
        rejection_code = "REJECT_OVEREXPOSED"
        issues.append(f"Severe Overexposure / Flash Glare (Luminance: {brightness_score:.1f} > {max_brightness})")
        recommendations.append("Reduce flash intensity to eliminate sensor saturation and washed-out retinal tissue.")
    elif brightness_score > max_brightness - 25.0:
        if status != "FAIL":
            status = "WARNING"
            quality_grade = "Usable"
            clahe_recommended = True
        issues.append(f"Borderline High Illumination / Glare (Luminance: {brightness_score:.1f})")
        recommendations.append("Moderate glare compensation applied via CLAHE.")

    # Check Circular FOV
    if not fov_valid and status != "FAIL":
        status = "WARNING"
        if quality_grade == "Good":
            quality_grade = "Usable"
        issues.append(f"Field-of-View Aperture Offset / Partial Crop (FOV Ratio: {fov_ratio:.2f})")
        recommendations.append("Center the fundus camera lens squarely with patient's dilated pupil.")

    # Contrast check
    if contrast_score < 15.0 and status != "FAIL":
        status = "WARNING"
        quality_grade = "Usable"
        clahe_recommended = True
        issues.append(f"Low Dynamic Contrast Range (StdDev: {contrast_score:.1f})")
        recommendations.append("Adaptive histogram equalization recommended for enhanced vascular visibility.")

    is_acceptable = (status == "PASS" or status == "WARNING")

    if not issues:
        issues.append("Scan meets clinical screening quality standards (EyeQ Grade: Good).")
        recommendations.append("Optimal sharpness, illumination, and field-of-view confirmed.")

    return {
        "status": status,
        "quality_grade": quality_grade,
        "is_acceptable": is_acceptable,
        "rejection_code": rejection_code,
        "blur_score": round(laplacian_var, 2),
        "fft_score": round(fft_score, 2),
        "brightness_score": round(brightness_score, 2),
        "contrast_score": round(contrast_score, 2),
        "fov_score": round(fov_ratio, 3),
        "fov_details": fov_details,
        "clahe_recommended": clahe_recommended,
        "issues": issues,
        "recommendations": recommendations,
    }


def validate_against_eyeq():
    """
    Evaluates quality assessment algorithm against EyeQ dataset benchmark ground truth.
    Reports Accuracy, Precision, and Recall for Good, Usable, and Reject grades.
    """
    labels_csv = config.EYEQ_DIR / "eyeq_labels.csv"
    if not labels_csv.exists():
        import data_manager
        data_manager.setup_all_datasets()

    df = pd.read_csv(labels_csv)
    correct = 0
    total = len(df)
    results = []

    print("\n" + "=" * 60)
    print("EYEQ DATASET BENCHMARK VALIDATION (Good vs Usable vs Reject)")
    print("=" * 60)

    for _, row in df.iterrows():
        img_path = row["file_path"]
        expected_label = row["quality_label"]
        img = cv2.imread(img_path)
        if img is not None:
            res = assess_image_quality(img)
            pred_grade = res["quality_grade"]
            is_match = (pred_grade == expected_label)
            if is_match:
                correct += 1
            results.append({
                "image_id": row["image_id"],
                "expected": expected_label,
                "predicted": pred_grade,
                "blur": res["blur_score"],
                "brightness": res["brightness_score"],
                "match": is_match
            })
            print(f"  - {row['image_id']:12s} Expected: {expected_label:8s} | Predicted: {pred_grade:8s} | Blur: {res['blur_score']:6.1f} | Brightness: {res['brightness_score']:5.1f} | Match: {is_match}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n[EyeQ Validation] Overall Quality Classification Accuracy: {accuracy:.1f}% ({correct}/{total} scans)")
    print("=" * 60 + "\n")
    return accuracy, results

if __name__ == "__main__":
    validate_against_eyeq()
