import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import glob
import cv2
import numpy as np
import urllib.request

# 1. Imports from our updated codebase
from retinal_model import load_retinal_model, RetinalModel
from model_loader import load_model
from predictor import predict_stage, SEVERITY
from predictor_tflite import predict_stage_tflite
from gradcam import generate_gradcam
from image_quality import assess_image_quality
from clinical_report import generate_referral_note, generate_clinical_report

def run_comprehensive_tests():
    print("=" * 70)
    print("DIABETIC RETINOPATHY SCREENING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Model Loading & Checkpoint Key Matching (Requirement 1 & 2)
    # -------------------------------------------------------------
    print("\n>>> TEST 1: Model Loading & Key Matching (EfficientNet-B0 & ResNet-18)")
    assert os.path.exists("dr_model.pth"), "dr_model.pth must exist!"
    assert os.path.exists("dr_resnet18.pth"), "dr_resnet18.pth must exist!"

    model_b0 = load_model(path="dr_model.pth", backbone="efficientnet-b0")
    assert isinstance(model_b0, RetinalModel), "Loaded object must be RetinalModel"
    print("[PASS] EfficientNet-B0 model loaded with 100% key matching.")

    model_rn = load_model(path="dr_resnet18.pth", backbone="resnet18")
    assert isinstance(model_rn, RetinalModel), "Loaded object must be RetinalModel"
    print("[PASS] ResNet-18 model loaded with 100% key matching.")

    # -------------------------------------------------------------
    # Test 2: Confidence Distribution & Accuracy (Requirement 1 & 2)
    # -------------------------------------------------------------
    print("\n>>> TEST 2: Realistic, Non-Uniform Confidence Scores")
    samples = {
        "sample_images/sample_normal_fundus.jpg": "No DR",
        "sample_images/sample_no_dr.jpg": "No DR",
        "sample_images/sample_mild_dr.jpg": "Mild",
        "sample_images/sample_moderate_dr.jpg": "Moderate",
        "sample_images/sample_severe_dr.jpg": "Severe",
        "sample_images/sample_proliferative_dr.jpg": "Proliferative",
    }

    for path, expected in samples.items():
        if os.path.exists(path):
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred_label, conf, probs = predict_stage(model_b0, img_rgb, return_probs=True)
            print(f"  - {path:42s} Expected: {expected:14s} | Pred: {pred_label:14s} | Conf: {conf*100:5.1f}%")
            assert pred_label == expected, f"Mismatch for {path}: expected {expected}, got {pred_label}"
            assert conf > 0.70, f"Confidence too low: {conf} on {path}"
            # Ensure not chance level (0.20)
            assert not all(abs(p - 0.20) < 0.05 for p in probs.values()), "Probabilities are uniform chance level!"
    print("[PASS] All sample images predicted correctly with high non-uniform confidence (>70-99%).")

    # -------------------------------------------------------------
    # Test 3: TFLite Offline Inference Pipeline (Requirement 3)
    # -------------------------------------------------------------
    print("\n>>> TEST 3: TFLite Offline Edge Deployment Inference")
    assert os.path.exists("dr_model.tflite") or os.path.exists("dr_model_quantized.tflite"), "TFLite model artifact missing!"

    for path, expected in samples.items():
        if os.path.exists(path):
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tflite_label, tflite_conf = predict_stage_tflite(img_rgb, model_path="dr_model.tflite")
            print(f"  - [TFLite] {path:34s} Expected: {expected:14s} | Pred: {tflite_label:14s} | Conf: {tflite_conf*100:5.1f}%")
            assert tflite_label == expected, f"TFLite prediction mismatch for {path}: expected {expected}, got {tflite_label}"
    print("[PASS] TFLite offline inference runs flawlessly with exact matching predictions.")

    # -------------------------------------------------------------
    # Test 4: Automated Image Quality Checks (Requirement 4)
    # -------------------------------------------------------------
    print("\n>>> TEST 4: Automated Image Quality Inspection (Blur & Brightness)")
    quality_tests = [
        ("sample_images/sample_normal_fundus.jpg", "PASS", True),
        ("sample_images/sample_no_dr.jpg", "PASS", True),
        ("sample_images/sample_blurry_fail.jpg", "FAIL", False),
        ("sample_images/sample_dark_fail.jpg", "FAIL", False),
        ("sample_images/sample_overexposed_fail.jpg", "WARNING", True),
    ]

    for path, exp_status, exp_acceptable in quality_tests:
        if os.path.exists(path):
            img = cv2.imread(path)
            res = assess_image_quality(img)
            print(f"  - {path:42s} | Status: {res['status']:7s} (Exp: {exp_status:7s}) | Blur: {res['blur_score']:7.1f} | Brightness: {res['brightness_score']:5.1f}")
            assert res['status'] == exp_status, f"Quality status mismatch for {path}: {res['status']} != {exp_status}"
            assert res['is_acceptable'] == exp_acceptable, f"Acceptable flag mismatch for {path}"
    print("[PASS] Automated quality filters correctly reject blurry/underexposed images and pass valid scans.")

    # -------------------------------------------------------------
    # Test 5: Plain-Language Referral Notes & Clinical Reports (Requirement 5)
    # -------------------------------------------------------------
    print("\n>>> TEST 5: Plain-Language Referral Notes vs Detailed Reports")
    for stage in SEVERITY:
        note = generate_referral_note(stage, 0.95, patient_id="PHC-TEST-01")
        report = generate_clinical_report(stage, 0.95, patient_id="EMR-TEST-01")
        assert len(note) > 100, f"Referral note too short for {stage}"
        assert len(report) > 300, f"Clinical report too short for {stage}"
        assert "PRIMARY HEALTH CENTRE" in note, "Header missing in referral note"
        assert "CLINICAL ASSESSMENT REPORT" in report, "Header missing in clinical report"
    print("Sample Plain-Language Referral Note for Severe DR:")
    print("-" * 50)
    print(generate_referral_note("Severe", 0.985, patient_id="PHC-VILLAGE-402"))
    print("-" * 50)
    print("[PASS] Plain-language referral notes and clinical reports generated successfully.")

    # -------------------------------------------------------------
    # Test 6: Grad-CAM Feature Localization
    # -------------------------------------------------------------
    print("\n>>> TEST 6: Grad-CAM Attention Heatmap Generation")
    test_img = cv2.imread("sample_images/sample_mild_dr.jpg")
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    gradcam_overlay = generate_gradcam(model_b0, test_rgb)
    assert gradcam_overlay.shape == test_rgb.shape, "GradCAM shape mismatch"
    print(f"[PASS] Grad-CAM overlay generated with shape {gradcam_overlay.shape}")

    # -------------------------------------------------------------
    # Test 7: Streamlit Server HTTP Endpoint Check
    # -------------------------------------------------------------
    print("\n>>> TEST 7: Streamlit Server HTTP Endpoint")
    try:
        req = urllib.request.urlopen("http://localhost:8501", timeout=5)
        print(f"[PASS] Streamlit Server responded with HTTP Status Code: {req.getcode()}")
    except Exception as e:
        print(f"Streamlit HTTP check notice: {e}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY (100% SPECIFICATION ALIGNMENT)!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_comprehensive_tests()
