import cv2
import numpy as np
import os

os.makedirs('sample_images', exist_ok=True)

# 1. From single_patient_report.png (has 4 panels: Original, AI Attention Map, Overlay, Report)
img1 = cv2.imread('single_patient_report.png')
if img1 is not None:
    h, w, _ = img1.shape
    # Leftmost ~25% contains original retinal image
    crop1 = img1[180:1350, 100:1100]
    cv2.imwrite('sample_images/sample_normal_fundus.jpg', crop1)
    print("Saved sample_normal_fundus.jpg:", crop1.shape)

# 2. From clinical_reports_sample.png (has a 3x5 grid of sample cases from notebook)
img2 = cv2.imread('clinical_reports_sample.png')
if img2 is not None:
    h, w, _ = img2.shape
    # The top row contains 5 original fundus images of different DR stages
    # Let's crop them
    col_w = w // 5
    for c in range(5):
        # Top row roughly in y: 150 to 1150
        panel = img2[150:1150, c*col_w + 80 : (c+1)*col_w - 80]
        if panel.size > 0:
            names = ["sample_no_dr.jpg", "sample_mild_dr.jpg", "sample_moderate_dr.jpg", "sample_severe_dr.jpg", "sample_proliferative_dr.jpg"]
            cv2.imwrite(f'sample_images/{names[c]}', panel)
            print(f"Saved {names[c]}:", panel.shape)

# Also create quality test images: blurry and dark/bright variants for testing image_quality.py
if os.path.exists('sample_images/sample_no_dr.jpg'):
    base = cv2.imread('sample_images/sample_no_dr.jpg')
    # Blurry image (Gaussian blur)
    blurry = cv2.GaussianBlur(base, (45, 45), 0)
    cv2.imwrite('sample_images/sample_blurry_fail.jpg', blurry)
    
    # Dark image (underexposed)
    dark = (base * 0.15).astype(np.uint8)
    cv2.imwrite('sample_images/sample_dark_fail.jpg', dark)
    
    # Overexposed image
    bright = np.clip(base * 2.2 + 80, 0, 255).astype(np.uint8)
    cv2.imwrite('sample_images/sample_overexposed_fail.jpg', bright)
    print("Saved quality test images: blurry, dark, overexposed.")
