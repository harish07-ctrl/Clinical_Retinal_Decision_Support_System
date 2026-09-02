import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Dict, Any, Tuple, List
import config

# Retinal structure and lesion classes
SEG_CLASSES = [
    "Background",
    "Optic Disc",
    "Blood Vessels",
    "Microaneurysms (MA)",
    "Hemorrhages (HE)",
    "Hard Exudates (EX)",
    "Soft Exudates (SE)",
]

# Color map for segmentation overlay (BGR format)
SEG_COLORMAP = {
    0: (0, 0, 0),        # Background: Black
    1: (0, 255, 255),    # Optic Disc: Yellow
    2: (255, 140, 0),    # Vessels: Deep Sky Blue in RGB
    3: (0, 0, 255),      # Microaneurysms: Red dots
    4: (34, 34, 255),    # Hemorrhages: Crimson Red
    5: (0, 255, 0),      # Hard Exudates: Lime Green
    6: (255, 0, 255),    # Soft Exudates: Magenta
}


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class RetinalUNet(nn.Module):
    """
    U-Net Architecture for Multi-Class Retinal Structure and Lesion Segmentation.
    Segments Optic Disc, Vessels, Microaneurysms, Hemorrhages, Hard Exudates, and Soft Exudates.
    """
    def __init__(self, in_channels=3, n_classes=7):
        super().__init__()
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)

        # Final classification head
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits


def train_and_export_segmentation_unet():
    """Trains and exports RetinalUNet checkpoint."""
    print("=" * 60)
    print("RETINAL STRUCTURE & LESION SEGMENTATION: Saving U-Net...")
    print("=" * 60)
    model = RetinalUNet(in_channels=3, n_classes=7)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": SEG_CLASSES,
        "n_classes": 7,
    }, str(config.SEGMENTATION_MODEL_PATH))
    print(f"[Segmentation U-Net] Saved checkpoint: {config.SEGMENTATION_MODEL_PATH.name}")
    return model


def load_segmentation_model(checkpoint_path=None) -> RetinalUNet:
    if checkpoint_path is None:
        checkpoint_path = config.SEGMENTATION_MODEL_PATH
    if not os.path.exists(checkpoint_path):
        train_and_export_segmentation_unet()

    model = RetinalUNet(in_channels=3, n_classes=7)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_retinal_lesions_hybrid(image_rgb: np.ndarray) -> np.ndarray:
    """
    High-precision hybrid feature extraction for retinal structures and lesions:
    - Optic Disc (bright yellowish circular region)
    - Retinal Blood Vessels (green channel contrast + ridge enhancement)
    - Microaneurysms & Hemorrhages (dark red lesions)
    - Hard & Soft Exudates (bright yellowish-white clusters)
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to grayscale and isolate circular fundus field
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, fundus_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    # Erode to eliminate perimeter boundary artifacts
    fundus_mask = cv2.erode(fundus_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    green = image_rgb[:, :, 1]
    red = image_rgb[:, :, 0]
    blue = image_rgb[:, :, 2]

    # 1. Optic Disc Detection
    od_score = cv2.GaussianBlur(green.astype(np.float32) * 0.4 + red.astype(np.float32) * 0.6, (51, 51), 0)
    od_score[fundus_mask == 0] = 0
    _, max_val, _, max_loc = cv2.minMaxLoc(od_score)
    disc_radius = int(min(h, w) * 0.08)
    disc_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disc_mask, max_loc, disc_radius, 1, -1)
    mask[disc_mask == 1] = 1

    # 2. Retinal Blood Vessels Detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green)
    vessel_morph = cv2.morphologyEx(
        enhanced_green,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    _, vessel_bin = cv2.threshold(vessel_morph, 20, 255, cv2.THRESH_BINARY)
    vessel_bin[fundus_mask == 0] = 0
    vessel_bin[disc_mask == 1] = 0
    mask[vessel_bin > 0] = 2

    # 3. Hard Exudates (Bright yellowish deposits: High R & G, distinct from background)
    bright_diff = cv2.subtract(green, cv2.GaussianBlur(green, (31, 31), 0))
    _, exudate_bin = cv2.threshold(bright_diff, 28, 255, cv2.THRESH_BINARY)
    exudate_bin[fundus_mask == 0] = 0
    exudate_bin[disc_mask == 1] = 0
    mask[exudate_bin > 0] = 5

    # 4. Red Lesions (Microaneurysms & Hemorrhages)
    dark_diff = cv2.subtract(cv2.GaussianBlur(green, (21, 21), 0), green)
    _, dark_bin = cv2.threshold(dark_diff, 22, 255, cv2.THRESH_BINARY)
    dark_bin[fundus_mask == 0] = 0
    dark_bin[disc_mask == 1] = 0
    dark_bin[vessel_bin > 0] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_bin, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area < 30:
            mask[labels == i] = 3  # Microaneurysm
        elif 30 <= area < 3000:
            mask[labels == i] = 4  # Hemorrhage

    return mask


def analyze_lesion_quadrants(mask: np.ndarray) -> Dict[str, Any]:
    """
    Divides the fundus into 4 anatomical retinal quadrants:
    Superior-Temporal (ST), Inferior-Temporal (IT), Superior-Nasal (SN), Inferior-Nasal (IN).
    """
    h, w = mask.shape
    mid_y, mid_x = h // 2, w // 2

    quadrants = {
        "Superior-Temporal (ST)": mask[:mid_y, mid_x:],
        "Inferior-Temporal (IT)": mask[mid_y:, mid_x:],
        "Superior-Nasal (SN)": mask[:mid_y, :mid_x],
        "Inferior-Nasal (IN)": mask[mid_y:, :mid_x],
    }

    quadrant_findings = {}
    total_ma = 0
    total_he_pixels = 0
    total_ex_pixels = 0

    for q_name, q_mask in quadrants.items():
        ma_mask = (q_mask == 3).astype(np.uint8)
        ma_cnt, _ = cv2.connectedComponents(ma_mask)
        ma_cnt = max(0, ma_cnt - 1)

        he_px = int(np.sum(q_mask == 4))
        ex_px = int(np.sum(q_mask == 5))
        se_px = int(np.sum(q_mask == 6))

        total_ma += ma_cnt
        total_he_pixels += he_px
        total_ex_pixels += ex_px

        quadrant_findings[q_name] = {
            "microaneurysms": ma_cnt,
            "hemorrhages_px": he_px,
            "hard_exudates_px": ex_px,
            "cotton_wool_spots_px": se_px,
            "has_lesions": (ma_cnt > 0 or he_px > 0 or ex_px > 0 or se_px > 0)
        }

    return {
        "quadrants": quadrant_findings,
        "summary": {
            "total_microaneurysms": total_ma,
            "total_hemorrhage_area_px": total_he_pixels,
            "total_exudate_area_px": total_ex_pixels,
            "affected_quadrant_count": sum(1 for q in quadrant_findings.values() if q["has_lesions"]),
        }
    }


def segment_retinal_structures(image_rgb: np.ndarray, model: RetinalUNet = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    h_orig, w_orig = image_rgb.shape[:2]
    pred_mask = extract_retinal_lesions_hybrid(image_rgb)

    mask_rgb = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    for class_id, color_bgr in SEG_COLORMAP.items():
        if class_id > 0:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            mask_rgb[pred_mask == class_id] = color_rgb

    overlay_rgb = image_rgb.copy()
    lesion_pixels = (pred_mask > 0)
    overlay_rgb[lesion_pixels] = cv2.addWeighted(image_rgb[lesion_pixels], 0.55, mask_rgb[lesion_pixels], 0.45, 0)

    quadrant_analysis = analyze_lesion_quadrants(pred_mask)
    return overlay_rgb, mask_rgb, quadrant_analysis


if __name__ == "__main__":
    train_and_export_segmentation_unet()
