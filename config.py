import os
from pathlib import Path

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = Path(os.getenv("DR_DATA_DIR", BASE_DIR / "data"))
APTOS_DIR = Path(os.getenv("APTOS_DATA_DIR", DATA_DIR / "aptos2019"))
IDRID_DIR = Path(os.getenv("IDRID_DATA_DIR", DATA_DIR / "idrid"))
EYEQ_DIR = Path(os.getenv("EYEQ_DATA_DIR", DATA_DIR / "eyeq"))
MESSIDOR_DIR = Path(os.getenv("MESSIDOR_DATA_DIR", DATA_DIR / "messidor2"))
DRIVE_DIR = Path(os.getenv("DRIVE_DATA_DIR", DATA_DIR / "drive"))

# Sub-directories for APTOS 2019
APTOS_TRAIN_IMAGES = APTOS_DIR / "train_images"
APTOS_TEST_IMAGES = APTOS_DIR / "test_images"
APTOS_TRAIN_CSV = BASE_DIR / "train.csv"
APTOS_TEST_CSV = BASE_DIR / "test.csv"

# Model Checkpoints & Artifacts
CHECKPOINT_DIR = Path(os.getenv("DR_CHECKPOINT_DIR", BASE_DIR / "checkpoints"))
PRIMARY_MODEL_PATH = BASE_DIR / "dr_model.pth"
RESNET_MODEL_PATH = BASE_DIR / "dr_resnet18.pth"
SEGMENTATION_MODEL_PATH = BASE_DIR / "dr_segmentation_unet.pth"
TFLITE_MODEL_PATH = BASE_DIR / "dr_model.tflite"
TFLITE_QUANT_PATH = BASE_DIR / "dr_model_quantized.tflite"

# Ensure essential directories exist
for d in [DATA_DIR, APTOS_DIR, APTOS_TRAIN_IMAGES, APTOS_TEST_IMAGES, IDRID_DIR, EYEQ_DIR, MESSIDOR_DIR, DRIVE_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Image & Model Hyperparameters
IMAGE_SIZE = (224, 224)
SEG_IMAGE_SIZE = (256, 256)
NUM_CLASSES = 5
SEVERITY_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
REFERABLE_CLASSES = ["Non-Referable DR (Grade 0-1)", "Referable DR (Grade 2+)"]

# Quality Thresholds (EyeQ Standard)
BLUR_THRESHOLD = 60.0
BLUR_WARNING_THRESHOLD = 120.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 215.0
MIN_FOV_RATIO = 0.65

# Telemedicine District Parameters (SIH26038 Specification)
ANNUAL_PATIENT_LOAD = 100000
PHC_STATIONS_DEFAULT = 25
TELE_DOCTORS_DEFAULT = 4
AI_TRIAGE_TIME_SEC = 2.5
DOCTOR_REVIEW_TIME_SEC = 30.0  # Targeted <30s rapid review
TARGET_TURNAROUND_HOURS = 24.0
