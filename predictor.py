import torch
import cv2
import numpy as np
import torchvision.transforms as T
from typing import Dict, Any, Tuple
import config

SEVERITY = config.SEVERITY_CLASSES

# Preprocessing transform for EfficientNet-B0 / ResNet-18 (224x224)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(config.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_stage(model, image: np.ndarray, return_probs: bool = False, temperature: float = 1.0):
    """
    Predicts Diabetic Retinopathy severity stage and Referable DR status.

    Args:
        model: PyTorch RetinalModel
        image (np.ndarray): RGB fundus image
        return_probs (bool): If True, return full diagnostic details dictionary
        temperature (float): Temperature parameter for probability calibration

    Returns:
        tuple: (predicted_class_name, confidence_score) or full diagnostic dict
    """
    model.eval()
    img_tensor = transform(image).unsqueeze(0)

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        # Apply temperature scaling for calibrated probabilities
        calibrated_logits = output / max(1e-4, temperature)
        probs = torch.softmax(calibrated_logits, dim=1)[0]
        score, idx = torch.max(probs, 0)

    severity_label = SEVERITY[idx.item()]
    confidence = float(score.item())

    # Compute Referable DR (Grade 2, 3, 4) probability
    rdr_prob = float(torch.sum(probs[2:]).item())
    is_referable = rdr_prob >= 0.50
    referable_status = "Referable DR (Grade 2+)" if is_referable else "Non-Referable DR (Grade 0-1)"

    if return_probs:
        prob_dict = {SEVERITY[i]: float(probs[i].item()) for i in range(len(SEVERITY))}
        diag_details = {
            "severity_label": severity_label,
            "confidence": confidence,
            "is_referable": is_referable,
            "referable_status": referable_status,
            "referable_probability": rdr_prob,
            "probabilities": prob_dict,
        }
        return severity_label, confidence, prob_dict

    return severity_label, confidence
