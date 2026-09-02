import torch
from retinal_model import load_retinal_model, RetinalModel


def load_model(path="dr_model.pth", backbone="efficientnet-b0"):
    """
    Load the Diabetic Retinopathy model using specified backbone (default: EfficientNet-B0).
    Aligns with Pitch Deck Slide 3: PyTorch (EfficientNet-B0 / ResNet18).
    """
    model = load_retinal_model(checkpoint_path=path, backbone=backbone)
    model.eval()
    return model
