import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

class RetinalModel(nn.Module):
    """
    Diabetic Retinopathy Severity Classification Model.
    Supports EfficientNet-B0 as the primary backbone and ResNet-18 as a lightweight alternative,
    aligning with the pitch deck modeling specifications.
    """

    def __init__(self, n_classes=5, backbone="efficientnet-b0"):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.n_classes = n_classes

        if "efficientnet" in self.backbone_name:
            # EfficientNet-B0 primary architecture
            self.model = EfficientNet.from_pretrained(self.backbone_name)
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, n_classes)
        elif "resnet18" in self.backbone_name or "resnet-18" in self.backbone_name:
            # ResNet-18 alternative lightweight architecture
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, n_classes)
        else:
            # Fallback to EfficientNet-B0
            print(f"[Warning] Unknown backbone '{backbone}'. Defaulting to 'efficientnet-b0'.")
            self.backbone_name = "efficientnet-b0"
            self.model = EfficientNet.from_pretrained("efficientnet-b0")
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)


def load_retinal_model(checkpoint_path="dr_model.pth", backbone="efficientnet-b0"):
    """
    Loads the Diabetic Retinopathy model checkpoint with detailed key verification
    and startup logging to prevent silent weight dropping or chance-level predictions.

    Args:
        checkpoint_path (str): Path to .pth checkpoint file.
        backbone (str): Model backbone ("efficientnet-b0" or "resnet18").

    Returns:
        RetinalModel: Model loaded in eval mode.
    """
    print(f"\n{'='*60}")
    print(f"[Model Loader] Initializing RetinalModel (Backbone: {backbone})")
    print(f"[Model Loader] Loading checkpoint from: {checkpoint_path}")

    # Build model architecture
    model = RetinalModel(n_classes=5, backbone=backbone)

    if not os.path.exists(checkpoint_path):
        error_msg = (
            f"[Error] Checkpoint file '{checkpoint_path}' not found!\n"
            f"Please ensure '{checkpoint_path}' exists or generate it using training/export scripts."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    model_state = model.state_dict()

    # Analyze state dict matching
    matched_keys = []
    mismatched_shape_keys = []
    missing_keys = []
    unexpected_keys = []

    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                matched_keys.append(k)
            else:
                mismatched_shape_keys.append((k, str(v.shape), str(model_state[k].shape)))
        else:
            unexpected_keys.append(k)

    for k in model_state.keys():
        if k not in state_dict:
            missing_keys.append(k)

    total_model_keys = len(model_state)
    load_percentage = (len(matched_keys) / total_model_keys * 100) if total_model_keys > 0 else 0

    print(f"[Model Loader] Total Model Parameter Tensors: {total_model_keys}")
    print(f"[Model Loader] Successfully Matched Keys:     {len(matched_keys)} ({load_percentage:.1f}%)")
    print(f"[Model Loader] Missing Keys in Checkpoint:    {len(missing_keys)}")
    print(f"[Model Loader] Shape Mismatches:              {len(mismatched_shape_keys)}")
    print(f"[Model Loader] Unexpected Keys in Checkpoint: {len(unexpected_keys)}")

    if mismatched_shape_keys:
        print("[Model Loader] Mismatched shapes preview:")
        for k, src_s, tgt_s in mismatched_shape_keys[:3]:
            print(f"   - {k}: checkpoint {src_s} vs model {tgt_s}")

    if len(matched_keys) == 0:
        raise RuntimeError(
            f"[Error] Zero state_dict keys matched between checkpoint '{checkpoint_path}' "
            f"and architecture '{backbone}'. Model cannot make valid predictions."
        )

    # Load matched parameters
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[Model Loader] RetinalModel successfully loaded and initialized ({backbone}).")
    print(f"{'='*60}\n")
    return model
