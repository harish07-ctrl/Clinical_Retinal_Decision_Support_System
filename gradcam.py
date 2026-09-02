import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(model, image: np.ndarray, return_raw: bool = False):
    """
    Generates Grad-CAM visual explanations for EfficientNet-B0 and ResNet-18 models.

    Args:
        model: RetinalModel instance
        image (np.ndarray): RGB input image
        return_raw (bool): If True, returns tuple (overlay_rgb, raw_heatmap_norm)

    Returns:
        np.ndarray or tuple: RGB overlay image with Grad-CAM heatmap, or (overlay, raw_heatmap)
    """
    model.eval()
    img_tensor = transform(image).unsqueeze(0)

    # Determine target layer based on architecture
    target_layer = None
    if hasattr(model, "model"):
        inner_model = model.model
        if hasattr(inner_model, "_conv_head"):
            target_layer = inner_model._conv_head
        elif hasattr(inner_model, "_blocks") and len(inner_model._blocks) > 0:
            target_layer = inner_model._blocks[-1]
        elif hasattr(inner_model, "layer4"):
            target_layer = inner_model.layer4[-1]

    if target_layer is None:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    try:
        output = model(img_tensor)
        class_idx = torch.argmax(output, dim=1)

        model.zero_grad()
        output[0, class_idx].backward()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    if not gradients or not activations:
        raw_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        return (image, raw_map) if return_raw else image

    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    activation = activations[0]

    cam = torch.relu((activation * grad).sum(dim=1)).squeeze().detach().cpu().numpy()
    
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap_rgb, 0.4, 0)

    if return_raw:
        return overlay, cam_resized

    return overlay
