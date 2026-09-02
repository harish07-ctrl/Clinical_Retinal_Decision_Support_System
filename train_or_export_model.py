import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from retinal_model import RetinalModel

def train_and_export_fast():
    print("[Export] Building fast-calibrated Diabetic Retinopathy models...")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Classes and samples
    samples = [
        ("sample_images/sample_no_dr.jpg", 0),
        ("sample_images/sample_normal_fundus.jpg", 0),
        ("sample_images/sample_mild_dr.jpg", 1),
        ("sample_images/sample_moderate_dr.jpg", 2),
        ("sample_images/sample_severe_dr.jpg", 3),
        ("sample_images/sample_proliferative_dr.jpg", 4)
    ]
    severity_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    aug_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.08, contrast=0.08),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    raw_images = []
    raw_labels = []

    for path, label in samples:
        if os.path.exists(path):
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            raw_images.append(transform(img_rgb))
            raw_labels.append(label)
            for _ in range(15):
                raw_images.append(aug_transform(img_rgb))
                raw_labels.append(label)

    X = torch.stack(raw_images)
    y = torch.tensor(raw_labels, dtype=torch.long)
    print(f"[Export] Prepared {len(X)} sample images across 5 classes.")

    # -------------------------------------------------------------
    # 2. EfficientNet-B0 Model Calibration
    # -------------------------------------------------------------
    print("\n[Export] 1/2 Calibrating EfficientNet-B0...")
    eff_model = RetinalModel(n_classes=5, backbone="efficientnet-b0")
    
    # Extract features using extract_features
    eff_model.eval()
    with torch.no_grad():
        feats = eff_model.model.extract_features(X)
        feats = eff_model.model._avg_pooling(feats)
        if eff_model.model._global_params.include_top:
            feats = feats.flatten(start_dim=1)
            feats = eff_model.model._dropout(feats)

    # Train linear head
    fc = eff_model.model._fc
    optimizer = torch.optim.AdamW(fc.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(120):
        optimizer.zero_grad()
        logits = fc(feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    eff_model.eval()
    print("[Export] EfficientNet-B0 test evaluation:")
    for path, exp_lbl in samples:
        if os.path.exists(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            inp = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = eff_model(inp)
                probs = torch.softmax(out, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                print(f"  - {path:40s} | Exp: {severity_names[exp_lbl]:14s} | Pred: {severity_names[pred_idx]:14s} | Conf: {probs[pred_idx]*100:5.1f}%")

    # Save dr_model.pth
    torch.save({
        "model_state_dict": eff_model.state_dict(),
        "backbone": "efficientnet-b0",
        "n_classes": 5,
        "classes": severity_names
    }, "dr_model.pth")
    print(f"[Export] Successfully saved dr_model.pth ({os.path.getsize('dr_model.pth')/(1024*1024):.2f} MB)")

    # -------------------------------------------------------------
    # 3. ResNet-18 Model Calibration
    # -------------------------------------------------------------
    print("\n[Export] 2/2 Calibrating ResNet-18...")
    resnet_model = RetinalModel(n_classes=5, backbone="resnet18")
    
    # Extract features before fc
    resnet_model.eval()
    modules = list(resnet_model.model.children())[:-1]
    resnet_feat_extractor = nn.Sequential(*modules)
    
    with torch.no_grad():
        rn_feats = resnet_feat_extractor(X)
        rn_feats = torch.flatten(rn_feats, 1)

    rn_fc = resnet_model.model.fc
    rn_opt = torch.optim.AdamW(rn_fc.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(120):
        rn_opt.zero_grad()
        logits = rn_fc(rn_feats)
        loss = criterion(logits, y)
        loss.backward()
        rn_opt.step()

    resnet_model.eval()
    print("[Export] ResNet-18 test evaluation:")
    for path, exp_lbl in samples:
        if os.path.exists(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            inp = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = resnet_model(inp)
                probs = torch.softmax(out, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                print(f"  - {path:40s} | Exp: {severity_names[exp_lbl]:14s} | Pred: {severity_names[pred_idx]:14s} | Conf: {probs[pred_idx]*100:5.1f}%")

    torch.save({
        "model_state_dict": resnet_model.state_dict(),
        "backbone": "resnet18",
        "n_classes": 5,
        "classes": severity_names
    }, "dr_resnet18.pth")
    print(f"[Export] Successfully saved dr_resnet18.pth ({os.path.getsize('dr_resnet18.pth')/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    train_and_export_fast()
