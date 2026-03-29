# backend/model.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = os.path.join("models", "detector.pth")

class AIDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

        if not os.path.exists(model_path):
            raise Exception("Trained model not found")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        print("✅ Loaded trained model")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image: Image.Image) -> float:
        image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            prob_real = torch.sigmoid(logits).item()

        # calibration
        prob_real = np.clip(prob_real, 0.1, 0.9)
        prob_real = (prob_real * 0.9) + 0.05
        prob_real = 0.8 * prob_real + 0.2 * 0.5

        return float(prob_real)

    def generate_heatmap(self, image: Image.Image):
        self.model.eval()

        image = image.convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        target_layer = self.model.features[-1]

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)

        output = self.model(img_tensor)
        loss = output[0]
        loss.backward()

        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        if cam.max() != 0:
            cam = cam - cam.min()
            cam = cam / cam.max()

        handle_f.remove()
        handle_b.remove()

        return cam