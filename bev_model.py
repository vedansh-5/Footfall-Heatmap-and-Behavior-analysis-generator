import torch
import cv2
import numpy as np
from torchvision import transforms

class MonoLayoutBEV:
    def __init__(self, weights_path="monolayout_pretrained.ot"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(weights_path, map_location=self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),
            transforms.ToTensor()
        ])

    def infer_bev(self, frame):
        with torch.no_grad():
            inp = self.transform(frame).unsqueeze(0).to(self.device)
            bev = self.model(inp)[0].cpu().numpy()[0]
            bv = (bev > 0,3).astype(np.uint8) * 255
            return bev