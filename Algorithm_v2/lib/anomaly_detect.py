# lib/anomaly_detect.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from collections import deque
import cv2
import os
from model.autoencoder import convAE
from utils1 import psnr

class AnomalyDetector:
    def __init__(self, seq_len=16, frame_idx=8, resize=(256, 256), device=None):
        self.SEQ_LEN = seq_len
        self.FRAME_IDX = frame_idx
        self.RESIZE_H, self.RESIZE_W = resize
        self.DEVICE = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../model/anormal/avenue8535.pth")
        )
        self.model = convAE()
        self.model = nn.DataParallel(self.model)
        checkpoint = torch.load(model_path)
        try:
            self.model.load_state_dict(checkpoint['model'].state_dict())
        except KeyError:
            self.model.load_state_dict(checkpoint['model_statedict'])
        self.model.to(self.DEVICE).eval()

        # 初始化帧缓冲
        self.frame_buffer = deque(maxlen=self.SEQ_LEN)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def update_and_detect(self, frame_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.RESIZE_W, self.RESIZE_H))
        tensor = self.transform(gray).to(self.DEVICE)  # 1xHxW
        self.frame_buffer.append(tensor)

        if len(self.frame_buffer) < self.SEQ_LEN:
            return {"ready": False, "score": None}

        seq = torch.stack(list(self.frame_buffer), dim=1).unsqueeze(0)  # [1,C,T,H,W]

        with torch.no_grad():
            outputs = self.model(seq)

        recon = outputs[0, :, self.FRAME_IDX]
        orig = seq[0, :, self.FRAME_IDX]
        mse_map = ((recon - orig) ** 2).mean(dim=0).cpu().numpy()
        mse_mean = mse_map.mean()
        score = psnr(mse_mean)  # 例如越小越异常

        return {"ready": True, "score": float(score)}
