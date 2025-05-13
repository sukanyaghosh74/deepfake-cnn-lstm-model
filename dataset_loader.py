# dataset_loader.py

import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from cnn_extractor import CNNFeatureExtractor

class DeepfakeDataset(Dataset):
    def __init__(self, video_dirs, labels, num_frames=30):
        self.video_dirs = video_dirs
        self.labels = labels
        self.num_frames = num_frames
        self.cnn = CNNFeatureExtractor().eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = list(range(0, total_frames, max(1, total_frames // self.num_frames)))
        frames = []

        for idx in frame_idxs[:self.num_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()
        return torch.stack(frames)

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]
        frames = self.extract_frames(video_path)
        features = torch.stack([self.cnn(f.unsqueeze(0)).squeeze() for f in frames])
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return features, label
