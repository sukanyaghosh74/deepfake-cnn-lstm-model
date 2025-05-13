# utils.py

import cv2
import os

def extract_frames(video_path, out_dir, num_frames=30):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = list(range(0, total, max(1, total // num_frames)))

    for i, idx in enumerate(idxs[:num_frames]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()
