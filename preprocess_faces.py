# -*- coding: utf-8 -*-
"""
Preprocessing Script for Deepfake Detector
Extracts faces and caches them as torch tensors (.pt)
Speeds up training by 5–10x on subsequent runs
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from facenet_pytorch import MTCNN
from torchvision import transforms

# ==========================
# CONFIG
# ==========================
RAW_REAL_DIR = r"C:\Users\ASUS\OneDrive\Capstone Project\Facenet Code\original_sequences-20251018T143304Z-1-001\original_sequences\youtube\c23\videos"
RAW_FAKE_DIR = r"C:\Users\ASUS\OneDrive\Capstone Project\Facenet Code\manipulated_sequences-20251018T143143Z-1-001\manipulated_sequences\Deepfakes\c23\videos"

PROCESSED_ROOT = r"C:\Users\ASUS\OneDrive\Capstone Project\Facenet Code\processed_faces"
N_FRAMES = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================
# Setup
# ==========================
os.makedirs(os.path.join(PROCESSED_ROOT, "real"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_ROOT, "fake"), exist_ok=True)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mtcnn = MTCNN(keep_all=False, device=DEVICE)

def extract_and_save(video_path, save_path, n_frames=N_FRAMES):
    """Extracts up to N_FRAMES faces and saves as .pt tensor."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"⚠️ Skipping empty video: {video_path}")
        return False

    idxs = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames, count = [], 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in idxs:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(frame_rgb)
            if face is not None:
                face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                frames.append(transform(face_np))
            count += 1
        frame_id += 1
    cap.release()

    # pad if fewer than n_frames
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))

    torch.save(torch.stack(frames[:n_frames]), save_path)
    return True

def process_folder(src_dir, dst_dir):
    videos = list(Path(src_dir).glob("*.mp4"))
    print(f"📹 Found {len(videos)} videos in {src_dir}")
    for vid in tqdm(videos, desc=f"Processing {os.path.basename(src_dir)}"):
        out_path = os.path.join(dst_dir, Path(vid).stem + ".pt")
        if os.path.exists(out_path):
            continue  # skip if already processed
        try:
            extract_and_save(str(vid), out_path)
        except Exception as e:
            print(f"❌ Error processing {vid}: {e}")

def main():
    print(f"🚀 Starting preprocessing on {DEVICE}")
    process_folder(RAW_REAL_DIR, os.path.join(PROCESSED_ROOT, "real"))
    process_folder(RAW_FAKE_DIR, os.path.join(PROCESSED_ROOT, "fake"))
    print("✅ Preprocessing complete! Faces cached successfully.")

if __name__ == "__main__":
    main()
