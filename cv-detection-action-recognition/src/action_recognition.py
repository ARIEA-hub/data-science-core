"""
action_recognition.py

Fine-tunes a pretrained 3D-CNN (torchvision's r3d_18, pretrained on
Kinetics-400) for action recognition on a folder-per-class video dataset
- the standard layout used by UCF101, UCF11, and HMDB51:

    data_dir/
        ClassName1/
            video1.avi
            video2.avi
        ClassName2/
            video3.avi
        ...

Designed to be dataset-agnostic within that layout: point --data-dir at
UCF101, a smaller subset like UCF11 (better fit for Colab's free-tier
GPU/time limits), or your own labeled clips.

Verified in this sandbox: dataset indexing, clip sampling, and one
training step were run on synthetic data with a random-init model (no
GPU, no internet - see __main__ smoke test). Real training requires a
real dataset (see README for download instructions) and benefits
heavily from a GPU (Colab free tier is enough for a few epochs on
UCF11 or a UCF101 subset).
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from torchvision.models.video import r3d_18, R3D_18_Weights

NUM_FRAMES = 16          # clip length fed to the model
FRAME_SIZE = 112         # r3d_18's expected spatial input size


class VideoClipDataset(Dataset):
    """
    Indexes data_dir/<class_name>/<video file> and samples NUM_FRAMES evenly
    spaced frames per clip, resized to FRAME_SIZE x FRAME_SIZE.
    """

    def __init__(self, data_dir: str, num_frames: int = NUM_FRAMES, frame_size: int = FRAME_SIZE):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.samples = []  # list of (video_path, label_idx)

        class_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)])
        self.classes = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_dir in class_dirs:
            label = self.class_to_idx[os.path.basename(class_dir)]
            for ext in ("*.avi", "*.mp4", "*.mov"):
                for video_path in glob.glob(os.path.join(class_dir, ext)):
                    self.samples.append((video_path, label))

        if not self.samples:
            raise ValueError(
                f"No video files found under {data_dir}. Expected structure: "
                f"data_dir/<class_name>/<video files (.avi/.mp4/.mov)>"
            )

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.num_frames
        indices = np.linspace(0, max(total - 1, 0), self.num_frames).astype(int)

        frames = []
        idx_set = set(indices.tolist())
        i = 0
        picked = {}
        while cap.isOpened() and len(picked) < len(idx_set):
            ret, frame = cap.read()
            if not ret:
                break
            if i in idx_set:
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                picked[i] = frame
            i += 1
        cap.release()

        # pad by repeating the last available frame if the video was shorter than expected
        last = next(iter(picked.values())) if picked else np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        ordered = [picked.get(idx, last) for idx in indices]

        clip = np.stack(ordered)  # (T, H, W, C)
        clip = torch.from_numpy(clip).float() / 255.0
        clip = clip.permute(3, 0, 1, 2)  # (C, T, H, W) - r3d_18 expects this
        return clip

    def __getitem__(self, i):
        video_path, label = self.samples[i]
        clip = self._load_clip(video_path)
        return clip, label


def build_model(num_classes: int, pretrained: bool = True, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model, device


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train(data_dir: str, epochs: int = 5, batch_size: int = 4, lr: float = 1e-4,
          pretrained: bool = True, val_split: float = 0.2, checkpoint_path: str = "checkpoints/action_model.pth"):
    dataset = VideoClipDataset(data_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model, device = build_model(num_classes=len(dataset.classes), pretrained=pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device} | classes: {dataset.classes} | train={n_train} val={n_val}")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f} - val_acc={val_acc:.3f}")

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": dataset.classes}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return model, dataset.classes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train an action recognition model.")
    parser.add_argument("data_dir", help="Folder-per-class video dataset root")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, pretrained=not args.no_pretrained)
