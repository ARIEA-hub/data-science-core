"""
utils.py

Shared video I/O helpers for both the object detection and action
recognition pipelines:
  - frame extraction from a video file
  - bounding-box drawing for annotated output
  - a synthetic test video generator, used to verify pipeline mechanics
    without needing a real dataset (useful for CI / sandboxed dev, and
    as a quick smoke test before pointing the pipeline at real data)
"""

import os
import cv2
import numpy as np


def extract_frames(video_path: str, every_n: int = 1, max_frames: int | None = None):
    """Yields (frame_index, frame_bgr) from a video, sampling every `every_n` frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    idx = 0
    yielded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            yield idx, frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        idx += 1
    cap.release()


def draw_boxes(frame: np.ndarray, boxes, labels, scores, class_names, score_thresh: float = 0.5):
    """Draws bounding boxes + label/score text on a copy of the frame (BGR, in-place-safe)."""
    out = frame.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        name = class_names[label] if label < len(class_names) else str(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name} {score:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def generate_synthetic_video(path: str, n_frames: int = 60, width: int = 320, height: int = 240,
                              fps: int = 15) -> str:
    """
    Generates a short synthetic .mp4 with a moving colored rectangle and circle,
    purely for smoke-testing the detection/frame pipelines end-to-end without a
    real dataset. NOT meant to test detection accuracy - a pretrained COCO model
    won't recognize these shapes as anything meaningful. It only proves frame
    I/O, model forward pass, and video-writing all work correctly together.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        x = int((i / n_frames) * (width - 60))
        cv2.rectangle(frame, (x, 40), (x + 60, 100), (60, 180, 60), -1)
        cy = int(height / 2 + 40 * np.sin(i / 5))
        cv2.circle(frame, (width // 2, cy), 25, (60, 60, 200), -1)
        writer.write(frame)

    writer.release()
    return path


def frame_to_tensor(frame_bgr: np.ndarray):
    """BGR (OpenCV) -> RGB float tensor in [0,1], CHW, for torchvision detection models."""
    import torch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor
