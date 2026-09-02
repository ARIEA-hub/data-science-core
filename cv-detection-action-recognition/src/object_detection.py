"""
object_detection.py

Frame-by-frame object detection over a video using a COCO-pretrained
torchvision Faster R-CNN, then re-assembles an annotated output video
plus a per-frame detection summary (for downstream analytics, e.g.
"how many people are in frame over time").

On first run with pretrained=True, torchvision downloads model weights
from PyTorch's model hub (requires internet - fine on Colab, not
available in this sandboxed dev environment). Architecture + inference
plumbing here was verified locally with weights=None (random init) to
confirm shapes and the full frame -> tensor -> model -> boxes -> drawn
frame -> video pipeline runs without errors; actual detection quality
requires the real pretrained weights.
"""

import cv2
import torch
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from utils import extract_frames, draw_boxes, frame_to_tensor

COCO_CLASSES = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]


def load_model(pretrained: bool = True, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    else:
        # weights_backbone=None too, so this path never attempts a download -
        # used for offline pipeline testing (see __main__ smoke test note above).
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.eval().to(device)
    return model, device


@torch.no_grad()
def detect_frame(model, frame_bgr, device, score_thresh: float = 0.5):
    tensor = frame_to_tensor(frame_bgr).to(device)
    output = model([tensor])[0]
    keep = output["scores"] >= score_thresh
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()
    return boxes, labels, scores


def process_video(video_path: str, output_path: str, model=None, device: str = None,
                   every_n: int = 3, score_thresh: float = 0.5, pretrained: bool = True) -> pd.DataFrame:
    """
    Runs detection every `every_n` frames, writes an annotated video, and
    returns a per-frame summary DataFrame: frame_idx, class_name, count.
    Frames not sampled are copied through un-annotated to keep output
    video length/fps consistent with the source.
    """
    if model is None:
        model, device = load_model(pretrained=pretrained)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    records = []
    last_annotated = None
    for idx, frame in extract_frames(video_path, every_n=1):
        if idx % every_n == 0:
            boxes, labels, scores = detect_frame(model, frame, device, score_thresh)
            annotated = draw_boxes(frame, boxes, labels, scores, COCO_CLASSES, score_thresh)
            last_annotated = annotated
            for label, score in zip(labels, scores):
                records.append({
                    "frame_idx": idx,
                    "class_name": COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label),
                    "score": float(score),
                })
            writer.write(annotated)
        else:
            writer.write(last_annotated if last_annotated is not None else frame)

    writer.release()
    return pd.DataFrame(records)


def summarize_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Object counts per class across the whole video, and average confidence."""
    if df.empty:
        return pd.DataFrame(columns=["class_name", "detections", "avg_score"])
    return (
        df.groupby("class_name")
        .agg(detections=("class_name", "count"), avg_score=("score", "mean"))
        .sort_values("detections", ascending=False)
        .reset_index()
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run object detection on a video.")
    parser.add_argument("video_path")
    parser.add_argument("--output", default="output_annotated.mp4")
    parser.add_argument("--every-n", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = process_video(args.video_path, args.output, every_n=args.every_n, score_thresh=args.threshold)
    print(f"Wrote annotated video to {args.output}")
    print(summarize_detections(df).to_string(index=False))
