# Computer Vision: Object Detection + Action Recognition (PyTorch)

Two complementary video CV pipelines, both PyTorch/torchvision-based, both
verified for plumbing correctness on CPU with synthetic data and designed to
run for real on **Google Colab with a GPU runtime**.

## 1. Object Detection

`src/object_detection.py` — COCO-pretrained Faster R-CNN, run frame-by-frame
over any video. Produces an annotated output video (bounding boxes + labels)
and a per-frame/per-class detection summary DataFrame.

```python
from object_detection import process_video, summarize_detections
df = process_video("my_video.mp4", "annotated.mp4", every_n=3, score_thresh=0.5)
summarize_detections(df)
```

## 2. Action Recognition

`src/action_recognition.py` — fine-tunes a Kinetics-pretrained `r3d_18` 3D-CNN
on a folder-per-class video dataset (the layout used by UCF101, UCF11, and
HMDB51):

```
data_dir/
    ClassA/
        clip1.avi
    ClassB/
        clip2.avi
```

```python
from action_recognition import train
model, classes = train("data/ucf11", epochs=5, batch_size=8)
```

### Recommended dataset: UCF11, not full UCF101

Full UCF101 (~13,000 clips, ~7GB) will eat Colab's free-tier time/disk limits
fast. **UCF11** (~1,600 clips, 11 classes) trains in a reasonable number of
epochs on a free Colab GPU and is available via Kaggle — see
`notebooks/run_on_colab.ipynb` for the exact download cell (needs a free
Kaggle account + API token). Swap in full UCF101 or HMDB51 later using the
same folder-per-class layout if you want a harder benchmark.

## Run it

**Recommended: `notebooks/run_on_colab.ipynb`** — clones the repo, installs
deps, runs the object detection demo, walks through the UCF11 download, and
includes an optional synthetic-data smoke test so you can confirm the
training loop works before committing to a full dataset download.

Locally (no GPU needed for object detection; action recognition training
will be slow on CPU):

```bash
pip install -r requirements.txt
python src/object_detection.py my_video.mp4 --output annotated.mp4
python src/action_recognition.py data/ucf11 --epochs 5
```

## What was and wasn't verified in this sandbox

This was built in a dev sandbox with **no GPU and no internet access to
`download.pytorch.org`** (pretrained weight downloads), so I could not run
real detection/classification here. What *was* verified end-to-end on CPU,
using synthetic test videos and random-initialized (`pretrained=False`)
models to isolate plumbing from model quality:

- Frame extraction, box-drawing, and video writing (`src/utils.py`)
- Full object-detection pipeline: frame → tensor → model forward pass →
  boxes → annotated video → summary DataFrame (600 detection records
  produced on a 30-frame synthetic clip)
- Full action-recognition pipeline: dataset indexing → clip sampling
  (confirmed tensor shape `(3, 16, 112, 112)`) → one complete
  `train_one_epoch` + `evaluate` cycle on a tiny synthetic 2-class dataset

**Before treating this as done**, run it on Colab with `pretrained=True` and
a real dataset, and sanity-check actual detection/classification quality —
architecture correctness doesn't guarantee good results on real data.
