# Traffic Monitoring (YOLO + Video Frame Sampling)

This repo contains **code only** (no large videos/images) for a traffic monitoring pipeline:
- Stratified frame sampling from videos (OpenCV)
- Train/val/test splitting **by video** (prevents leakage)
- YOLO training & evaluation using **Ultralytics** (recommended in Google Colab)

## Recommended workflow (Drive + Colab + GitHub)
1. Store **videos + frames + annotations** in Google Drive (or your university GPU sandbox storage).
2. Keep **scripts + docs** here in GitHub.
3. In Colab: mount Drive, `git clone` this repo, run scripts to sample/split/train.

## Quick start (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-username>/traffic-monitoring.git
%cd traffic-monitoring
!pip install -r requirements.txt
```

### 1) Sample frames (recommended: stratified 20 frames per video)
```bash
python sampling/sample_stratified.py   --video_dir "/content/drive/MyDrive/traffic_project/videos"   --out_dir "/content/drive/MyDrive/traffic_project/sampled_frames"   --per_video 20
```

### 2) Create train/val/test split lists by VIDEO IDs
```bash
python splits/make_splits.py   --frames_root "/content/drive/MyDrive/traffic_project/sampled_frames"   --out_dir "/content/drive/MyDrive/traffic_project/splits"   --train M01 M02 M03 M04 M05 M06 M07 A01 A02 A03 A04 A05 A06 A07 E01 E02 E03 E04   --val M08 M09 A08 E05 E06   --test M10 A09
```

### 3) Prepare Ultralytics dataset folders after exporting annotations in YOLO format
```bash
python utils/prepare_yolo_dataset.py   --annotations_images_dir "/content/drive/MyDrive/traffic_project/annotations/images"   --annotations_labels_dir "/content/drive/MyDrive/traffic_project/annotations/labels"   --splits_dir "/content/drive/MyDrive/traffic_project/splits"   --dataset_out "/content/drive/MyDrive/traffic_project/dataset_yolo"
```
### 4) Train YOLO model
```bash
python training/train.py --data configs/data.yaml --project training --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --device 0
```


## Notes
- Keep large files out of GitHub. Use Drive (or sandbox storage).
- `.gitignore` included.
