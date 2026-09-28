# Muzzle Biometric Module (FYP Phase 1)

Pipeline: YOLOv8n (muzzle detection) -> crop -> ResNet18 + ArcFace (512-d embedding) -> cosine similarity -> SHA-256 biometric hash.

## Steps
1. Place zips in `data/raw/` as `cow-muzzle-dataset.zip` and `beefcattle-muzzle.zip`.
2. `python scripts/prepare_yolo_data.py --zip data/raw/cow-muzzle-dataset.zip`
3. `python scripts/train_yolo.py --epochs 50`
4. `python scripts/prepare_arcface_data.py --zip data/raw/beefcattle-muzzle.zip --crop`
5. `python scripts/train_arcface.py --epochs 40`
6. `python app.py`  -> open http://127.0.0.1:7860
