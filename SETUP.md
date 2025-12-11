# CPS843 Vehicle Project – Steps

This document is a big “everything in one place” dump of how to use the
scripts in this repo to:

- Download and prepare datasets
- Train plate detection and car classification models
- Evaluate OCR + classifier accuracy
- Run the vehicle detection + logging pipeline
- Simulate a police-style hotlist alert system
- Run batch experiments

You can trim / rearrange this as you like.

---

## 0. Files in this repo

These are the main scripts you’re shipping:

- `download_datasets.py`  
- `prepare_ufpr_alpr.py`  
- `build_ufpr_ground_truth_csv.py`  
- `make_ufpr_videos.sh`  
- `train_ufpr_alpr_yolov8.py`  
- `train_car_classifier.py`  
- `train_car_classifier_combined.py`  
- `eval_plate_ocr_accuracy.py`  
- `eval_car_classifier_ufpr.py`  
- `warmup_easyocr.py`  
- `vehicle_detection_pipeline.py`  
- `runtime_vehicle_logger.py`  
- `cross_reference_hotlist.py`  
- `batch_executor.sh`

Assume `steps.md` lives in the same folder as these files and you run
the commands from this folder.

---

## 1. Environment Setup

### 1.1 Create and activate a virtual environment

```bash
# from the project root
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install --upgrade pip

pip install \
  torch torchvision torchaudio \
  ultralytics \
  easyocr \
  opencv-python \
  pandas \
  numpy \
  scikit-learn \
  tqdm \
  matplotlib
