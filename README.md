# CPS843-Computer-Vision-Project
# Vehicle Detector and Logger for Policing and Traffic Enforcement

This project implements an end-to-end computer vision pipeline for:

- Detecting and tracking vehicles in video
- Reading license plates with OCR
- Classifying vehicle make, model, and color
- Logging detections in structured CSV files
- Cross-referencing detected vehicles against a **police hotlist** to trigger alerts

Developed as a final project for **CPS843** (Toronto Metropolitan University).

---

## 1. Repository contents

### 1.1 Main scripts (this folder)

These are the scripts included in this repo and what each one does.

#### Dataset & preparation

- **`download_datasets.py`**  
  Helper script to download / fetch required datasets (e.g., UFPR-ALPR, Stanford Cars, etc.).  
  Run with `python download_datasets.py --help` to see available options.

- **`prepare_ufpr_alpr.py`**  
  Converts the UFPR-ALPR dataset into the format needed by YOLO and the rest of this project  
  (e.g., YOLO txt labels, consistent folder layout).

- **`build_ufpr_ground_truth_csv.py`**  
  Builds a **ground-truth CSV** for UFPR plates (plate text, frame index, bounding boxes…) used
  for later evaluation of plate OCR accuracy.

- **`make_ufpr_videos.sh`**  
  Shell script to generate or re-encode UFPR videos / tracks into a consistent video format
  (e.g., `.mp4`) to be used as input for detection and logging.

#### Training scripts

- **`train_ufpr_alpr_yolov8.py`**  
  Trains a YOLOv8 model for **license plate detection** on the UFPR-ALPR dataset, using the
  YOLO-formatted labels produced by `prepare_ufpr_alpr.py`.

- **`train_car_classifier.py`**  
  Trains a car make/model (and possibly color) classifier on a base dataset such as **Stanford Cars**.

- **`train_car_classifier_combined.py`**  
  Trains or fine-tunes a **combined** car classifier that mixes multiple datasets  
  (e.g., Stanford Cars + UFPR vehicle crops) for better performance on this project’s data.

#### Evaluation scripts

- **`eval_plate_ocr_accuracy.py`**  
  Evaluates the **license plate OCR** (EasyOCR + pipeline) against UFPR ground truth created by  
  `build_ufpr_ground_truth_csv.py`. Outputs accuracy metrics (e.g., exact match rate, edit distance).

- **`eval_car_classifier_ufpr.py`**  
  Evaluates the **car classifier** on UFPR or UFPR-like crops, using the trained weights from
  `train_car_classifier.py` / `train_car_classifier_combined.py`. Prints accuracy / confusion matrix.

#### Runtime / pipeline scripts

- **`warmup_easyocr.py`**  
  Simple script to “warm up” EasyOCR (load model weights, cache, etc.) so that the main pipeline
  can start faster and avoid first-run latency.

- **`vehicle_detection_pipeline.py`**  
  Main **offline pipeline** for processing videos:
  - Detects vehicles and plates (YOLO)
  - Tracks them over time
  - Runs plate OCR (EasyOCR)
  - Runs car make/model/color classification
  - Writes structured logs (CSV) and saves crops for each vehicle

- **`runtime_vehicle_logger.py`**  
  A more **runtime-oriented logger** intended to simulate a police in-car system.  
  Uses the same detection / OCR / classification pipeline but is focused on logging events in real-time
  (or near real-time) for a live video stream or a looped set of videos.

- **`cross_reference_hotlist.py`**  
  Takes a **refined vehicle log** and a **hotlist CSV** (vehicles / owners of interest) and
  produces `alerts.csv` with matched vehicles, reasons, and priority levels.

#### Automation / batch helpers

- **`batch_executor.sh`**  
  Shell script to run a sequence of experiments / evaluations automatically  
  (e.g., loop over videos, thresholds, or models and call the Python scripts above).

---

### 1.2 Example directory layout (outside of this folder)

This repo assumes a structure roughly like:

```text
project_root/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/                     # (this folder – where the files above live)
├── configs/
│   └── ufpr_alpr_plate.yaml     # YOLO dataset config for UFPR-ALPR
├── data/
│   ├── UFPR-ALPR/               # downloaded UFPR-ALPR dataset
│   └── stanford_cars/           # downloaded Stanford Cars dataset
├── logs/
│   ├── raw/                     # raw logs + crops from pipeline
│   └── refined/                 # refined logs used for hotlist / evaluation
├── weights/
│   ├── car_classifier.pth
│   ├── car_classifier_combined.pth
│   └── ufpr_alpr_yolov8.pt
└── data_samples/
    ├── hotlist_sample.csv
    ├── vehicle_events_example.csv
    ├── vehicles_refined_example.csv
    └── alerts_example.csv
