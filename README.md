# CPS843-Computer-Vision-Project
# Vehicle Detector and Logger for Policing and Traffic Enforcement

This project implements an end-to-end computer vision pipeline for:

- Detecting and tracking vehicles in video
- Reading license plates (OCR)
- Classifying vehicle make, model, and color
- Logging all detections in structured CSV files
- Cross-referencing vehicles against a policing **hotlist** to raise alerts

It was developed as a final project for CPS843 (Toronto Metropolitan University).

---

## Features

- **Vehicle detection and tracking** using YOLO and simple multi-object tracking.
- **License plate OCR** using EasyOCR on cropped plate regions.
- **Make / model / color classification** using a custom classifier trained on Stanford Cars and related datasets.
- **Structured logging** to CSV with timestamps, bounding boxes, attributes, and evidence image paths.
- **Hotlist alerting** that matches plate + make + model + color against a police hotlist and outputs `alerts.csv`.

---

## Repository structure

A suggested layout:

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs
│   └── ufpr_alpr_plate.yaml
├── data_samples
│   ├── hotlist_sample.csv
│   ├── vehicle_events_example.csv
│   ├── vehicles_refined_example.csv
│   └── alerts_example.csv
├── logs
│   ├── raw/                # raw logs + crops (not versioned, except small samples)
│   └── refined/            # refined logs (not versioned, except small samples)
├── weights                 # model weights (not included; see below)
├── utils
│   ├── __init__.py
│   ├── detection_utils.py
│   ├── ocr_utils.py
│   ├── classifier_utils.py
│   └── logging_utils.py
├── run_vehicle_pipeline.py
├── refine_vehicle_logs.py
├── cross_reference_hotlist.py
├── train_car_classifier.py
├── eval_car_classifier.py
└── prepare_ufpr_to_yolo.py (optional)
