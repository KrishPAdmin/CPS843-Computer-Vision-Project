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

## Repository contents

### Main scripts (this folder)

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

### Example directory layout (outside of this folder)

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
```
---
## Results of Experiments

### Number Plate accuracy (plate_conf)
| min `plate_conf` | # preds kept | strict exact-match | lenient (I/L/1 & O/0/Q) |
| ---------------- | ------------ | ------------------ | ----------------------- |
| 0.0 (all)        | 126          | **25.4%**          | **30.2%**               |
| 0.2              | 85           | 37.6%              | 43.5%                   |
| 0.3              | 67           | 46.3%              | 52.2%                   |
| 0.4              | 51           | 49.0%              | 56.9%                   |
| 0.5              | 34           | **58.8%**          | **64.7%**               |
| 0.6              | 25           | 60.0%              | 68.0%                   |
| 0.7              | 16           | 68.8%              | 75.0%                   |
| 0.8              | 10           | 80.0%              | 80.0%                   |

### Make+Model accuracy vs plate_conf

| min `plate_conf` | # preds kept | strict exact-match |
| ---------------- | ------------ | ------------------ |
| 0.0 (all) | 126 | 4.8% |
| 0.2 | 85 | 7.1% |
| 0.3 | 67 | 7.5% |
| 0.4 | 51 | 9.8% |
| 0.5 | 34 | 8.8% |
| 0.6 | 25 | 12.0% |
| 0.7 | 16 | 12.5% |
| 0.8 | 10 | 10.0% |

### Vehicle type accuracy vs plate_conf

| min `plate_conf` | # preds kept | strict exact-match |
| ---------------- | ------------ | ------------------ |
| 0.0 (all) | 126 | 56.3% |
| 0.2 | 85 | 57.6% |
| 0.3 | 67 | 61.2% |
| 0.4 | 51 | 58.8% |
| 0.5 | 34 | 64.7% |
| 0.6 | 25 | 60.0% |
| 0.7 | 16 | 62.5% |
| 0.8 | 10 | 60.0% |

### Color stability vs plate_conf (clip-level)

> strict = mode_ratio ≥ 0.70  |  lenient = mode_ratio ≥ 0.60

| min `plate_conf` | # preds kept | strict stable clips |
| ---------------- | ------------ | ------------------- |
| 0.0 (all) | 126 | 96.0% |
| 0.2 | 85 | 97.4% |
| 0.3 | 67 | 98.4% |
| 0.4 | 51 | 97.9% |
| 0.5 | 34 | 96.9% |
| 0.6 | 25 | 100.0% |
| 0.7 | 16 | 100.0% |
| 0.8 | 10 | 100.0% |



### Acknowledgements/References:
[1] R. Laroca et al., “A robust real-time automatic license plate recognition based on the YOLO detector,” in Proc. Int. Joint Conf. Neural Networks (IJCNN), Rio de Janeiro, Brazil, 2018, doi: 10.1109/IJCNN.2018.8489629.

[2] J. Krause, J. Deng, M. Stark, and L. Fei-Fei, “Collecting a large-scale dataset of fine-grained cars,” in Proc. 1st IEEE Workshop on Fine-Grained Visual Classification (FGVC2), in conjunction with IEEE Conf. Computer Vision and Pattern Recognition (CVPR), Portland, OR, USA, 2013.

[3] UFPR Vision, Robotics and Imaging Lab, “UFPR-ALPR Dataset,” Federal University of Paraná, Brazil. [Online]. Available: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/.

[4] Stanford Vision Lab, “Stanford Cars Dataset.” [Online]. Available: https://ai.stanford.edu/~jkrause/cars/car_dataset.html. 

[5] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” GitHub repository. [Online]. Available: https://github.com/ultralytics/ultralytics.

[6] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 2019.

[7] G. Bradski, “The OpenCV library,” Dr. Dobb’s Journal of Software Tools, 2000.

[8] JaidedAI, “EasyOCR,” GitHub repository. [Online]. Available: https://github.com/JaidedAI/EasyOCR.

[9] OpenAI, “ChatGPT” [Large language model]. [Online]. Available: https://chat.openai.com/
