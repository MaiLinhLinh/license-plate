Dataset + Slide: https://drive.google.com/drive/folders/1-Isq3XqpZbdXc8rrvtC33V3Jg0YHOimv?usp=drive_link
# License Plate Detection - Clean Deployment

Portable Flask app for license plate detection using YOLOv8 + TrOCR.

## Structure
```
license-plate-clean/
├── app.py                          # Flask backend
├── index.html                      # Frontend UI
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/
│   ├── yolo_model/
│   │   └── best.pt                 # YOLOv8 detection model
│   └── trocr_license_plate_finetuned-20260429T192651Z-3-002/
│       └── trocr_license_plate_finetuned/  # TrOCR OCR model
└── temp_uploads/                   # Temporary upload folder (auto-created)
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run App
```bash
python app.py
```

Server will start at: **http://127.0.0.1:5000**

### 3. Use
- Open browser to http://127.0.0.1:5000
- Upload an image with a license plate
- View detection results

## Requirements
- Python 3.8+
- PyTorch (CPU or CUDA)
- 2GB+ RAM for models

## Notes
- Models are included - no download needed
- temp_uploads folder is auto-created
- Works offline after dependencies installed
- Supports: JPG, JPEG, PNG, BMP (max 10MB)

