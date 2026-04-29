"""
Flask backend for license plate detection.
YOLOv8 detection + TrOCR OCR.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import os
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = Path(__file__).parent
YOLO_PATH = BASE_DIR / "models" / "yolo_model" / "best.pt"
TROCR_PATH = BASE_DIR / "models" / "trocr_license_plate_finetuned-20260429T192651Z-3-002" / "trocr_license_plate_finetuned"
UPLOAD_FOLDER = BASE_DIR / "temp_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ============================================================================
# APP SETUP
# ============================================================================
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ============================================================================
# MODEL LOADING (at startup)
# ============================================================================
print("Loading YOLOv8 model...")
yolo_model = YOLO(str(YOLO_PATH))
print("YOLOv8 loaded.")

print("Loading TrOCR model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_processor = TrOCRProcessor.from_pretrained(str(TROCR_PATH))
trocr_model = VisionEncoderDecoderModel.from_pretrained(str(TROCR_PATH))
trocr_model.to(device)
trocr_model.eval()
print(f"TrOCR loaded on {device}.")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def detect_plates(image_path):
    """Run YOLOv8 detection, return all bounding boxes with confidence and annotated image."""
    results = yolo_model(image_path, verbose=False)
    plates = []
    annotated_img = None

    if results and len(results) > 0:
        # Generate annotated image with bounding boxes
        annotated_array = results[0].plot()  # Returns numpy array (BGR format)
        # Convert BGR to RGB
        annotated_img = Image.fromarray(annotated_array[..., ::-1])

        # Extract bounding box information
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())  # confidence score
                plates.append((xyxy, conf))

    return plates, annotated_img  # Return both bboxes and annotated image

def crop_plate(image, box):
    """Crop license plate region from image."""
    x1, y1, x2, y2 = map(int, box)
    return image.crop((x1, y1, x2, y2))

def ocr_plate(crop_img):
    """Run TrOCR on cropped plate image."""
    pixel_values = trocr_processor(images=crop_img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
        text = trocr_processor.decode(generated_ids[0], skip_special_tokens=True)
    return text.strip()

def validate_file(file):
    """Validate uploaded file."""
    if not file:
        return None, "No file provided"
    filename = file.filename
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return None, f"Invalid file type: {ext}"
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return None, f"File too large: {size/1024/1024:.1f}MB"

    # Validate file content by checking if it's a valid image
    try:
        file.seek(0)
        from PIL import Image
        img = Image.open(file)
        img.verify()
        file.seek(0)
    except Exception:
        return None, "Invalid image file"

    return filename, None

# ============================================================================
# ROUTES
# ============================================================================
@app.route("/")
def index():
    """Serve frontend."""
    return send_from_directory(".", "index.html")

@app.route("/api/detect", methods=["POST"])
def detect():
    """Detect and OCR license plate from uploaded image."""
    temp_path = None
    try:
        # Validate file
        file = request.files.get("image")
        filename, error = validate_file(file)
        if error:
            return jsonify({"error": error}), 400

        # Save temp file
        temp_path = UPLOAD_FOLDER / f"upload_{os.urandom(4).hex()}{Path(filename).suffix}"
        file.save(str(temp_path))

        # Load image
        image = Image.open(temp_path).convert("RGB")

        # Detect plates and get annotated image
        plates_data, annotated_img = detect_plates(temp_path)
        if not plates_data:
            return jsonify({"error": "No license plate detected"}), 404

        # Convert annotated image to base64
        import base64
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_data_url = f"data:image/jpeg;base64,{img_base64}"

        # Process each plate
        results = []
        for bbox, conf in plates_data:
            crop = crop_plate(image, bbox)
            plate_text = ocr_plate(crop)
            x1, y1, x2, y2 = map(int, bbox.tolist())
            results.append({
                "text": plate_text,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": round(conf, 3)
            })

        return jsonify({
            "success": True,
            "count": len(results),
            "plates": results,
            "annotated_image": img_data_url,
            "model": "yolo-v8-trocr"
        })

    except Exception as e:
        # Log error for debugging (don't expose to client)
        print(f"Error in detect(): {str(e)}")
        return jsonify({"error": "Processing failed. Please try again."}), 500
    finally:
        # Clean up temp file in all cases
        if temp_path and temp_path.exists():
            temp_path.unlink()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
