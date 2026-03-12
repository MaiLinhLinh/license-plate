# import os
# import random
# import shutil

# # ===== CONFIG =====
# RAW_DATASET = "dataset_raw"
# OUTPUT_DATASET = "dataset"
# TRAIN_RATIO = 0.8
# SEED = 42
# # ==================

# random.seed(SEED)

# image_dir = os.path.join(RAW_DATASET, "images")
# label_dir = os.path.join(RAW_DATASET, "labels")

# train_img_dir = os.path.join(OUTPUT_DATASET, "images/train")
# val_img_dir   = os.path.join(OUTPUT_DATASET, "images/val")
# train_lbl_dir = os.path.join(OUTPUT_DATASET, "labels/train")
# val_lbl_dir   = os.path.join(OUTPUT_DATASET, "labels/val")

# for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
#     os.makedirs(d, exist_ok=True)

# images = [f for f in os.listdir(image_dir)
#           if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# random.shuffle(images)

# split_idx = int(len(images) * TRAIN_RATIO)
# train_images = images[:split_idx]
# val_images   = images[split_idx:]

# def move_files(image_list, img_dst, lbl_dst):
#     for img in image_list:
#         name = os.path.splitext(img)[0]
#         img_src = os.path.join(image_dir, img)
#         lbl_src = os.path.join(label_dir, name + ".txt")

#         if not os.path.exists(lbl_src):
#             print(f"⚠️ Missing label for {img}, skipped")
#             continue

#         shutil.copy(img_src, img_dst)
#         shutil.copy(lbl_src, lbl_dst)

# move_files(train_images, train_img_dir, train_lbl_dir)
# move_files(val_images, val_img_dir, val_lbl_dir)

# train_label_count = len(os.listdir(train_lbl_dir))
# val_label_count   = len(os.listdir(val_lbl_dir))

# print(f"✅ Done!")
# print(f"Train: {len(train_images)} images| {train_label_count} labels")
# print(f"Val:   {len(val_images)} images| {val_label_count} labels")

from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model on the COCO8 example dataset for 80 epochs
results = model.train(data="my_data.yaml", epochs=80, imgsz=640)


