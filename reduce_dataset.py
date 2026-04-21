"""
CropWise AI – Dataset Reduction Script
Creates a balanced subset from the full PlantVillage dataset.
"""
import os
import random
import shutil

SOURCE_DIR      = "dataset/PlantVillage"
TARGET_DIR      = "dataset_reduced"
NUM_CLASSES     = 10
IMAGES_PER_CLS  = 150

os.makedirs(TARGET_DIR, exist_ok=True)

all_classes = sorted(os.listdir(SOURCE_DIR))
selected_classes = all_classes[:10]

for cls in selected_classes:
    src = os.path.join(SOURCE_DIR, cls)
    dst = os.path.join(TARGET_DIR, cls)
    os.makedirs(dst, exist_ok=True)
    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg",".jpeg",".png"))]
    chosen = random.sample(images, min(IMAGES_PER_CLS, len(images)))
    for img in chosen:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))
    print(f"  ✓ {cls}: {len(chosen)} images")

print(f"\n✅ Reduced dataset ready at '{TARGET_DIR}' ({NUM_CLASSES} classes, up to {IMAGES_PER_CLS} images each)")