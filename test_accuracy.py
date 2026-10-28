
from pathlib import Path
from collections import Counter
import re

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

from src.predict import model, transform, DEVICE, CLASS_NAMES


# ==================================================
# 1. SET YOUR GOOGLE DRIVE TESTING FOLDER PATH
# ==================================================

TEST_DIR = Path(
    r"G:\My Drive\data\Testing"
)



# ==================================================
# 2. CONFIGURATION
# ==================================================

BATCH_SIZE = 16
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Map the actual folder names to checkpoint class names.
class_lookup = {
    normalize_name(name): index
    for index, name in enumerate(CLASS_NAMES)
}

if not TEST_DIR.is_dir():
    raise FileNotFoundError(
        f"Testing folder not found: {TEST_DIR}\n"
        "Check your Google Drive path."
    )

image_paths = []
true_labels = []

for folder in sorted(TEST_DIR.iterdir()):
    if not folder.is_dir():
        continue

    folder_key = normalize_name(folder.name)

    if folder_key not in class_lookup:
        print(f"Skipping unrecognized class folder: {folder.name}")
        continue

    label = class_lookup[folder_key]

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    image_paths.extend(files)
    true_labels.extend([label] * len(files))

if not image_paths:
    raise RuntimeError(
        "No labeled images found. Check the Testing folder structure."
    )

print("Classes:", CLASS_NAMES)
print("Device:", DEVICE)
print("Total test images:", len(image_paths))
print("Batch size:", BATCH_SIZE)
print("Loading test images...")


# ==================================================
# 3. LOAD IMAGES IN BATCHES
# ==================================================

class BrainMRIDataset(Dataset):
    def __init__(self, paths, labels, image_transform):
        self.paths = paths
        self.labels = labels
        self.transform = image_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as error:
            raise RuntimeError(
                f"Could not read image: {path}"
            ) from error

        return tensor, self.labels[index]


dataset = BrainMRIDataset(image_paths, true_labels, transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 4. RUN BATCH PREDICTIONS
# ==================================================

model.eval()
all_predictions = []
all_labels = []

print("\nEvaluating model...")

with torch.inference_mode():
    for images, labels in tqdm(loader, desc="Testing", unit="batch"):
        images = images.to(DEVICE)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())


# ==================================================
# 5. ACCURACY AND CLASSIFICATION METRICS
# ==================================================

accuracy = accuracy_score(all_labels, all_predictions)
correct = sum(
    pred == actual
    for pred, actual in zip(all_predictions, all_labels)
)

print("\n========== NEUROSCAN TEST RESULTS ==========")
print("Device:", DEVICE)
print("Images evaluated:", len(all_labels))
print("Correct predictions:", correct)
print("Incorrect predictions:", len(all_labels) - correct)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix (actual rows, predicted columns):")
print(confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES)))
))

print("\nEvaluation completed.")
