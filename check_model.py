
from pathlib import Path
import torch

MODEL_DIR = Path(__file__).resolve().parent / "models"

classifier_state = torch.load(
    MODEL_DIR / "tumor_classifier.pth",
    map_location="cpu",
    weights_only=True
)

print("Classifier checkpoint keys:", list(classifier_state.keys()))

print(
    "Classifier weight shape:",
    classifier_state["weight"].shape
)

print(
    "Classifier bias shape:",
    classifier_state["bias"].shape
)
