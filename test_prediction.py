
from pathlib import Path
from src.predict import predict_image

TEST_DIR = Path(__file__).resolve().parent / "test_images"

for image_path in sorted(TEST_DIR.glob("*")):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    print(f"\nImage: {image_path.name}")

    try:
        result = predict_image(image_path)

        print("Predicted class:", result["predicted_class"])
        print("Model confidence:", f'{result["confidence"]:.2f}%')

        print("Class probabilities:")
        for name, score in result["class_probabilities"].items():
            print(f"  {name}: {score:.2f}%")

    except Exception as error:
        print("Prediction failed:", error)
