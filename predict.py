
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_classifier.joblib"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run python train.py first.")
        return

    model = joblib.load(MODEL_PATH)

    print("=== Spam Message Classifier ===")
    print("Type 'exit' to stop.\n")

    while True:
        message = input("Enter a message: ").strip()

        if message.lower() in {"exit", "quit"}:
            print("Program ended.")
            break

        if not message:
            print("Please enter a message.\n")
            continue

        prediction = model.predict([message])[0]

        if prediction == "spam":
            print("Result: SPAM")
        else:
            print("Result: NON-SPAM")

        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        confidence = probabilities[classes.index(prediction)]

        print(f"Predicted-class score: {confidence:.2%}\n")


if __name__ == "__main__":
    main()
