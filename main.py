import argparse
from src.train import train
from src.inference import AITextDetector

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--predict", type=str, help="Run inference on text")

args = parser.parse_args()

if args.train:
    train()
elif args.predict:
    detector = AITextDetector("models/distilbert_finetuned", "distilbert-base-uncased")
    score = detector.predict(args.predict)
    print(f"AI-generated confidence: {score:.4f}")

    