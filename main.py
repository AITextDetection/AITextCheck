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
    detector = AITextDetector("models/bert_finetuned", "bert-base-uncased")
    print(detector.predict(args.predict))

    