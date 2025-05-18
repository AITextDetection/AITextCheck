import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

from inference import AITextDetector


def predict_in_batches(detector, texts, batch_size=32):
    all_scores = []
    total = len(texts)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        scores = detector.predict(batch)
        all_scores.extend(scores)
        if (i // batch_size) % 10 == 0 or i + batch_size >= total:
            print(f"Processed {min(i + batch_size, total)} / {total} texts")
    return all_scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI text detector on test data"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Path to test CSV file (optional)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Load trained model
    detector = AITextDetector("models/distilbert_finetuned", "distilbert-base-uncased")

    if args.data:
        print(f"Loading test data from CSV: {args.data}")
        df = pd.read_csv(args.data)
        texts = df["text"].tolist()
        labels = df["generated"].tolist()
    else:
        print("Loading test data from Hugging Face dataset...")
        dataset = load_dataset("andythetechnerd03/AI-human-text")
        test_dataset = dataset["test"]
        texts = test_dataset["text"]
        labels = test_dataset["generated"]

    # Predict probabilities in batches
    probs = predict_in_batches(detector, texts, batch_size=args.batch_size)

    # Convert probabilities to binary predictions (threshold 0.5)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(
        "\nClassification Report:\n",
        classification_report(labels, preds, target_names=["Human", "AI"]),
    )

    # Confusion matrix plot
    cm = confusion_matrix(labels, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
