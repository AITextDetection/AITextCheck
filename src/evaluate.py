import argparse
<<<<<<< HEAD
import random
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
=======
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
import seaborn as sns
import matplotlib.pyplot as plt

from inference import AITextDetector

# --- Arguments ---
parser = argparse.ArgumentParser()
<<<<<<< HEAD
parser.add_argument(
    "--data", type=str, default="data/AI_Human.csv", help="Path to CSV file"
)
parser.add_argument("--sample", action="store_true", help="Use 1000-row random sample")
parser.add_argument(
    "--chunksize", type=int, default=1000, help="Chunk size for full processing"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for model inference"
)
=======
parser.add_argument("--data", type=str, default="data/AI_Human.csv", help="Path to CSV file")
parser.add_argument("--sample", action="store_true", help="Use 1000-row random sample")
parser.add_argument("--chunksize", type=int, default=1000, help="Chunk size for full processing")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference")
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
args = parser.parse_args()

# 1. Load model
detector = AITextDetector("models/distilbert_finetuned", "distilbert-base-uncased")

<<<<<<< HEAD

=======
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
# 2. Add batched prediction method
def predict_in_batches(texts, batch_size=32):
    all_scores = []
    for i in range(0, len(texts), batch_size):
<<<<<<< HEAD
        batch = texts[i : i + batch_size]
=======
        batch = texts[i:i + batch_size]
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
        scores = detector.predict(batch)
        all_scores.extend(scores)
    return all_scores

<<<<<<< HEAD

=======
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
# 3. Initialize accumulators
all_labels = []
all_preds = []
all_probs = []

# 4. Read and process
if args.sample:
<<<<<<< HEAD
    print("Sampling 1,000 rows from large CSV without loading full file...")

    # Step 1: Get total number of lines (excluding header)
    with open(args.data, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # exclude header

    # Step 2: Randomly choose which rows to keep (excluding header)
    sample_indices = sorted(
        random.sample(range(1, total_lines + 1), 1000)
    )  # 1-based because header is row 0

    # Step 3: Read only those rows
    sample_df = pd.read_csv(
        args.data, skiprows=lambda i: i != 0 and i not in sample_indices
    )

    print(f"Loaded sample shape: {sample_df.shape}")

    # Step 4: Process in chunks
    for i in range(0, len(sample_df), args.chunksize):
        chunk = sample_df.iloc[i : i + args.chunksize]

        texts = chunk["text"].tolist()
        labels = chunk["generated"].tolist()

        ai_probs = predict_in_batches(texts, batch_size=args.batch_size)
        preds = [1 if prob >= 0.5 else 0 for prob in ai_probs]

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(ai_probs)
=======
    print("Using 1000-row random sample...")
    df = pd.read_csv(args.data)
    sample = df.sample(n=50000, random_state=42) if len(df) > 1000 else df

    texts = sample["text"].tolist()
    labels = sample["generated"].tolist()

    ai_probs = predict_in_batches(texts, batch_size=args.batch_size)
    preds = [1 if prob >= 0.5 else 0 for prob in ai_probs]

    all_labels.extend(labels)
    all_preds.extend(preds)
    all_probs.extend(ai_probs)
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9

else:
    print(f"Processing full file in chunks of {args.chunksize}...")
    for chunk in pd.read_csv(args.data, chunksize=args.chunksize):
        texts = chunk["text"].tolist()
        labels = chunk["generated"].tolist()

        ai_probs = predict_in_batches(texts, batch_size=args.batch_size)
        preds = [1 if prob >= 0.5 else 0 for prob in ai_probs]

        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(ai_probs)

# 5. Calculate metrics
print("\n--- Results ---")
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
<<<<<<< HEAD
print(
    "\nClassification Report:\n",
    classification_report(all_labels, all_preds, target_names=["Human", "AI"]),
)

# 6. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Human", "AI"],
    yticklabels=["Human", "AI"],
)
=======
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["Human", "AI"]))

# 6. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
<<<<<<< HEAD
plt.show()
=======
plt.show()
>>>>>>> d8dcd12030802dd5cb7ea109f29b0f2e4bb0eec9
