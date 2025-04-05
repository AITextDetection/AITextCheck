import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from inference import AITextDetector

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/AI_Human.csv", help="Path to CSV file")
parser.add_argument("--sample", action="store_true", help="Use 1000-row random sample")
parser.add_argument("--chunksize", type=int, default=1000, help="Chunk size for full processing")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference")
args = parser.parse_args()

# 1. Load model
detector = AITextDetector("models/distilbert_finetuned", "distilbert-base-uncased")

# 2. Add batched prediction method
def predict_in_batches(texts, batch_size=32):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        scores = detector.predict(batch)
        all_scores.extend(scores)
    return all_scores

# 3. Initialize accumulators
all_labels = []
all_preds = []
all_probs = []

# 4. Read and process
if args.sample:
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
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["Human", "AI"]))

# 6. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()