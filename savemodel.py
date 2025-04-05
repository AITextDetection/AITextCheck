import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

MODEL_PATH = "models/distilbert_finetuned"  # Path where your trained model is stored
SAVE_PATH = "saved_model"  # Destination where you want to save it

# Load the trained model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Save model
model.save_pretrained(SAVE_PATH)

# Save tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained(SAVE_PATH)

print(f"Model and tokenizer saved to '{SAVE_PATH}/'")