import torch
from transformers import BertForSequenceClassification, BertTokenizer


class AITextDetector:
    def __init__(self, model_path, tokenizer_name):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
            ai_score = probabilities[0, 1].item()  # Confidence of AI-generated class

        return ai_score  # Return probability score instead of 0/1
