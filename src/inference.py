import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class AITextDetector:
    def __init__(self, model_path, tokenizer_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def predict(self, text):
        # Tokenize and move to correct device
        print("Getting the inputs")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits / 2, dim=1)  # Convert logits to probabilities
            ai_score = probabilities[:, 1].cpu().numpy().tolist()  # Confidence of AI-generated class

        return ai_score if isinstance(text, list) else ai_score[0]  # Support batch/single prediction


