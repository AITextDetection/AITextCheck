import os
import json
import torch
import logging
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Global variables
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init():
    """
    This function is called when the container starts. It loads the model and tokenizer into memory.
    """
    global model, tokenizer

    # Get the model directory from the environment variable
    model_dir = os.getenv("AZUREML_MODEL_DIR", "./")  # Fallback to local if running outside Azure
    model_path = os.path.join(model_dir, "Bert_1gb")  # Update with your model folder name

    try:
        # Load model and tokenizer
        model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")  # Change if custom tokenizer used

        logging.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")

def run(raw_data):
    """
    This function is called for each API request. It takes input text and returns predictions.
    """
    try:
        logging.info("Received request for inference.")
        
        # Parse JSON input
        data = json.loads(raw_data)
        text_input = data.get("text", "")  # Expecting {"text": "input sentence"}
        
        if not text_input:
            return json.dumps({"error": "No text input provided."})
        
        # Tokenize input
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            ai_score = probabilities[:, 1].cpu().numpy().tolist()  # Confidence for AI-generated class

        logging.info("Inference successful.")
        return json.dumps({"ai_score": ai_score[0] if isinstance(text_input, str) else ai_score})

    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return json.dumps({"error": str(e)})