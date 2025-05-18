from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from fastapi.middleware.cors import CORSMiddleware

import torch

from src.inference import AITextDetector

# Initialize FastAPI
app = FastAPI()

# Allowed origins (frontend URL, '*' for all)
origins = ["*"]  # Change this to specific domains if needed, e.g., ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Load model and tokenizer
detector = AITextDetector(model_path="models/distilbert_finetuned", tokenizer_name="distilbert-base-uncased")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    score = detector.predict(data.text)
    print(f"AI-generated confidence: {score:.4f}")
    score *= 100
    score = round(score, 2)
    return {"score": score}

# Root endpoint for health check
@app.get("/")
def root():
    return {"message": "Model is running!"}