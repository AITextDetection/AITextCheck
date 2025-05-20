# 🤖 AI Text Detector using DistilBERT

This project fine-tunes [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) to classify text as **AI-generated** or **human-written**, and exposes it via a FastAPI endpoint.

---

## 📦 Project Structure

```
.
├── app.py                 # FastAPI app
├── config.yaml            # Config file
├── data/
│   └── train.csv          # Your training data
├── models/
│   └── distilbert_finetuned/  # Saved model
├── src/
│   ├── tokenize_data.py   # Tokenization script
│   ├── train_model.py     # Model training logic
│   ├── inference.py       # Inference logic
│   └── evaluate_model.py  # Optional: model evaluation
├── main.py                # Entrypoint for training
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## ✅ Prerequisites

- Python 3.8+
- pip
- Git
- For Mac users:
  ```bash
  brew install libomp
  ```

---

## 🖥️ Setup (Windows/macOS)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-text-detector.git
cd ai-text-detector
```

### 2. Create and activate a virtual environment

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Tokenization & Training

```bash
python -m src.tokenize_data
python main.py --train
```

---

## 🔍 Inference

```bash
python main.py --predict "how are you"
```

---

## 🚀 Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📈 TensorBoard (Optional)

```bash
tensorboard --logdir=./logs
```

---

## 🧾 Evaluation

Evaluate on a custom test set:

```bash
python src/evaluate_model.py --data path/to/your/test.csv --batch_size 32
```

Evaluate on default settings:

```bash
python src/evaluate_model.py
```

---

## 📌 Notes

- To update `requirements.txt` after installing new packages:

  ```bash
  pip freeze > requirements.txt
  ```
