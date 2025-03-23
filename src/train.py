import torch
import yaml
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from src.dataset import TextDataset

def train():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    train_data = TextDataset(config["train_data"], config["model_name"], config["max_length"]).get_dataset()

    # Load model
    model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="models/bert_finetuned",
        eval_strategy="no",
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=float(config["learning_rate"]),
    )

    # Train model
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
    trainer.train()

    # Save trained model
    model.save_pretrained("models/bert_finetuned")

# Allow running directly
if __name__ == "__main__":
    train()