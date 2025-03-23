import yaml
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from src.dataset import TextDataset

def train():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load pre-tokenized dataset
    train_data = TextDataset(
        config["train_data"], config["model_name"], config["max_length"],
        save_path=config["tokenized_save_path"], load_cached=True
    ).get_dataset()

    # Load DistilBERT model (smaller & faster)
    model = DistilBertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="models/distilbert_finetuned",
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=float(config["learning_rate"]),
        fp16=True,  # Enable mixed precision for faster training
        save_total_limit=1,  # Keep only the latest checkpoint
        logging_steps=100
    )

    # Train model
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
    trainer.train()

    # Save trained model
    model.save_pretrained("models/distilbert_finetuned")

if __name__ == "__main__":
    train()