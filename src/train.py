import yaml
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from src.dataset import TextDataset

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_data, eval_data = TextDataset(
        config["model_name"],
        config["max_length"],
        save_path=config["tokenized_save_path"],
        load_cached=config["load_cached"],
    ).get_datasets()

    print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")

    model = DistilBertForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=float(config["learning_rate"]),
        fp16=config["fp16"],
        save_total_limit=config["save_total_limit"],
        logging_steps=config["logging_steps"],
        evaluation_strategy="epoch",  # <-- Run eval each epoch
        save_strategy="epoch",  # <-- Save checkpoint each epoch
        logging_dir="./logs",  # <-- TensorBoard logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,  # <-- Metrics
    )

    trainer.train()
    trainer.evaluate()  # <-- Final evaluation

    model.save_pretrained(config["model_save_path"])


if __name__ == "__main__":
    train()
