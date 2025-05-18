# tokenize.py
import yaml
from src.dataset import TextDataset


def tokenize_data():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Pre-tokenize and save dataset
    TextDataset(
        tokenizer_name=config["model_name"],
        max_length=config["max_length"],
        save_path=config["tokenized_save_path"],
        load_cached=False,  # force tokenization and save
    )


if __name__ == "__main__":
    tokenize_data()
