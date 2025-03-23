import yaml
from src.dataset import TextDataset

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Pre-tokenize and save dataset
dataset = TextDataset(
    config["train_data"], config["model_name"], config["max_length"], 
    save_path=config["tokenized_save_path"]
)