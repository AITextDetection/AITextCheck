import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer

class TextDataset:
    def __init__(self, data_path, tokenizer_name, max_length, save_path=None, load_cached=False):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

        if load_cached and save_path:
            print(f"Loading pre-tokenized dataset from {save_path}")
            self.dataset = Dataset.load_from_disk(save_path)  # Load cached dataset
        else:
            print("Tokenizing dataset...")
            self.data = pd.read_csv(
                data_path, 
                usecols=["text", "generated"],  # Load only required columns
                dtype={"generated": int}  # Optimize memory usage
            )
            self.data.rename(columns={"generated": "label"}, inplace=True)
            self.data["label"] = self.data["label"].astype(int)

            raw_dataset = Dataset.from_dict({"text": self.data["text"].tolist(), "label": self.data["label"].tolist()})
            self.dataset = raw_dataset.map(self.tokenize_function, batched=True, batch_size=4096, num_proc=4)
            if save_path:
                print(f"Saving pre-tokenized dataset to {save_path}")
                self.dataset.save_to_disk(save_path)  # Save tokenized dataset

    def tokenize_function(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def get_dataset(self):
        return self.dataset