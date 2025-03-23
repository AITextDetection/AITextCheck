import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer

class TextDataset:
    def __init__(self, data_path, tokenizer_name, max_length):
        self.data = pd.read_csv(data_path)

        # Rename "generated" column to "label"
        self.data.rename(columns={"generated": "label"}, inplace=True)
        # Ensure labels are 0 or 1 (if needed)
        self.data["label"] = self.data["label"].astype(int)
        
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def tokenize_function(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def get_dataset(self):
        dataset = Dataset.from_pandas(self.data)
        return dataset.map(self.tokenize_function, batched=True)