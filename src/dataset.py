import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer

class TextDataset:
    def __init__(self, data_path, tokenizer_name, max_length, save_path=None, load_cached=False, batch_size=4096):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        self.batch_size = batch_size

        if load_cached and save_path:
            print(f"Loading pre-tokenized dataset from {save_path}")
            self.dataset = Dataset.load_from_disk(save_path)
        else:
            self.dataset = self.load_and_tokenize(data_path, save_path)

    def load_and_tokenize(self, data_path, save_path):
        print("Loading dataset...")
        df = pd.read_csv(data_path, usecols=["text", "generated"], dtype={"generated": int})
        df.rename(columns={"generated": "label"}, inplace=True)
        
        raw_dataset = Dataset.from_pandas(df)
        print("Tokenizing dataset...")
        
        # Use batched tokenization with optimized settings
        dataset = raw_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["text"],  # Remove raw text after tokenization to save memory
            desc="Tokenizing dataset"
        )
        
        if save_path:
            print(f"Saving pre-tokenized dataset to {save_path}")
            dataset.save_to_disk(save_path)
        
        return dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

    def get_dataset(self):
        return self.dataset
