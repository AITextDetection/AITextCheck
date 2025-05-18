import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer
from collections import Counter


class TextDataset:
    def __init__(
        self,
        tokenizer_name,
        max_length,
        save_path=None,
        load_cached=False,
        data_folder="data",
    ):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data_folder = data_folder

        if load_cached and save_path:
            print(f"Loading pre-tokenized dataset from {save_path}")
            dataset = DatasetDict.load_from_disk(save_path)
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["eval"]

            train_labels = self.train_dataset["label"]
            eval_labels = self.eval_dataset["label"]

            train_counts = Counter(train_labels)
            eval_counts = Counter(eval_labels)

            print("\nTrain label distribution:")
            for k, v in train_counts.items():
                print(f"Label {k}: {v} ({v / sum(train_counts.values()):.2%})")

            print("\nEval label distribution:")
            for k, v in eval_counts.items():
                print(f"Label {k}: {v} ({v / sum(eval_counts.values()):.2%})")

        else:
            print("Loading dataset from local CSV files...")
            # Load CSV files
            df_train = pd.read_csv(f"{self.data_folder}/train.csv")
            df_eval = pd.read_csv(f"{self.data_folder}/eval.csv")

            # Ensure label is int
            df_train["label"] = df_train["label"].astype(int)
            df_eval["label"] = df_eval["label"].astype(int)

            # Convert to Huggingface Dataset
            ds_train = Dataset.from_pandas(df_train)
            ds_eval = Dataset.from_pandas(df_eval)

            print("Tokenizing datasets...")
            self.train_dataset = ds_train.map(self.tokenize_function, batched=True)
            self.eval_dataset = ds_eval.map(self.tokenize_function, batched=True)

            if save_path:
                print(f"Saving tokenized datasets to {save_path}")
                DatasetDict(
                    {"train": self.train_dataset, "eval": self.eval_dataset}
                ).save_to_disk(save_path)

            train_counts = Counter(self.train_dataset["label"])
            eval_counts = Counter(self.eval_dataset["label"])

            print("\nTrain label distribution:")
            for k, v in train_counts.items():
                print(f"Label {k}: {v} ({v / sum(train_counts.values()):.2%})")

            print("\nEval label distribution:")
            for k, v in eval_counts.items():
                print(f"Label {k}: {v} ({v / sum(eval_counts.values()):.2%})")

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def get_datasets(self):
        return self.train_dataset, self.eval_dataset
