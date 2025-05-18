from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer
from datasets import load_dataset, Dataset, DatasetDict


class TextDataset:
    def __init__(self, tokenizer_name, max_length, save_path=None, load_cached=False):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if load_cached and save_path:
            print(f"Loading pre-tokenized dataset from {save_path}")
            dataset = Dataset.load_from_disk(save_path)
            self.train_dataset = dataset["train"]
            self.test_dataset = dataset["test"]
        else:
            print("Fetching dataset from Hugging Face...")
            dataset = load_dataset("andythetechnerd03/AI-human-text")
            df_train = dataset["train"].to_pandas()
            df_test = dataset["test"].to_pandas()

            # Rename and convert label column
            for df in [df_train, df_test]:
                df.rename(columns={"generated": "label"}, inplace=True)
                df["label"] = df["label"].astype(int)

            ds_train = Dataset.from_pandas(df_train)
            ds_test = Dataset.from_pandas(df_test)

            print("Tokenizing dataset...")
            self.train_dataset = ds_train.map(self.tokenize_function, batched=True)
            self.test_dataset = ds_test.map(self.tokenize_function, batched=True)

            # Save tokenized dataset
            if save_path:
                print(f"Saving pre-tokenized datasets to {save_path}")
                DatasetDict(
                    {"train": self.train_dataset, "test": self.test_dataset}
                ).save_to_disk(save_path)

    def tokenize_function(self, example):
        return self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def get_datasets(self):
        return self.train_dataset, self.test_dataset
