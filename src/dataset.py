import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizer

class TextDataset:
    def __init__(self, data_path, tokenizer_name, max_length, save_path=None, load_cached=False):
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if load_cached and save_path:
            print(f"Loading pre-tokenized dataset from {save_path}")
            self.dataset = Dataset.load_from_disk(save_path)  # Load cached dataset
        else:
            print("Tokenizing dataset...")

            # If data_path is missing, generate data from API
            api = True
            if api:
                print("No data path provided. Fetching HC3 dataset...")
                self.data = prepare_hc3_dataset(save_csv=True, filename="train.csv")  # Save as CSV
                data_path = "train.csv"
            else:
                self.data = pd.read_csv(data_path)  # Load dataset from CSV

            # Process dataset
            self.data.rename(columns={"generated": "label"}, inplace=True, errors="ignore")
            self.data["label"] = self.data["label"].astype(int)

            raw_dataset = Dataset.from_pandas(self.data)
            self.dataset = raw_dataset.map(self.tokenize_function, batched=True, batch_size=1000)

            # Save tokenized dataset
            if save_path:
                print(f"Saving pre-tokenized dataset to {save_path}")
                self.dataset.save_to_disk(save_path)

    def tokenize_function(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def get_dataset(self):
        return self.dataset
    

def prepare_hc3_dataset(dataset_name="Hello-SimpleAI/HC3", split="train", save_csv=False, filename="train.csv"):
    """
    Loads the HC3 dataset from Hugging Face and transforms it into a binary classification dataset.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, name="all", split=split)    
    print(f"Loaded dataset with {len(dataset)} records.")

    # Transform data
    data = []
    for record in dataset:
        for answer in record.get("human_answers", []):
            data.append({"text": answer, "label": 0})
        for answer in record.get("chatgpt_answers", []):
            data.append({"text": answer, "label": 1})

    print(f"Transformed dataset contains {len(data)} examples.")  # ✅ Fix

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)

    if save_csv:
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")

    return df