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
            
            # From api  
            prepare_hc3_dataset()

            # From file
            self.data = pd.read_csv(data_path)
            self.data.rename(columns={"generated": "label"}, inplace=True)
            self.data["label"] = self.data["label"].astype(int)

            raw_dataset = Dataset.from_pandas(self.data)
            self.dataset = raw_dataset.map(self.tokenize_function, batched=True)

            if save_path:
                print(f"Saving pre-tokenized dataset to {save_path}")
                self.dataset.save_to_disk(save_path)  # Save tokenized dataset

    def tokenize_function(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def get_dataset(self):
        return self.dataset
    


def prepare_hc3_dataset(dataset_name="Hello-SimpleAI/HC3", split="train", save_csv=False, filename="train.csv"):
    """
    Loads the HC3 dataset from Hugging Face and transforms it into a binary classification dataset.
    
    Args:
        dataset_name (str): The name of the dataset on Hugging Face.
        split (str): The dataset split to load (default is "train").
        save_csv (bool): Whether to save the dataset as a CSV file.
        filename (str): The name of the CSV file if save_csv is True.

    Returns:
        pd.DataFrame: A DataFrame containing "text" and "label" (0 for human, 1 for AI).
    """
    # Load dataset
    dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")    
    print(dataset)

    # List to store transformed data
    data = []

    # Iterate through each record
    for record in dataset:
        # Add human answers with label 0
        for answer in record.get("human_answers", []):
            data.append({"text": answer, "label": 0})

        # Add AI-generated answers with label 1
        for answer in record.get("chatgpt_answers", []):
            data.append({"text": answer, "label": 1})
            print(answer)
    print(data.count)

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Shuffle the dataset (optional)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save as CSV if required
    if save_csv:
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")

    return df