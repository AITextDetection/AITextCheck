# save_test_dataset_as_csv.py

from datasets import load_dataset
import pandas as pd


def main():
    print("Downloading 'test' split from 'andythetechnerd03/AI-human-text'...")
    dataset = load_dataset("andythetechnerd03/AI-human-text", split="validation")
    print(f"Loaded {len(dataset)} samples.")

    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Save as CSV
    output_path = "validation_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved test dataset to '{output_path}'.")


if __name__ == "__main__":
    main()
