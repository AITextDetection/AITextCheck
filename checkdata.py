import pandas as pd
pd.set_option('display.max_colwidth', None)  # This will prevent truncating of text

# Load the CSV file into a DataFrame
file_path = 'data/AI_Human.csv'  # Replace with your actual file path

df = pd.read_csv(file_path)

# Filter rows where 'generated' column has value 1
generated_text = df[df['generated'] == 0]

# Print top 10 rows of generated text
print("Top 10 rows with generated = 1:")
print(generated_text.head(10))

# Print bottom 10 rows of generated text
print("\nBottom 10 rows with generated = 1:")
print(generated_text.tail(10))