import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("data/AI_Human.csv")  # change to your actual file path

# Rename and convert label
df.rename(columns={"generated": "label"}, inplace=True)
df["label"] = df["label"].astype(int)

# Split train (70%) and temp (30%)
df_train, df_temp = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["label"]
)

# Split temp equally into eval (15%) and test (15%)
df_eval, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=42, stratify=df_temp["label"]
)

# Save to CSV inside the data/ folder
df_train.to_csv("data/train.csv", index=False)
df_eval.to_csv("data/eval.csv", index=False)
df_test.to_csv("data/test.csv", index=False)

print(
    f"Saved splits:\nTrain: {len(df_train)}\nEval: {len(df_eval)}\nTest: {len(df_test)}"
)
