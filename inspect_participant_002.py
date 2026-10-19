import pandas as pd
from pathlib import Path

path = Path("data/raw/P2 W1/metadata.csv")
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nFootwear:")
print(df["Footwear"].value_counts())

print("\nSpeed:")
print(df["Speed"].value_counts())

print("\nQuality flags:")
print(df[["Incomplete", "Outlier", "Exclude"]].sum())

print("\nFirst five records:")
print(df.head().to_string(index=False))
