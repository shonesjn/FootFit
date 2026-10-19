import pandas as pd
from pathlib import Path

path = Path("data/raw/W1/metadata.csv")

if not path.exists():
    print("File not found:", path.resolve())
    raise SystemExit(1)

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10).to_string())
print("\nData types:")
print(df.dtypes.to_string())
print("\nMissing values:")
print(df.isnull().sum().to_string())
