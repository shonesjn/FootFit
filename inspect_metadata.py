import pandas as pd
from pathlib import Path

path = Path("data/raw/participant_metadata.csv")

if not path.exists():
    print("Metadata file not found:", path.resolve())
    raise SystemExit(1)

df = pd.read_csv(path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.head().to_string())
print("\nData types:")
print(df.dtypes.to_string())
print("\nMissing values:")
print(df.isnull().sum().to_string())
