from pathlib import Path
import pandas as pd

DATA_DIR = Path("../data/raw")

if not DATA_DIR.exists():
    DATA_DIR = Path("data/raw")

print("FOOTFIT AI - DATASET INSPECTION")
print("-" * 40)

if not DATA_DIR.exists():
    print("Dataset folder not found:", DATA_DIR.resolve())
    print("Create data/raw and place the dataset files there.")
else:
    files = list(DATA_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]

    print("Dataset directory:", DATA_DIR.resolve())
    print("Total files:", len(files))

    for file in files[:30]:
        print(file.relative_to(DATA_DIR))

    csv_files = [f for f in files if f.suffix.lower() == ".csv"]

    for csv_file in csv_files[:5]:
        print("\nCSV file:", csv_file.name)

        try:
            df = pd.read_csv(csv_file, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.head())
        except Exception as error:
            print("Could not read CSV:", error)