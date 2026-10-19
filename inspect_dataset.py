import numpy as np
from pathlib import Path

file_path = Path("data/raw/preprocessed_001_BF_S1.npz")

if not file_path.exists():
    print("Dataset file not found:", file_path.resolve())
    raise SystemExit(1)

print("File size:", round(file_path.stat().st_size / (1024**2), 2), "MB")

with np.load(file_path, allow_pickle=False) as data:
    print("\nAvailable arrays:")

    for key in data.files:
        array = data[key]
        print(f"\nName: {key}")
        print(f"Shape: {array.shape}")
        print(f"Data type: {array.dtype}")

        if np.issubdtype(array.dtype, np.number):
            print("Minimum:", np.nanmin(array))
            print("Maximum:", np.nanmax(array))
