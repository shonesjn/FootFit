import numpy as np
from pathlib import Path

files = [
    Path("data/raw/W1/pipeline_1.npz"),
    Path("data/raw/P2 W1/pipeline_1.npz"),
]

for path in files:
    print("\n" + "=" * 60)
    print("FILE:", path)
    print("EXISTS:", path.exists())

    if not path.exists():
        continue

    with np.load(path, allow_pickle=False) as data:
        print("ARRAY NAMES:", data.files)

        for name in data.files:
            arr = data[name]
            print(f"\nArray: {name}")
            print("Shape:", arr.shape)
            print("Data type:", arr.dtype)

            if arr.size > 0 and np.issubdtype(
                arr.dtype, np.number
            ):
                print("Minimum:", arr.min())
                print("Maximum:", arr.max())
