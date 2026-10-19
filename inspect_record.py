import numpy as np
from pathlib import Path

files = [
    Path("data/raw/W1/pipeline_1.npz"),
    Path("data/raw/P2 W1/pipeline_1.npz"),
]

for path in files:
    print("\n" + "=" * 60)
    print("FILE:", path)

    with np.load(path, allow_pickle=False) as data:
        arr = data["arr_0"]

        # Inspect the first footstep
        sample = arr[0]

        print("Full shape:", arr.shape)
        print("First record shape:", sample.shape)
        print("First record min:", sample.min())
        print("First record max:", sample.max())
        print("First record mean:", sample.mean())

        # Check variation along each axis
        for axis in range(sample.ndim):
            variation = np.std(sample, axis=axis)
            print(
                f"After reducing axis {axis}:",
                variation.shape,
                "std range:",
                float(variation.min()),
                float(variation.max()),
            )

        # Inspect several positions
        print("Sample values:", sample.ravel()[:20])
