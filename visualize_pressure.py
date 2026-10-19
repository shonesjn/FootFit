
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("data/raw/W1/pipeline_1.npz")

with np.load(path, allow_pickle=False) as data:
    arr = data["arr_0"]

# Select the first footstep
sample = arr[0]

print("Record shape:", sample.shape)

# Display selected slices along the third axis
indices = [0, 10, 20, 30, 39]

fig, axes = plt.subplots(1, len(indices), figsize=(16, 4))

for ax, idx in zip(axes, indices):
    image = sample[:, :, idx]
    ax.imshow(image, cmap="inferno", origin="lower")
    ax.set_title(f"Slice {idx}")
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 0")

plt.suptitle("FootFit AI — Pressure Record Inspection")
plt.tight_layout()
plt.show()
