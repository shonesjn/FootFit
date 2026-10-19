import numpy as np
from pathlib import Path

file_path = Path("data/raw/preprocessed_001_BF_S1.npz")

with np.load(file_path) as data:
    frames = data["arr_0"]

print("Dataset shape:", frames.shape)
print("Data type:", frames.dtype)

# Analyze each frame
active_pixels = np.count_nonzero(frames, axis=(1, 2))
mean_intensity = frames.mean(axis=(1, 2))
max_intensity = frames.max(axis=(1, 2))

print("\nFrame statistics")
print("Average active pixels:", active_pixels.mean())
print("Minimum active pixels:", active_pixels.min())
print("Maximum active pixels:", active_pixels.max())

print("\nIntensity statistics")
print("Average intensity:", mean_intensity.mean())
print("Minimum frame maximum:", max_intensity.min())
print("Maximum frame maximum:", max_intensity.max())

# Check whether the frames vary
print("\nUnique frame count:", np.unique(frames, axis=0).shape[0])
