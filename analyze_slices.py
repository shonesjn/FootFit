import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("data/raw/W1/pipeline_1.npz")

with np.load(path, allow_pickle=False) as data:
    sample = data["arr_0"][0]

slice_max = []
slice_mean = []
active_pixels = []
total_pressure = []

for i in range(sample.shape[2]):
    frame = sample[:, :, i]

    slice_max.append(float(frame.max()))
    slice_mean.append(float(frame.mean()))
    active_pixels.append(int(np.count_nonzero(frame)))
    total_pressure.append(float(frame.sum()))

print("Slice | Max | Mean | Active pixels | Sum")
print("-" * 55)

for i in range(len(slice_max)):
    print(
        f"{i:5d} | {slice_max[i]:5.1f} | "
        f"{slice_mean[i]:7.3f} | "
        f"{active_pixels[i]:13d} | "
        f"{total_pressure[i]:10.1f}"
    )

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].plot(slice_max, marker=".")
axes[0].set_title("Maximum Value per Slice")
axes[0].set_xlabel("Slice index")
axes[0].set_ylabel("Maximum value")

axes[1].plot(active_pixels, marker=".")
axes[1].set_title("Nonzero Pixels per Slice")
axes[1].set_xlabel("Slice index")
axes[1].set_ylabel("Active pixels")

plt.tight_layout()

# Save plot to results folder
output_img = Path("results/analyze_slices_plot.png")
output_img.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_img, dpi=150)

plt.show()
