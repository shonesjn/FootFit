import numpy as np
import pandas as pd
from pathlib import Path

sources = {
    "P2": Path("data/raw/W1/pipeline_1.npz"),
    "P1": Path("data/raw/P2 W1/pipeline_1.npz"),
}

output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)

rows = []

for footwear, path in sources.items():
    if not path.exists():
        print(f"Skipping missing file: {path}")
        continue

    with np.load(path, allow_pickle=False) as data:
        records = data["arr_0"]

    print(f"Processing {footwear}: {records.shape}")

    for idx, sample in enumerate(records):
        # Current data layout: spatial dimensions are
        # provisionally treated as axes 0 and 1,
        # and the final dimension as slices.
        values = sample.astype(np.float64)
        active = values[values > 0]

        row = {
            "footwear_label": footwear,
            "record_index": idx,
            "record_max": float(values.max()),
            "record_mean_all": float(values.mean()),
            "record_sum": float(values.sum()),
            "active_pixel_count_total": int(
                np.count_nonzero(values)
            ),
            "active_value_mean": (
                float(active.mean()) if active.size else 0.0
            ),
            "active_value_std": (
                float(active.std()) if active.size else 0.0
            ),
        }

        # Per-slice features
        slice_max = values.max(axis=(0, 1))
        slice_active = np.count_nonzero(values, axis=(0, 1))
        slice_sum = values.sum(axis=(0, 1))

        row["peak_slice_index"] = int(np.argmax(slice_max))
        row["peak_slice_max"] = float(slice_max.max())
        row["max_active_pixels_in_slice"] = int(
            slice_active.max()
        )
        row["max_slice_sum"] = float(slice_sum.max())
        row["active_slice_count"] = int(
            np.count_nonzero(slice_active)
        )

        rows.append(row)

df = pd.DataFrame(rows)
destination = output / "pilot_features.csv"
df.to_csv(destination, index=False)

print("\nFeature table shape:", df.shape)
print("Saved to:", destination)
print(df.head())
print("\nSummary by label:")
print(
    df.groupby("footwear_label")[
        ["record_max", "active_value_mean",
         "active_pixel_count_total", "active_slice_count"]
    ].mean().round(2)
)
