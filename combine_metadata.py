import pandas as pd
from pathlib import Path

files = [
    Path("data/raw/W1/metadata.csv"),
    Path("data/raw/P2 W1/metadata.csv"),
]

frames = []
for file in files:
    if not file.exists():
        raise FileNotFoundError(file)
    df = pd.read_csv(file)
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

print("Combined shape:", combined.shape)
print("\nParticipants:")
print(combined["ParticipantID"].value_counts())

print("\nFootwear conditions:")
print(combined["Footwear"].value_counts())

print("\nQuality-control summary:")
print(
    combined.groupby("ParticipantID")[
        ["Incomplete", "Outlier", "Exclude"]
    ].sum()
)

output = Path("data/processed")
output.mkdir(parents=True, exist_ok=True)
combined.to_csv(
    output / "pilot_metadata.csv",
    index=False
)
print("\nSaved to data/processed/pilot_metadata.csv")

valid = combined[combined["Exclude"] == 0].copy()
print("\nPILOT DATASET REPORT")
print("-" * 35)
print("Total footsteps:", len(combined))
print("Candidate valid footsteps:", len(valid))
print("Participants:", valid["ParticipantID"].nunique())
print("Footwear conditions:", valid["Footwear"].unique())
print("Walking speeds:", valid["Speed"].unique())

print("\nMean pressure by footwear:")
print(
    valid.groupby("Footwear")["MeanPressure"].agg(
        ["count", "mean", "median", "std"]
    )
)

print("\nFoot dimensions by footwear:")
print(valid.groupby("Footwear")[["FootLength", "FootWidth"]].mean())

