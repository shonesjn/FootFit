import pandas as pd
from pathlib import Path

path = Path("data/raw/W1/metadata.csv")
df = pd.read_csv(path)

print("DATASET SUMMARY")
print("=" * 45)
print("Total footsteps:", len(df))
print("Participants:", df["ParticipantID"].nunique())

print("\nFootwear counts:")
print(df["Footwear"].value_counts())

print("\nSpeed counts:")
print(df["Speed"].value_counts())

print("\nFoot side counts:")
print(df["Side"].value_counts())

print("\nQuality-control flags:")
for col in ["Incomplete", "Outlier", "Exclude", "Standing"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\nFootwear by participant:")
print(
    df.groupby("Footwear")["ParticipantID"]
      .nunique()
)

print("\nPressure summary:")
print(df["MeanPressure"].describe())

print("\nRecords marked for exclusion:")
print(
    df.loc[
        df["Exclude"] == 1,
        [
            "ParticipantID", "Footwear", "Speed",
            "FootstepID", "Incomplete", "Outlier",
            "MeanPressure"
        ]
    ].to_string(index=False)
)
