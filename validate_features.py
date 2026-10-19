import pandas as pd
from pathlib import Path

feature_path = Path("data/processed/pilot_features.csv")

metadata_files = {
    "P2": Path("data/raw/W1/metadata.csv"),
    "P1": Path("data/raw/participant_002/W1/metadata.csv"),
}

features = pd.read_csv(feature_path)

for label, path in metadata_files.items():
    print("\n" + "=" * 60)
    print("LABEL:", label)
    print("Metadata:", path)

    if not path.exists():
        print("ERROR: Metadata file not found")
        continue

    metadata = pd.read_csv(path)
    subset = features[
        features["footwear_label"] == label
    ].copy()

    print("Feature records:", len(subset))
    print("Metadata records:", len(metadata))
    print("Record counts match:", len(subset) == len(metadata))

    print("\nMetadata columns:")
    print(metadata.columns.tolist())

    for column in [
        "ParticipantID",
        "Footwear",
        "Speed",
        "FootstepID",
        "Exclude",
        "Incomplete",
        "Outlier",
    ]:
        if column in metadata.columns:
            print(f"\n{column} distribution:")
            print(metadata[column].value_counts(dropna=False))

    # Compare row ordering by checking selected identifiers.
    print("\nFirst 5 metadata identifiers:")
    available = [
        c for c in ["ParticipantID", "Footwear",
                     "Speed", "FootstepID"]
        if c in metadata.columns
    ]
    print(metadata[available].head().to_string(index=False))

print("\n" + "=" * 60)
print("FEATURE TABLE LABEL COUNTS")
print(features["footwear_label"].value_counts())
