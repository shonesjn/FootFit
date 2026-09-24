
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# FootFit AI: Build Metadata-Linked Pilot Dataset
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
FEATURE_FILE = PROJECT_DIR / "data" / "processed" / "pilot_features.csv"

METADATA_SOURCES = {
    "P2": PROJECT_DIR / "data" / "raw" / "W1" / "metadata.csv",
    "P1": PROJECT_DIR / "data" / "raw" / "participant_002" / "W1" / "metadata.csv",
}

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUTPUT = OUTPUT_DIR / "pilot_features_with_metadata.csv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "pilot_candidate_records.csv"


def main():
    # 1. Check input files
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)

    if "footwear_label" not in features.columns:
        raise ValueError(
            "Feature file must contain 'footwear_label'."
        )

    merged_parts = []

    # 2. Join each footwear group's features with metadata
    for label, metadata_path in METADATA_SOURCES.items():
        print("\n" + "=" * 60)
        print(f"Processing footwear label: {label}")
        print(f"Metadata file: {metadata_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        metadata = pd.read_csv(metadata_path)

        subset = features[
            features["footwear_label"] == label
        ].copy()

        # Verify the expected record counts
        if len(subset) != len(metadata):
            raise ValueError(
                f"Record-count mismatch for {label}: "
                f"{len(subset)} feature records versus "
                f"{len(metadata)} metadata records."
            )

        # Check that record indices are sequential
        expected_indices = list(range(len(subset)))

        if subset["record_index"].tolist() != expected_indices:
            raise ValueError(
                f"Unexpected record_index ordering for {label}."
            )

        # Verify the expected metadata labels
        if "Footwear" not in metadata.columns:
            raise ValueError(
                f"'Footwear' column missing for {label}."
            )

        if not metadata["Footwear"].eq(label).all():
            raise ValueError(
                f"Metadata footwear labels do not match {label}."
            )

        # Avoid duplicate footwear columns by dropping redundant footwear_label from features
        if "footwear_label" in subset.columns:
            subset = subset.drop(columns=["footwear_label"])

        # Preserve row order for this preliminary join.
        # Original pipeline-to-metadata alignment must still
        # be confirmed from dataset documentation.
        metadata = metadata.reset_index(drop=True)
        subset = subset.reset_index(drop=True)

        combined = pd.concat(
            [metadata, subset], axis=1
        )

        merged_parts.append(combined)

        print("Feature records:", len(subset))
        print("Metadata records:", len(metadata))
        print("Record counts match: True")

    # 3. Combine all records
    all_records = pd.concat(
        merged_parts, ignore_index=True
    )

    # 4. Save the complete metadata-linked dataset
    all_records.to_csv(ALL_OUTPUT, index=False)

    # 5. Validate quality-control columns
    required_flags = ["Exclude", "Incomplete", "Outlier"]

    for column in required_flags:
        if column not in all_records.columns:
            raise ValueError(
                f"Required metadata column missing: {column}"
            )

    # 6. Create the candidate subset using the existing
    # dataset exclusion flag only
    candidates = all_records[
        all_records["Exclude"].eq(0)
    ].copy()

    candidates.to_csv(CANDIDATE_OUTPUT, index=False)

    # 7. Report results
    print("\n" + "=" * 60)
    print("FOOTFIT AI PILOT DATASET SUMMARY")
    print("=" * 60)

    print("Complete dataset shape:", all_records.shape)
    print("Candidate dataset shape:", candidates.shape)

    print("\nComplete record counts by participant and footwear:")
    print(
        all_records.groupby(
            ["ParticipantID", "Footwear"]
        ).size().to_string()
    )

    print("\nCandidate record counts by participant and footwear:")
    print(
        candidates.groupby(
            ["ParticipantID", "Footwear"]
        ).size().to_string()
    )

    print("\nQuality flags in complete dataset:")
    for column in required_flags:
        print(f"\n{column}:")
        print(
            all_records[column]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    print("\nQuality flags among candidate records:")
    for column in required_flags:
        print(f"\n{column}:")
        print(
            candidates[column]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    print("\nFirst 5 candidate records:")
    preview_columns = [
        "ParticipantID",
        "Footwear",
        "Speed",
        "FootstepID",
        "Exclude",
        "Incomplete",
        "Outlier",
        "record_max",
        "active_value_mean",
    ]

    print(
        candidates[preview_columns]
        .head()
        .to_string(index=False)
    )

    print("\nFiles saved successfully:")
    print(ALL_OUTPUT)
    print(CANDIDATE_OUTPUT)

    print(
        "\nNOTE: The metadata join currently assumes that array "
        "records follow metadata row order. Verify this against "
        "the dataset documentation before research conclusions."
    )


if __name__ == "__main__":
    main()
