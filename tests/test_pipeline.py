from pathlib import Path
import pandas as pd


def test_audit_csv_exists_and_valid():
    audit_file = Path(__file__).parent.parent / "audit.csv"
    assert audit_file.exists(), "audit.csv should exist"

    df = pd.read_csv(audit_file)
    required_cols = {"path", "class", "file_size", "sha256", "width", "height", "mode", "format", "corrupt", "is_duplicate"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in audit.csv: {required_cols - set(df.columns)}"
    assert len(df) == 975, f"Expected 975 images, found {len(df)}"
    assert set(df["class"].unique()) == {"ai", "real"}


def test_splits_files_exist_and_non_empty():
    splits_dir = Path(__file__).parent.parent / "splits"
    assert splits_dir.exists()

    for split_name in ["train.txt", "val.txt", "test.txt"]:
        split_file = splits_dir / split_name
        assert split_file.exists(), f"{split_name} must exist"
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) > 1, f"{split_name} must contain data"
        # First line is header 'path,label'
        assert lines[0] == "path,label"
        # Data rows
        for line in lines[1:10]:
            parts = line.split(",")
            assert len(parts) == 2, f"Invalid split format in line: {line}"
            assert parts[1] in ["0", "1"], f"Invalid label {parts[1]}"
