"""
cli.py
------
Lightweight command-line prototype (no browser needed).

Usage:
    python cli.py path/to/dataset.csv target_column_name

Example:
    python cli.py data/sample_titanic.csv Survived
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from router import MetaMLRouter  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print("Usage: python cli.py <path_to_csv> <target_column>")
        sys.exit(1)

    csv_path, target_col = sys.argv[1], sys.argv[2]
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        print(f"Error: column '{target_col}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    router = MetaMLRouter()
    result = router.recommend(X, y)
    router.pretty_print(result)


if __name__ == "__main__":
    main()
