from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate(csv_files: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["source"] = path.name
        dfs.append(df)
    if not dfs:
        raise ValueError("No CSV files to aggregate")
    return pd.concat(dfs, ignore_index=True)


def summarize(df: pd.DataFrame, value_cols: list[str], group_col: str = "source") -> pd.DataFrame:
    return df.groupby(group_col)[value_cols].agg(["mean", "std", "min", "max"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", type=str, help="glob pattern, e.g. 'results/sine_*.csv'")
    args = parser.parse_args()

    files = list(Path().glob(args.pattern))
    if not files:
        raise SystemExit("No files matched pattern")

    df = aggregate(files)
    value_cols = [c for c in df.columns if c not in {"step", "source"}]
    summary = summarize(df, value_cols)
    print(summary)


if __name__ == "__main__":
    main()



