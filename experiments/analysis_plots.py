from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_loss(df: pd.DataFrame, columns: list[str], title: str, out_path: Path) -> None:
    fig = px.line(df, x="step", y=columns, title=title)
    fig.write_image(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="输入 CSV 文件")
    parser.add_argument("--loss-cols", nargs="*", default=["loss"], help="要绘制的损失列")
    parser.add_argument("--out", type=str, default="results/figures/loss_plot.png")
    args = parser.parse_args()

    path = Path(args.csv)
    df = load_csv(path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_loss(df, args.loss_cols, f"Loss Curves - {path.name}", out_path)
    print(f"已保存图像到 {out_path}")


if __name__ == "__main__":
    main()



