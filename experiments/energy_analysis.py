from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_frames(pattern: str) -> list[pd.DataFrame]:
    files = list(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有匹配的结果文件：{pattern}")
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        df["source"] = path.name
        frames.append(df)
    return frames


def compute_energy(df: pd.DataFrame, flops_per_eval: float, time_per_eval: float) -> pd.DataFrame:
    df = df.copy()
    # 优先使用显式的 NFE/迭代计数列
    nfe_col = None
    for candidate in ["nfe", "NFE", "function_eval", "cg_iters"]:
        if candidate in df.columns:
            nfe_col = candidate
            break

    if nfe_col is not None:
        # 将每步的评估次数累加为累计 NFE
        per_step = pd.to_numeric(df[nfe_col], errors="coerce").fillna(0)
        df["nfe"] = per_step.cumsum()
    else:
        # 回退：根据文件名或列名推断每步函数评估次数（例如 RK4=4），并累积
        source = df["source"].iloc[0] if "source" in df.columns and len(df) > 0 else ""
        cols = set(df.columns)
        if ("rk4_loss" in cols) or ("rk4" in source.lower()):
            per_eval = 4
        elif ("rk2" in source.lower()):
            per_eval = 2
        elif ("euler" in source.lower()):
            per_eval = 1
        elif ("imex" in source.lower()):
            # 若无 cg_iters，则保守估计每步一次隐式求解等价 1 次评估
            per_eval = 1
        else:
            per_eval = 1
        steps_series = df["step"] if "step" in df.columns else pd.Series(range(len(df)))
        df["nfe"] = (steps_series.astype(float) + 1.0) * float(per_eval)

    df["flops_energy"] = df["nfe"] * flops_per_eval
    df["time_energy"] = df["nfe"] * time_per_eval
    return df


def summarize(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    return df.groupby("source")[value_cols].agg(["mean", "std", "min", "max", "last"])


def main():
    parser = argparse.ArgumentParser(description="基于 NFE 等指标估算能耗代理并输出统计。")
    parser.add_argument("--pattern", required=True, help="结果 CSV 匹配模式，如 'results/*rk*.csv'")
    parser.add_argument("--flops-per-eval", type=float, default=1e6, help="单次函数评估的 FLOPs 估计")
    parser.add_argument("--time-per-eval", type=float, default=1e-4, help="单次函数评估耗时（秒）的估计")
    parser.add_argument("--out", type=str, default="results/summary_energy.csv")
    args = parser.parse_args()

    frames = load_frames(args.pattern)
    processed = []
    for df in frames:
        processed.append(compute_energy(df, args.flops_per_eval, args.time_per_eval))
    concat_df = pd.concat(processed, ignore_index=True)
    summary_cols = [c for c in ["loss", "rk4_loss", "imex_loss", "train_loss", "val_loss", "flops_energy", "time_energy"] if c in concat_df.columns]
    summary = summarize(concat_df, summary_cols)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path)
    print(summary)


if __name__ == "__main__":
    main()



