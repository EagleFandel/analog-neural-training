from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


RESULTS_DIR = Path("results")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def list_result_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.csv"))


def plot_curve(df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_title="Step", yaxis_title="Value", template="plotly_white")
    return fig


def main():
    st.set_page_config(page_title="ANN Analog-Inspired Training Dashboard", layout="wide")
    st.title("基于模拟计算思想的 ANN 训练优化 - 可视化仪表板")
    st.markdown(
        """
        该页面汇总不同训练范式（数字优化、ODE/辛积分、IMEX、SDE 等）的对比实验结果，
        支持交互式查看 Loss、能量漂移、能耗代理等曲线，并可导出原始数据用于论文与展板。
        """
    )

    files = list_result_files()
    if not files:
        st.warning("尚未找到结果文件，请先运行 experiments/*.py 生成 CSV。")
        return

    file_labels = {f.name: f for f in files}
    selected = st.sidebar.multiselect("选择结果文件", list(file_labels.keys()), default=list(file_labels.keys())[:1])

    st.sidebar.markdown("### 展示设置")
    show_energy = st.sidebar.checkbox("显示能量漂移/能耗列", value=True)
    show_table = st.sidebar.checkbox("显示原始数据表", value=False)

    for name in selected:
        path = file_labels[name]
        st.subheader(f"文件：{name}")
        df = load_csv(path)
        st.caption(f"记录数：{len(df)}，列：{', '.join(df.columns)}")

        # Loss curves
        loss_cols = [col for col in df.columns if "loss" in col]
        if loss_cols:
            st.plotly_chart(plot_curve(df, loss_cols, title="Loss 曲线"), use_container_width=True)

        # Energy metrics
        if show_energy:
            energy_cols = [c for c in df.columns if "energy" in c or "nfe" in c or "flops" in c]
            if energy_cols:
                st.plotly_chart(plot_curve(df, energy_cols, title="能量/能耗指标"), use_container_width=True)

        if show_table:
            st.dataframe(df)

        st.download_button(
            label="下载CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=name,
            mime="text/csv",
        )


if __name__ == "__main__":
    main()



