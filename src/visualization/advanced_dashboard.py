"""
增强可视化Dashboard

提供交互式能耗对比、Pareto前沿、硬件参数调节等高级功能
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PDF导出模块
PDF_EXPORT_AVAILABLE = True
try:
    from src.pdf_export import DashboardDataCollector, ReportLabExporter
except ImportError as e:
    PDF_EXPORT_AVAILABLE = False
    PDF_IMPORT_ERROR = str(e)

RESULTS_DIR = Path("results")


def load_all_results() -> List[Dict]:
    """加载所有结果文件（JSON和CSV）"""
    if not RESULTS_DIR.exists():
        return []
    
    results = []
    
    # 加载JSON文件
    for file in RESULTS_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 检查数据类型
                if isinstance(data, list):
                    # 如果是数组，包装成对象
                    wrapped_data = {
                        "results": data,
                        "dataset": data[0].get("dataset", "unknown") if data else "unknown",
                        "_filename": file.name,
                        "_filetype": "json",
                        "_is_array": True
                    }
                    results.append(wrapped_data)
                elif isinstance(data, dict):
                    # 如果是字典，直接添加元数据
                    data["_filename"] = file.name
                    data["_filetype"] = "json"
                    data["_is_array"] = False
                    results.append(data)
        except Exception as e:
            st.warning(f"无法加载 {file.name}: {e}")
    
    # 加载CSV文件
    for file in RESULTS_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            data = {
                "_filename": file.name,
                "_filetype": "csv",
                "dataframe": df,
                "columns": df.columns.tolist()
            }
            results.append(data)
        except Exception as e:
            st.warning(f"无法加载 {file.name}: {e}")
    
    return results


def load_json_results(pattern: str = "*.json") -> List[Dict]:
    """加载JSON格式的结果文件（保持向后兼容）"""
    if not RESULTS_DIR.exists():
        return []
    
    results = []
    for file in RESULTS_DIR.glob(pattern):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_filename"] = file.name
                results.append(data)
        except Exception as e:
            st.warning(f"无法加载 {file.name}: {e}")
    
    return results


def plot_loss_curves(data: Dict) -> go.Figure:
    """绘制损失曲线"""
    fig = go.Figure()
    
    if data.get("_filetype") == "csv":
        # 处理CSV数据
        df = data.get("dataframe")
        if df is not None:
            # 找到所有包含"loss"的列
            loss_cols = [col for col in df.columns if "loss" in col.lower()]
            for col in loss_cols:
                fig.add_trace(go.Scatter(
                    y=df[col],
                    mode="lines",
                    name=col
                ))
    elif "results" in data:  # 基准测试格式
        for result in data["results"]:
            if "loss_history" in result:
                fig.add_trace(go.Scatter(
                    y=result["loss_history"],
                    mode="lines",
                    name=result["optimizer"]
                ))
    elif "loss_history" in data:  # 单个结果
        fig.add_trace(go.Scatter(
            y=data["loss_history"],
            mode="lines",
            name=data.get("optimizer", "Unknown")
        ))
    
    fig.update_layout(
        title="损失曲线",
        xaxis_title="步数",
        yaxis_title="损失",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig


def plot_energy_comparison(data: Dict) -> go.Figure:
    """能耗对比柱状图"""
    if "results" not in data:
        return go.Figure()
    
    optimizers = []
    energies = []
    nfes = []
    
    for result in data["results"]:
        optimizers.append(result.get("optimizer", "Unknown"))
        energies.append(result.get("energy_consumed_j", 0))
        nfes.append(result.get("nfe", 0))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("总能耗 (Joules)", "函数评估次数 (NFE)")
    )
    
    fig.add_trace(
        go.Bar(x=optimizers, y=energies, name="能耗", marker_color="indianred"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=optimizers, y=nfes, name="NFE", marker_color="lightsalmon"),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="能耗与计算量对比",
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def plot_pareto_frontier(benchmark_results: List[Dict]) -> go.Figure:
    """绘制Pareto前沿（准确率 vs 能耗）"""
    fig = go.Figure()
    
    for data in benchmark_results:
        if "results" not in data:
            continue
        
        dataset_name = data.get("_filename", "unknown")
        
        accuracies = []
        energies = []
        labels = []
        
        for result in data["results"]:
            acc = result.get("test_accuracy", result.get("test_acc", 0))
            energy = result.get("energy_consumed_j", 0)
            if energy > 0:  # 只显示有能耗数据的
                accuracies.append(acc * 100)  # 转换为百分比
                energies.append(energy)
                labels.append(result.get("optimizer", "Unknown"))
        
        if accuracies:
            fig.add_trace(go.Scatter(
                x=energies,
                y=accuracies,
                mode="markers+text",
                text=labels,
                textposition="top center",
                name=dataset_name,
                marker=dict(size=12)
            ))
    
    fig.update_layout(
        title="Pareto前沿：准确率 vs 能耗",
        xaxis_title="能耗 (Joules)",
        yaxis_title="测试准确率 (%)",
        template="plotly_white",
        hovermode="closest"
    )
    
    # 添加Pareto前沿线（理想情况）
    if len(fig.data) > 0:
        all_x = []
        all_y = []
        for trace in fig.data:
            all_x.extend(trace.x)
            all_y.extend(trace.y)
        
        if all_x and all_y:
            # 找到Pareto最优点
            points = list(zip(all_x, all_y))
            pareto_points = []
            for p in points:
                dominated = False
                for q in points:
                    if q[0] < p[0] and q[1] >= p[1]:  # q更省能耗且准确率不低
                        dominated = True
                        break
                if not dominated:
                    pareto_points.append(p)
            
            if pareto_points:
                pareto_points.sort(key=lambda p: p[0])
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pareto_points],
                    y=[p[1] for p in pareto_points],
                    mode="lines",
                    name="Pareto前沿",
                    line=dict(color="red", dash="dash")
                ))
    
    return fig


def plot_efficiency_radar(data: Dict) -> go.Figure:
    """效率雷达图"""
    if "results" not in data:
        return go.Figure()
    
    fig = go.Figure()
    
    categories = ["准确率", "能效", "速度", "NFE效率"]
    
    for result in data["results"]:
        optimizer = result.get("optimizer", "Unknown")
        
        # 归一化指标（0-1）
        acc = result.get("test_accuracy", result.get("test_acc", 0))
        energy = result.get("energy_consumed_j", 1)
        time_s = result.get("time_seconds", 1)
        nfe = result.get("nfe", 1)
        
        # 能效 = 准确率 / 能耗
        energy_efficiency = acc / energy if energy > 0 else 0
        
        # 速度 = 准确率 / 时间
        speed = acc / time_s if time_s > 0 else 0
        
        # NFE效率 = 准确率 / NFE
        nfe_efficiency = acc / nfe if nfe > 0 else 0
        
        # 归一化到0-1范围（这里简化处理）
        values = [
            acc,
            min(1.0, energy_efficiency * 10),  # 缩放
            min(1.0, speed * 10),
            min(1.0, nfe_efficiency * 100),
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=optimizer
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="优化器综合效率雷达图",
        template="plotly_white"
    )
    
    return fig


def simulate_hardware_impact(
    base_accuracy: float,
    adc_bits: int,
    thermal_noise: float,
    leakage_rate: float
) -> float:
    """模拟硬件参数对准确率的影响"""
    # 简化模型：准确率随精度和噪声降低
    
    # 量化损失
    quant_loss = (12 - adc_bits) * 0.005  # 每减少1bit损失0.5%
    
    # 噪声损失
    noise_loss = thermal_noise * 100  # 噪声越大损失越多
    
    # 泄漏损失
    leak_loss = leakage_rate * 50
    
    degraded_acc = base_accuracy - quant_loss - noise_loss - leak_loss
    
    return max(0, min(1, degraded_acc))


def main():
    st.set_page_config(page_title="模拟计算训练系统 - 高级Dashboard", layout="wide")
    
    st.title("🔬 模拟计算启发式神经网络训练 - 高级可视化")
    
    st.markdown("""
    该Dashboard提供：
    - 📊 多维度性能对比
    - ⚡ 实时能耗分析
    - 🎯 Pareto前沿优化
    - 🔧 硬件参数交互调节
    - 📈 参数空间轨迹可视化
    """)
    
    # 侧边栏：数据加载与设置
    st.sidebar.header("⚙️ 设置")
    
    # 加载所有结果（JSON + CSV）
    all_files = load_all_results()
    
    if not all_files:
        st.warning("未找到结果文件。请先运行实验生成数据。")
        st.info("""
        运行以下命令生成数据：
        ```bash
        python experiments/benchmark_suite.py --all
        python experiments/edge_device_demo.py --scenario all
        ```
        """)
        return
    
    # 显示文件列表（带类型标识）
    file_options = {}
    for f in all_files:
        filename = f["_filename"]
        filetype = f.get("_filetype", "unknown")
        display_name = f"{filename} ({filetype.upper()})"
        file_options[display_name] = f
    
    selected_file_display = st.sidebar.selectbox(
        "选择结果文件",
        list(file_options.keys())
    )
    
    data = file_options[selected_file_display]
    
    # Tab布局
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 损失曲线",
        "⚡ 能耗对比",
        "🎯 Pareto前沿",
        "📊 综合评估",
        "🔧 硬件仿真"
    ])
    
    with tab1:
        st.subheader("训练损失曲线")
        fig_loss = plot_loss_curves(data)
        st.plotly_chart(fig_loss, use_container_width=True)
        
        if data.get("_filetype") == "csv":
            # CSV数据显示
            st.subheader("CSV数据")
            df = data.get("dataframe")
            if df is not None:
                st.caption(f"记录数：{len(df)}，列：{', '.join(df.columns)}")
                st.dataframe(df, use_container_width=True)
                
                # 下载按钮
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载CSV",
                    data=csv_data,
                    file_name=data["_filename"],
                    mime="text/csv"
                )
        elif "results" in data:
            st.subheader("详细数据")
            results_df = pd.DataFrame([
                {
                    "优化器": r.get("optimizer", "Unknown"),
                    "最终损失": r.get("final_loss", 0),
                    "测试准确率": r.get("test_accuracy", r.get("test_acc", 0)),
                    "训练步数": r.get("steps_completed", r.get("steps", 0)),
                }
                for r in data["results"]
            ])
            st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        st.subheader("能耗与计算量对比")
        fig_energy = plot_energy_comparison(data)
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # 能耗分析
        st.subheader("数字 vs 模拟架构能耗估算")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_params = st.number_input("参数数量", min_value=1000, max_value=10000000, value=50000, step=1000)
        with col2:
            num_steps = st.number_input("训练步数", min_value=10, max_value=10000, value=100, step=10)
        
        if st.button("计算能耗对比"):
            from src.hardware.energy_models import compare_digital_vs_analog
            
            comparison = compare_digital_vs_analog(num_params, num_steps)
            
            st.metric("纯数字架构", f"{comparison['digital_total_joules']:.6f} J")
            st.metric("纯模拟架构", f"{comparison['analog_total_joules']:.6f} J", 
                     delta=f"节能 {comparison['energy_savings_analog']:.1f}%")
            st.metric("混合架构", f"{comparison['hybrid_total_joules']:.6f} J",
                     delta=f"节能 {comparison['energy_savings_hybrid']:.1f}%")
            
            st.success(f"✅ 模拟架构加速: {comparison['analog_speedup']:.1f}×")
    
    with tab3:
        st.subheader("Pareto前沿：准确率 vs 能耗")
        st.markdown("显示所有实验的多目标优化前沿")
        
        fig_pareto = plot_pareto_frontier(all_files)
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        st.info("""
        **Pareto前沿解读**：
        - 越靠左上方的点越优（高准确率、低能耗）
        - 红色虚线表示非支配解集合
        - 选择优化器时可根据能耗预算在前沿上权衡
        """)
    
    with tab4:
        st.subheader("优化器综合评估")
        
        fig_radar = plot_efficiency_radar(data)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 效率排名
        if "results" in data:
            st.subheader("效率排名")
            
            efficiency_data = []
            for r in data["results"]:
                acc = r.get("test_accuracy", r.get("test_acc", 0))
                energy = r.get("energy_consumed_j", 1)
                efficiency = acc / energy if energy > 0 else 0
                
                efficiency_data.append({
                    "优化器": r.get("optimizer", "Unknown"),
                    "准确率": f"{acc*100:.2f}%",
                    "能耗(J)": f"{energy:.4f}",
                    "效率": f"{efficiency:.4f}",
                    "类型": r.get("type", "unknown")
                })
            
            efficiency_df = pd.DataFrame(efficiency_data)
            efficiency_df = efficiency_df.sort_values("效率", ascending=False)
            st.dataframe(efficiency_df, use_container_width=True)
    
    with tab5:
        st.subheader("🔧 硬件参数交互仿真")
        st.markdown("调节模拟电路参数，观察对性能的影响")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 硬件参数")
            adc_bits = st.slider("ADC位宽", min_value=4, max_value=16, value=8, step=1)
            thermal_noise = st.slider("热噪声强度", min_value=0.0, max_value=0.01, value=0.0001, step=0.0001, format="%.4f")
            leakage_rate = st.slider("电容泄漏率", min_value=0.0, max_value=0.001, value=0.00001, step=0.00001, format="%.5f")
        
        with col2:
            st.markdown("### 性能影响预测")
            
            # 使用第一个结果作为基准
            if "results" in data and len(data["results"]) > 0:
                base_acc = data["results"][0].get("test_accuracy", data["results"][0].get("test_acc", 0.85))
            else:
                base_acc = 0.85
            
            predicted_acc = simulate_hardware_impact(base_acc, adc_bits, thermal_noise, leakage_rate)
            
            st.metric("基准准确率", f"{base_acc*100:.2f}%")
            st.metric("预测准确率", f"{predicted_acc*100:.2f}%", 
                     delta=f"{(predicted_acc-base_acc)*100:.2f}%")
            
            # 安全的性能退化计算（避免除零）
            if base_acc > 1e-6:
                degradation = (base_acc - predicted_acc) / base_acc * 100
                if degradation < 5:
                    st.success("✅ 性能退化可接受 (< 5%)")
                elif degradation < 10:
                    st.warning("⚠️ 性能退化中等 (5-10%)")
                else:
                    st.error("❌ 性能退化严重 (> 10%)")
            else:
                st.info("⚠️ 基准准确率过低，无法计算性能退化")
        
        # 参数扫描
        st.markdown("### 参数敏感性分析")
        
        if st.button("运行参数扫描"):
            bits_range = range(4, 13)
            accs = [simulate_hardware_impact(base_acc, b, thermal_noise, leakage_rate) for b in bits_range]
            
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=list(bits_range),
                y=[a*100 for a in accs],
                mode="lines+markers",
                name="准确率 vs ADC位宽"
            ))
            fig_sweep.update_layout(
                title="ADC位宽对准确率的影响",
                xaxis_title="ADC位宽 (bits)",
                yaxis_title="预测准确率 (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig_sweep, use_container_width=True)
    
    # 底部：导出功能
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 导出")
    
    # 直接显示下载按钮（不需要先点击触发）
    if data.get("_filetype") == "csv":
        # CSV格式下载
        csv_data = data.get("dataframe").to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            label="📥 下载CSV数据",
            data=csv_data,
            file_name=data["_filename"],
            mime="text/csv",
            use_container_width=True
        )
    else:
        # JSON格式下载
        if data.get("_is_array"):
            # 下载原始数组
            download_data = data.get("results", [])
        else:
            # 清理元数据字段
            download_data = {k: v for k, v in data.items() if not k.startswith("_")}
        
        st.sidebar.download_button(
            label="📥 下载JSON数据",
            data=json.dumps(download_data, ensure_ascii=False, indent=2),
            file_name=data["_filename"],
            mime="application/json",
            use_container_width=True
        )
    
    # PDF导出
    if not PDF_EXPORT_AVAILABLE:
        st.sidebar.warning("PDF导出功能不可用")
        st.sidebar.caption("请安装: pip install reportlab pyyaml")
    else:
        if st.sidebar.button("生成PDF报告", use_container_width=True, type="primary"):
            with st.spinner("正在生成PDF..."):
                try:
                    # 构建报告上下文
                    st.sidebar.text("步骤 1/3: 收集数据...")
                    collector = DashboardDataCollector(results_dir=Path(RESULTS_DIR))
                    context = collector.build_report_context(data, all_files)
                    
                    # 生成PDF
                    st.sidebar.text("步骤 2/3: 渲染图表...")
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"report_{timestamp}.pdf"
                    output_path = Path("reports") / output_filename
                    
                    exporter = ReportLabExporter(output_path)
                    
                    st.sidebar.text("步骤 3/3: 生成PDF文件...")
                    exporter.generate(context)
                    
                    # 提供下载
                    with open(output_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.sidebar.success("PDF生成成功!")
                    st.sidebar.download_button(
                        label="下载PDF报告",
                        data=pdf_bytes,
                        file_name=output_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.sidebar.error(f"PDF生成失败: {e}")
                    st.sidebar.caption("详细日志: logs/pdf_export.log")


if __name__ == "__main__":
    main()

