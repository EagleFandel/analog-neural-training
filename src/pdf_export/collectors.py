"""从Dashboard结果构建PDF导出上下文"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.pdf_export.models import (
    FigureArtifact,
    TableArtifact,
    Section,
    ReportContext,
)
from src.pdf_export.logging_utils import get_logger

logger = get_logger(__name__)


class DashboardDataCollector:
    """将Dashboard数据转换为ReportContext"""

    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir

    # region —— 结果文件加载 ——
    def load_all_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not self.results_dir.exists():
            logger.warning("结果目录不存在：%s", self.results_dir)
            return results

        for file in sorted(self.results_dir.glob("*.json")):
            payload = self._load_json(file)
            if payload:
                results.append(payload)

        for file in sorted(self.results_dir.glob("*.csv")):
            payload = self._load_csv(file)
            if payload:
                results.append(payload)

        return results

    def load_result_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        target = Path(filename)
        if not target.is_absolute():
            target = self.results_dir / target
        if not target.exists():
            logger.error("指定文件不存在：%s", target)
            return None

        if target.suffix.lower() == ".json":
            return self._load_json(target)
        if target.suffix.lower() == ".csv":
            return self._load_csv(target)
        logger.error("不支持的文件类型：%s", target)
        return None

    def _load_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.error("加载JSON失败 %s: %s", file_path.name, exc)
            return None

        payload: Dict[str, Any]
        if isinstance(data, list):
            payload = {
                "results": data,
                "dataset": data[0].get("dataset", "unknown") if data else "unknown",
                "_filename": file_path.name,
                "_filetype": "json",
                "_is_array": True,
            }
        elif isinstance(data, dict):
            payload = data
            payload.setdefault("_filename", file_path.name)
            payload.setdefault("_filetype", "json")
            payload.setdefault("_is_array", False)
        else:
            logger.error("JSON格式不支持：%s", file_path.name)
            return None
        return payload

    def _load_csv(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            logger.error("加载CSV失败 %s: %s", file_path.name, exc)
            return None
        return {
            "_filename": file_path.name,
            "_filetype": "csv",
            "_is_array": False,
            "dataframe": df,
            "columns": df.columns.tolist(),
        }

    # endregion

    # region —— ReportContext 构建 ——

    def build_report_context(self, data: Dict[str, Any], all_files: List[Dict[str, Any]]) -> ReportContext:
        metadata = {
            "文件名": data.get("_filename", "unknown"),
            "数据类型": data.get("_filetype", "unknown"),
        }
        dataset = data.get("dataset") or data.get("dataset_name")
        if dataset:
            metadata["数据集"] = dataset

        sections: List[Section] = [
            self._build_loss_section(data),
            self._build_energy_section(data),
            self._build_pareto_section(all_files),
            self._build_overview_section(data),
            self._build_hardware_section(data),
        ]

        context = ReportContext(
            title="模拟计算训练实验报告",
            subtitle=f"结果文件：{data.get('_filename', 'unknown')}",
            dataset_name=metadata.get("数据集"),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata,
            sections=sections,
            source_files=[data.get("_filename", "unknown")],
        )
        return context

    def _build_loss_section(self, data: Dict[str, Any]) -> Section:
        # 提取损失曲线原始数据
        chart_data = self._extract_loss_data(data)
        figure_artifact = FigureArtifact(
            title="训练损失曲线",
            description="展示各优化器在训练过程中的损失下降情况",
            chart_type="line",
            data=chart_data,
        )

        tables: List[TableArtifact] = []
        if data.get("_filetype") == "csv" and data.get("dataframe") is not None:
            df: pd.DataFrame = data["dataframe"]
            tables.append(
                TableArtifact(
                    title="CSV数据样例",
                    description="前10行训练记录",
                    headers=list(df.columns),
                    rows=[self._format_row(row) for row in df.head(10).values],
                )
            )
        elif "results" in data:
            tables.append(
                TableArtifact(
                    title="优化器结果概览",
                    description="来自基准测试的关键指标",
                    headers=["优化器", "最终损失", "测试准确率", "训练步数"],
                    rows=[
                        [
                            r.get("optimizer", "unknown"),
                            self._format_value(r.get("final_loss")),
                            self._format_percentage(r.get("test_accuracy", r.get("test_acc"))),
                            str(r.get("steps_completed", r.get("steps", "-"))),
                        ]
                        for r in data.get("results", [])
                    ],
                )
            )

        return Section(
            title="损失曲线",
            subtitle="Loss Curves",
            summary="展示训练损失随时间的变化情况，用于评估收敛速度。",
            figures=[figure_artifact],
            tables=tables,
        )

    def _build_energy_section(self, data: Dict[str, Any]) -> Section:
        # 提取能耗柱状图原始数据
        chart_data = self._extract_energy_data(data)
        figure_artifact = FigureArtifact(
            title="能耗与计算量对比",
            description="比较优化器在能耗与函数评估次数上的差异",
            chart_type="bar",
            data=chart_data,
        )

        tables: List[TableArtifact] = []
        if "results" in data:
            rows = []
            for r in data["results"]:
                rows.append(
                    [
                        r.get("optimizer", "unknown"),
                        self._format_value(r.get("energy_consumed_j")),
                        str(r.get("nfe", "-")),
                        r.get("type", "unknown"),
                    ]
                )
            tables.append(
                TableArtifact(
                    title="能耗指标",
                    description="基于实验结果的能耗与NFE统计",
                    headers=["优化器", "能耗(J)", "NFE", "类型"],
                    rows=rows,
                )
            )

        return Section(
            title="能耗对比",
            subtitle="Energy Comparison",
            summary="对比数字、模拟或混合架构的能耗表现。",
            figures=[figure_artifact],
            tables=tables,
        )

    def _build_pareto_section(self, all_files: List[Dict[str, Any]]) -> Section:
        # 提取Pareto散点图原始数据
        chart_data = self._extract_pareto_data(all_files)
        figure_artifact = FigureArtifact(
            title="Pareto前沿",
            description="展示跨实验集合中准确率与能耗的Pareto最优点",
            chart_type="scatter",
            data=chart_data,
        )
        return Section(
            title="多目标优化",
            subtitle="Pareto Frontier",
            summary="识别在准确率和能耗之间达到平衡的解。",
            figures=[figure_artifact],
            tables=[],
        )

    def _build_overview_section(self, data: Dict[str, Any]) -> Section:
        # 提取雷达图原始数据
        chart_data = self._extract_radar_data(data)
        figure_artifact = FigureArtifact(
            title="综合效率雷达图",
            description="通过雷达图比较效率、能耗等多项指标。",
            chart_type="radar",
            data=chart_data,
        )

        tables: List[TableArtifact] = []
        if "results" in data:
            rows = []
            for r in data["results"]:
                acc = self._safe_float(r.get("test_accuracy", r.get("test_acc", 0.0)))
                energy = self._safe_float(r.get("energy_consumed_j", 0.0))
                efficiency = acc / energy if energy > 0 else 0.0
                rows.append(
                    [
                        r.get("optimizer", "unknown"),
                        f"{acc*100:.2f}",
                        self._format_value(energy),
                        self._format_value(efficiency),
                    ]
                )
            tables.append(
                TableArtifact(
                    title="效率排名",
                    description="按照准确率/能耗比排序的优化器列表",
                    headers=["优化器", "准确率(%)", "能耗(J)", "效率"],
                    rows=rows,
                )
            )

        return Section(
            title="综合评估",
            subtitle="Optimizer Overview",
            summary="从准确率、能耗、效率等维度综合评估不同优化器。",
            figures=[figure_artifact],
            tables=tables,
        )

    def _build_hardware_section(self, data: Dict[str, Any]) -> Section:
        base_acc = 0.85
        if "results" in data and data["results"]:
            base_acc = self._safe_float(data["results"][0].get("test_accuracy", data["results"][0].get("test_acc", base_acc)))
        
        # 简化硬件影响模拟（避免外部依赖）
        adc_bits = 8
        thermal_noise = 0.0001
        leakage_rate = 0.00001
        quant_loss = (12 - adc_bits) * 0.005
        noise_loss = thermal_noise * 100
        leak_loss = leakage_rate * 50
        predicted = max(0, min(1, base_acc - quant_loss - noise_loss - leak_loss))

        tables = [
            TableArtifact(
                title="硬件影响预测",
                description="使用默认硬件参数估计对准确率的影响",
                headers=["指标", "数值"],
                rows=[
                    ["基准准确率", f"{base_acc*100:.2f}%"],
                    ["预测准确率", f"{predicted*100:.2f}%"],
                    ["性能退化", f"{(base_acc - predicted)*100:.2f}%"],
                ],
            )
        ]

        return Section(
            title="硬件仿真",
            subtitle="Hardware Impact",
            summary="模拟ADC位宽、噪声、泄漏等硬件因素对模型性能的影响。",
            figures=[],
            tables=tables,
            extra_notes="可在Dashboard中调整硬件参数获取更多细节。",
        )

    # endregion

    # region —— 数据提取方法 ——

    def _extract_loss_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取损失曲线数据"""
        series = {}
        
        if data.get("_filetype") == "csv":
            df = data.get("dataframe")
            if df is not None:
                loss_cols = [col for col in df.columns if "loss" in col.lower()]
                for col in loss_cols:
                    series[col] = df[col].tolist()
        elif "results" in data:
            for result in data["results"]:
                if "loss_history" in result:
                    opt_name = result.get("optimizer", "Unknown")
                    series[opt_name] = result["loss_history"]
        elif "loss_history" in data:
            opt_name = data.get("optimizer", "Unknown")
            series[opt_name] = data["loss_history"]
        
        return {"series": series}

    def _extract_energy_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取能耗柱状图数据"""
        if "results" not in data:
            return {"optimizers": [], "energies": [], "nfes": []}
        
        optimizers = []
        energies = []
        nfes = []
        
        for result in data["results"]:
            optimizers.append(result.get("optimizer", "Unknown"))
            energies.append(result.get("energy_consumed_j", 0))
            nfes.append(result.get("nfe", 0))
        
        return {
            "optimizers": optimizers,
            "energies": energies,
            "nfes": nfes,
        }

    def _extract_pareto_data(self, all_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取Pareto前沿散点数据"""
        datasets = []
        
        for data in all_files:
            if "results" not in data:
                continue
            
            dataset_name = data.get("_filename", "unknown")
            accuracies = []
            energies = []
            labels = []
            
            for result in data["results"]:
                acc = result.get("test_accuracy", result.get("test_acc", 0))
                energy = result.get("energy_consumed_j", 0)
                if energy > 0:
                    accuracies.append(acc * 100)
                    energies.append(energy)
                    labels.append(result.get("optimizer", "Unknown"))
            
            if accuracies:
                datasets.append({
                    "name": dataset_name,
                    "x": energies,
                    "y": accuracies,
                    "labels": labels,
                })
        
        return {"datasets": datasets}

    def _extract_radar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取雷达图数据"""
        if "results" not in data:
            return {"categories": [], "series": {}}
        
        categories = ["准确率", "能效", "速度", "NFE效率"]
        series = {}
        
        for result in data["results"]:
            optimizer = result.get("optimizer", "Unknown")
            
            acc = result.get("test_accuracy", result.get("test_acc", 0))
            energy = result.get("energy_consumed_j", 1)
            time_s = result.get("time_seconds", 1)
            nfe = result.get("nfe", 1)
            
            energy_efficiency = acc / energy if energy > 0 else 0
            speed = acc / time_s if time_s > 0 else 0
            nfe_efficiency = acc / nfe if nfe > 0 else 0
            
            values = [
                acc,
                min(1.0, energy_efficiency * 10),
                min(1.0, speed * 10),
                min(1.0, nfe_efficiency * 100),
            ]
            
            series[optimizer] = values
        
        return {
            "categories": categories,
            "series": series,
        }

    # endregion

    # region —— 工具方法 ——

    def context_to_summary(self, context: ReportContext) -> Dict[str, Any]:
        return context.to_summary()

    def context_to_json(self, context: ReportContext) -> str:
        return json.dumps(context.to_summary(), ensure_ascii=False, indent=2)

    def _format_value(self, value: Any) -> str:
        if value is None or value == "":
            return "-"
        if isinstance(value, (int, float)):
            return f"{value:.4f}" if isinstance(value, float) else str(value)
        return str(value)

    def _format_percentage(self, value: Any) -> str:
        val = self._safe_float(value)
        return f"{val*100:.2f}%"

    def _format_row(self, row: Any) -> List[str]:
        return [self._format_value(cell) for cell in row]

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # endregion


__all__ = ["DashboardDataCollector"]
