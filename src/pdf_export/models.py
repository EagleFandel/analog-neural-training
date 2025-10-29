"""ReportLab PDF 导出数据模型"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class FigureArtifact:
    """图表数据，用于 ReportLab 原生渲染"""

    title: str
    description: Optional[str]
    chart_type: str  # "line", "bar", "scatter", "radar"
    data: Dict[str, Any]  # 原始数据：{"x": [...], "y": [...], "series": {...}}
    width: int = 400
    height: int = 300

    def to_summary(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "chart_type": self.chart_type,
            "data_keys": list(self.data.keys()),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TableArtifact:
    """Dataframe / 表格数据，用于渲染"""

    title: str
    description: Optional[str]
    headers: List[str]
    rows: List[List[str]]

    def to_summary(self) -> Dict[str, Any]:
        preview_rows = self.rows[:5]
        return {
            "title": self.title,
            "description": self.description,
            "headers": self.headers,
            "row_count": len(self.rows),
            "preview": preview_rows,
        }


@dataclass
class Section:
    """报告章节，包含图表、表格与说明文本"""

    title: str
    subtitle: Optional[str]
    summary: Optional[str]
    figures: List[FigureArtifact] = field(default_factory=list)
    tables: List[TableArtifact] = field(default_factory=list)
    extra_notes: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "summary": self.summary,
            "figure_count": len(self.figures),
            "table_count": len(self.tables),
            "figures": [fig.to_summary() for fig in self.figures],
            "tables": [table.to_summary() for table in self.tables],
            "extra_notes": self.extra_notes,
        }


@dataclass
class ReportContext:
    """生成 PDF 所需的完整上下文"""

    title: str
    subtitle: Optional[str]
    dataset_name: Optional[str]
    generated_at: str
    metadata: Dict[str, str]
    sections: List[Section]
    source_files: List[str] = field(default_factory=list)
    debug_info: Optional[Dict[str, str]] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "dataset_name": self.dataset_name,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "section_count": len(self.sections),
            "sections": [section.to_summary() for section in self.sections],
            "source_files": self.source_files,
            "debug_info": self.debug_info,
        }


@dataclass
class ExportConfig:
    """导出相关配置"""

    output_path: Path
    include_sections: Optional[List[str]] = None
    image_quality: str = "standard"  # standard / high
    page_size: str = "A4"
    margin_mm: int = 20
    font_family: str = "Helvetica"
    log_to_file: bool = True

    def to_summary(self) -> Dict[str, Any]:
        data = asdict(self)
        data["output_path"] = str(self.output_path)
        return data
