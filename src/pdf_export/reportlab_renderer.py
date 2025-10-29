"""ReportLab PDF 渲染器"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.spider import SpiderChart
from reportlab.graphics.widgets.markers import makeMarker

from src.pdf_export.models import ReportContext, Section, FigureArtifact, TableArtifact
from src.pdf_export.logging_utils import get_logger

logger = get_logger(__name__)


class ReportLabExporter:
    """使用 ReportLab 生成 PDF 报告"""

    def __init__(
        self,
        output_path: Path,
        page_size=A4,
        margin_mm: int = 20,
    ):
        self.output_path = output_path
        self.page_size = page_size
        self.margin = margin_mm * mm
        self.styles = getSampleStyleSheet()
        self._setup_chinese_font()
        self._setup_styles()

    def _setup_chinese_font(self):
        """设置中文字体支持"""
        import os
        import platform
        
        # 尝试多种中文字体方案
        font_loaded = False
        
        # 方案1: 尝试使用系统字体文件
        system = platform.system()
        font_paths = []
        
        if system == "Windows":
            # Windows 系统字体路径
            win_fonts = [
                r"C:\Windows\Fonts\simhei.ttf",  # 黑体
                r"C:\Windows\Fonts\simsun.ttc",  # 宋体
                r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
            ]
            font_paths.extend(win_fonts)
        elif system == "Linux":
            # Linux 常见字体路径
            linux_fonts = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            ]
            font_paths.extend(linux_fonts)
        elif system == "Darwin":  # macOS
            mac_fonts = [
                "/System/Library/Fonts/PingFang.ttc",
                "/Library/Fonts/Songti.ttc",
            ]
            font_paths.extend(mac_fonts)
        
        # 尝试注册字体文件
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    self.chinese_font = 'ChineseFont'
                    font_loaded = True
                    logger.info("成功加载系统字体: %s", font_path)
                    break
                except Exception as e:
                    logger.debug("无法加载字体 %s: %s", font_path, e)
                    continue
        
        # 方案2: 使用 ReportLab 内置 CID 字体（备选）
        if not font_loaded:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
                self.chinese_font = 'STSong-Light'
                font_loaded = True
                logger.info("使用 ReportLab 内置字体: STSong-Light")
            except Exception as e:
                logger.warning("无法加载 CID 字体: %s", e)
        
        # 方案3: 回退到 Helvetica（仅支持英文）
        if not font_loaded:
            self.chinese_font = 'Helvetica'
            logger.warning("未找到中文字体，使用 Helvetica（仅支持英文）")

    def _setup_styles(self):
        """设置样式"""
        # 标题样式
        self.styles.add(
            ParagraphStyle(
                name="ChineseTitle",
                parent=self.styles["Title"],
                fontName=self.chinese_font,
                fontSize=24,
                textColor=colors.HexColor("#1f77b4"),
                spaceAfter=12,
            )
        )

        # 副标题样式
        self.styles.add(
            ParagraphStyle(
                name="ChineseSubtitle",
                parent=self.styles["Normal"],
                fontName=self.chinese_font,
                fontSize=14,
                textColor=colors.grey,
                spaceAfter=20,
            )
        )

        # 章节标题样式
        self.styles.add(
            ParagraphStyle(
                name="ChineseSectionTitle",
                parent=self.styles["Heading1"],
                fontName=self.chinese_font,
                fontSize=18,
                textColor=colors.HexColor("#2ca02c"),
                spaceAfter=10,
                spaceBefore=20,
            )
        )

        # 小节标题样式
        self.styles.add(
            ParagraphStyle(
                name="ChineseSubsectionTitle",
                parent=self.styles["Heading2"],
                fontName=self.chinese_font,
                fontSize=14,
                textColor=colors.HexColor("#d62728"),
                spaceAfter=8,
            )
        )

        # 正文样式
        self.styles.add(
            ParagraphStyle(
                name="ChineseBody",
                parent=self.styles["Normal"],
                fontName=self.chinese_font,
                fontSize=10,
                leading=14,
                spaceAfter=10,
            )
        )

    def generate(self, context: ReportContext) -> None:
        """生成 PDF"""
        logger.info("开始生成 PDF: %s", self.output_path)

        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建文档
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )

        # 构建内容
        story = []

        # 封面
        story.extend(self._build_cover(context))
        story.append(PageBreak())

        # 各章节
        for section in context.sections:
            story.extend(self._build_section(section))
            story.append(Spacer(1, 12))

        # 生成 PDF
        try:
            doc.build(story)
            logger.info("PDF 生成成功: %s", self.output_path)
        except Exception as e:
            logger.error("PDF 生成失败: %s", e, exc_info=True)
            raise

    def _build_cover(self, context: ReportContext) -> List:
        """构建封面"""
        elements = []

        # 标题
        elements.append(Spacer(1, 50))
        elements.append(Paragraph(context.title, self.styles["ChineseTitle"]))

        # 副标题
        if context.subtitle:
            elements.append(Paragraph(context.subtitle, self.styles["ChineseSubtitle"]))

        # 元数据
        elements.append(Spacer(1, 20))
        meta_data = [
            ["生成时间", context.generated_at],
        ]
        if context.dataset_name:
            meta_data.append(["数据集", context.dataset_name])

        for key, value in context.metadata.items():
            meta_data.append([key, str(value)])

        meta_table = Table(meta_data, colWidths=[80, 300])
        meta_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), self.chinese_font, 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                    ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        elements.append(meta_table)

        # 章节概览
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("报告内容", self.styles["ChineseSectionTitle"]))
        for idx, section in enumerate(context.sections, 1):
            section_info = f"{idx}. {section.title}"
            if section.subtitle:
                section_info += f" ({section.subtitle})"
            elements.append(Paragraph(section_info, self.styles["ChineseBody"]))
            elements.append(Spacer(1, 5))

        return elements

    def _build_section(self, section: Section) -> List:
        """构建章节"""
        elements = []

        # 章节标题
        title_text = section.title
        if section.subtitle:
            title_text += f" - {section.subtitle}"
        elements.append(Paragraph(title_text, self.styles["ChineseSectionTitle"]))

        # 章节摘要
        if section.summary:
            elements.append(Paragraph(section.summary, self.styles["ChineseBody"]))
            elements.append(Spacer(1, 10))

        # 图表
        for figure in section.figures:
            fig_elements = self._build_figure(figure)
            if fig_elements:
                elements.append(KeepTogether(fig_elements))
                elements.append(Spacer(1, 15))

        # 表格
        for table in section.tables:
            table_elements = self._build_table(table)
            if table_elements:
                elements.append(KeepTogether(table_elements))
                elements.append(Spacer(1, 15))

        # 额外备注
        if section.extra_notes:
            elements.append(Paragraph(f"<i>{section.extra_notes}</i>", self.styles["ChineseBody"]))

        return elements

    def _build_figure(self, figure: FigureArtifact) -> List:
        """构建图表"""
        elements = []

        # 图表标题
        elements.append(Paragraph(figure.title, self.styles["ChineseSubsectionTitle"]))

        # 图表描述
        if figure.description:
            elements.append(Paragraph(figure.description, self.styles["ChineseBody"]))
            elements.append(Spacer(1, 5))

        # 根据类型渲染图表
        try:
            if figure.chart_type == "line":
                chart = self._create_line_chart(figure)
            elif figure.chart_type == "bar":
                chart = self._create_bar_chart(figure)
            elif figure.chart_type == "scatter":
                chart = self._create_scatter_chart(figure)
            elif figure.chart_type == "radar":
                chart = self._create_radar_chart(figure)
            else:
                logger.warning("不支持的图表类型: %s", figure.chart_type)
                return elements

            if chart:
                elements.append(chart)
        except Exception as e:
            logger.error("图表渲染失败 (%s): %s", figure.title, e, exc_info=True)
            elements.append(Paragraph(f"<i>图表渲染失败: {e}</i>", self.styles["ChineseBody"]))

        return elements

    def _create_line_chart(self, figure: FigureArtifact) -> Optional[Drawing]:
        """创建线图"""
        data_dict = figure.data.get("series", {})
        if not data_dict:
            logger.warning("线图数据为空")
            return None

        # 准备数据
        series_data = []
        for series_name, values in data_dict.items():
            if values:
                series_data.append([(i, v) for i, v in enumerate(values)])

        if not series_data:
            return None

        # 创建绘图
        drawing = Drawing(figure.width, figure.height)
        lp = LinePlot()
        lp.x = 30
        lp.y = 30
        lp.width = figure.width - 60
        lp.height = figure.height - 60
        lp.data = series_data

        # 样式
        lp.joinedLines = 1
        lp.strokeColor = colors.blue
        lp.lines[0].strokeWidth = 2

        # 多系列颜色
        color_palette = [colors.blue, colors.red, colors.green, colors.orange, colors.purple]
        for i in range(min(len(series_data), len(color_palette))):
            lp.lines[i].strokeColor = color_palette[i]
            lp.lines[i].strokeWidth = 1.5

        # 坐标轴
        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = max(len(d) for d in series_data) if series_data else 100
        lp.xValueAxis.labels.fontName = self.chinese_font  # 支持中文
        lp.yValueAxis.valueMin = 0
        lp.yValueAxis.labels.fontName = self.chinese_font  # 支持中文

        drawing.add(lp)
        return drawing

    def _create_bar_chart(self, figure: FigureArtifact) -> Optional[Drawing]:
        """创建柱状图"""
        optimizers = figure.data.get("optimizers", [])
        energies = figure.data.get("energies", [])

        if not optimizers or not energies:
            logger.warning("柱状图数据为空")
            return None

        drawing = Drawing(figure.width, figure.height)
        bc = VerticalBarChart()
        bc.x = 30
        bc.y = 30
        bc.width = figure.width - 60
        bc.height = figure.height - 80

        # 数据
        bc.data = [energies]

        # 类别标签
        bc.categoryAxis.categoryNames = optimizers
        bc.categoryAxis.labels.angle = 45
        bc.categoryAxis.labels.fontSize = 8
        bc.categoryAxis.labels.fontName = self.chinese_font  # 支持中文

        # 值轴
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(energies) * 1.1 if energies else 1
        bc.valueAxis.labels.fontName = self.chinese_font  # 支持中文

        # 样式
        bc.bars[0].fillColor = colors.HexColor("#1f77b4")

        drawing.add(bc)
        return drawing

    def _create_scatter_chart(self, figure: FigureArtifact) -> Optional[Drawing]:
        """创建散点图"""
        datasets = figure.data.get("datasets", [])
        if not datasets:
            logger.warning("散点图数据为空")
            return None

        # 合并所有数据点
        all_points = []
        for dataset in datasets:
            x_values = dataset.get("x", [])
            y_values = dataset.get("y", [])
            points = list(zip(x_values, y_values))
            all_points.extend(points)

        if not all_points:
            return None

        drawing = Drawing(figure.width, figure.height)
        lp = LinePlot()
        lp.x = 40
        lp.y = 40
        lp.width = figure.width - 80
        lp.height = figure.height - 80

        # 数据
        lp.data = [all_points]
        lp.joinedLines = 0  # 不连线，仅显示点

        # 使用标记
        lp.lines[0].symbol = makeMarker("FilledCircle")
        lp.lines[0].strokeColor = colors.HexColor("#ff7f0e")

        # 坐标轴
        all_x = [p[0] for p in all_points]
        all_y = [p[1] for p in all_points]
        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = max(all_x) * 1.1 if all_x else 1
        lp.xValueAxis.labels.fontName = self.chinese_font  # 支持中文
        lp.yValueAxis.valueMin = 0
        lp.yValueAxis.valueMax = max(all_y) * 1.1 if all_y else 100
        lp.yValueAxis.labels.fontName = self.chinese_font  # 支持中文

        drawing.add(lp)
        return drawing

    def _create_radar_chart(self, figure: FigureArtifact) -> Optional[Drawing]:
        """创建雷达图"""
        categories = figure.data.get("categories", [])
        series_dict = figure.data.get("series", {})

        if not categories or not series_dict:
            logger.warning("雷达图数据为空")
            return None

        # 准备数据
        series_data = []
        for optimizer, values in series_dict.items():
            series_data.append(values)

        if not series_data:
            return None

        drawing = Drawing(figure.width, figure.height)
        sc = SpiderChart()
        sc.x = 50
        sc.y = 50
        sc.width = figure.width - 100
        sc.height = figure.height - 100

        # 数据
        sc.data = series_data
        sc.labels = categories

        # 标签字体设置（支持中文）
        sc.labels.fontName = self.chinese_font
        sc.labels.fontSize = 10

        # 样式
        sc.strands.strokeWidth = 1.5
        color_palette = [colors.blue, colors.red, colors.green, colors.orange]
        for i in range(min(len(series_data), len(color_palette))):
            sc.strands[i].fillColor = color_palette[i]
            sc.strands[i].strokeColor = color_palette[i]

        drawing.add(sc)
        return drawing

    def _build_table(self, table: TableArtifact) -> List:
        """构建表格"""
        elements = []

        # 表格标题
        elements.append(Paragraph(table.title, self.styles["ChineseSubsectionTitle"]))

        # 表格描述
        if table.description:
            elements.append(Paragraph(table.description, self.styles["ChineseBody"]))
            elements.append(Spacer(1, 5))

        # 构建表格数据
        data = [table.headers] + table.rows

        # 创建表格
        t = Table(data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    # 表头样式
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONT", (0, 0), (-1, 0), self.chinese_font, 10),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    # 数据行样式
                    ("FONT", (0, 1), (-1, -1), self.chinese_font, 9),
                    ("ALIGN", (0, 1), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # 边框
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
                    # 内边距
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        elements.append(t)
        return elements


__all__ = ["ReportLabExporter"]

