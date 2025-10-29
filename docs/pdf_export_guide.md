# PDF导出功能使用指南

## 概述

PDF导出功能现已基于 **ReportLab** 重构完成，支持导出所有标签页内容（损失曲线、能耗对比、Pareto前沿、综合评估、硬件仿真），提供专业的PDF报告生成。

## 功能特性

- **全面覆盖**：导出所有5个标签页的图表、表格和说明文字
- **原生渲染**：使用 ReportLab 原生图表库，无需外部图像转换
- **中文支持**：完整支持中文文本显示（STSong-Light字体）
- **专业排版**：A4页面、自动分页、样式化章节标题
- **详细日志**：完整的生成日志记录到 `logs/pdf_export.log`
- **错误提示**：清晰的错误信息和故障排查指导

## 使用方法

### 在 Dashboard 中导出

1. 启动可视化 Dashboard：
   ```bash
   streamlit run src/visualization/advanced_dashboard.py
   # 或使用快捷脚本
   python run_dashboard.py
   ```

2. 在侧边栏选择要导出的实验结果文件

3. 点击侧边栏底部的 **"生成PDF报告"** 按钮

4. 等待生成完成（通常需要 5-10 秒）

5. 点击 **"下载PDF报告"** 按钮保存文件

### 命令行生成

直接通过命令行生成示例报告：

```bash
python src/pdf_export/cli_demo.py
```

生成的PDF将保存到 `reports/example_report.pdf`。

## 依赖安装

PDF导出功能需要以下Python包：

```bash
pip install reportlab>=4.0 pyyaml>=6.0
```

或安装完整依赖：

```bash
pip install -r requirements.txt
```

## 编程接口

### 基本用法

```python
from pathlib import Path
from src.pdf_export import DashboardDataCollector, ReportLabExporter

# 1. 加载数据
collector = DashboardDataCollector(results_dir=Path("results"))
all_results = collector.load_all_results()

# 2. 构建报告上下文
first_result = all_results[0]
context = collector.build_report_context(first_result, all_results)

# 3. 生成PDF
output_path = Path("reports/my_report.pdf")
exporter = ReportLabExporter(output_path)
exporter.generate(context)
```

### 自定义配置

```python
from reportlab.lib.pagesizes import LETTER

# 使用自定义页面大小和边距
exporter = ReportLabExporter(
    output_path=Path("reports/custom.pdf"),
    page_size=LETTER,
    margin_mm=25
)
exporter.generate(context)
```

## 故障排除

### 问题：PDF导出按钮不显示

**解决方案**：
1. 确认已安装依赖：`pip install reportlab pyyaml`
2. 重启 Streamlit Dashboard

### 问题：图表显示为空

**原因**：部分实验结果文件可能缺少某些字段（如 `loss_history`、`energy_consumed_j`）

**解决方案**：
- 确保实验结果文件包含完整数据
- 检查 `logs/pdf_export.log` 查看详细错误信息

### 问题：中文显示为乱码

**原因**：ReportLab 默认字体不支持中文

**解决方案**：
- ✅ 当前版本已支持中文字体（使用 ReportLab 内置的 STSong-Light）
- 所有文本、表格均可正常显示中文
- 如遇到问题，请检查 `logs/pdf_export.log` 中的字体加载日志

## 技术细节

### 架构设计

PDF导出采用模块化管线：

1. **数据收集** (`collectors.py`)：从 Dashboard 结果文件提取5个标签页的原始数据
2. **数据建模** (`models.py`)：使用 `dataclass` 定义 `ReportContext`、`Section`、`FigureArtifact`、`TableArtifact`
3. **图表渲染** (`reportlab_renderer.py`)：使用 ReportLab 原生图表库渲染线图、柱状图、散点图、雷达图
4. **PDF合成** (`reportlab_renderer.py`)：组装封面、章节、图表、表格，生成完整PDF
5. **日志记录** (`logging_utils.py`)：记录每个阶段的执行情况到 `logs/pdf_export.log`

### 支持的图表类型

- **线图** (`LinePlot`)：训练损失曲线
- **柱状图** (`VerticalBarChart`)：能耗与NFE对比
- **散点图** (`LinePlot` 无连线)：Pareto前沿
- **雷达图** (`SpiderChart`)：综合效率评估

### 数据格式

图表数据采用字典格式存储：

```python
# 线图
{"series": {"optimizer1": [0.5, 0.3, 0.1], "optimizer2": [0.6, 0.4, 0.2]}}

# 柱状图
{"optimizers": ["Adam", "RK4"], "energies": [0.1, 0.05], "nfes": [100, 200]}

# 散点图
{"datasets": [{"name": "exp1", "x": [0.1, 0.2], "y": [80, 85], "labels": ["opt1", "opt2"]}]}

# 雷达图
{"categories": ["准确率", "能效", "速度"], "series": {"Adam": [0.8, 0.6, 0.7]}}
```

## 示例报告

运行 `python src/pdf_export/cli_demo.py` 后，可在 `reports/example_report.pdf` 查看示例报告结构：

1. **封面**：标题、生成时间、元数据、章节目录
2. **第1章 - 损失曲线**：训练损失随时间变化的线图 + 优化器结果表格
3. **第2章 - 能耗对比**：能耗与NFE柱状图 + 能耗指标表格
4. **第3章 - Pareto前沿**：准确率vs能耗散点图
5. **第4章 - 综合评估**：效率雷达图 + 效率排名表格
6. **第5章 - 硬件仿真**：硬件参数影响预测表格

## 常见问题

**Q: 为什么不使用 Plotly + Kaleido 生成图像？**

A: Kaleido 在某些环境下存在兼容性和性能问题（如生成速度慢、依赖复杂）。ReportLab 原生图表库更轻量、稳定，且无需外部依赖。

**Q: 可以自定义PDF样式吗？**

A: 当前版本使用预定义样式。如需自定义，可修改 `src/pdf_export/reportlab_renderer.py` 中的 `_setup_styles()` 方法。

**Q: 支持导出单个标签页吗？**

A: 当前版本导出所有标签页。未来版本将支持选择性导出。

## 未来改进

- [ ] 支持选择单个标签页导出
- [ ] 添加图表标注和解释
- [ ] 支持自定义模板
- [ ] 支持导出为Word格式
- [ ] 批量导出多个实验结果
- [ ] 添加实验对比分析章节

## 相关文档

- [用户指南](user_guide.md)
- [Dashboard使用说明](../README.md#可视化dashboard)
- [WeasyPrint文档](https://doc.courtbouillon.org/weasyprint/)
- [Kaleido文档](https://github.com/plotly/Kaleido)

