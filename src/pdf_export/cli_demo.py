"""CLI 演示脚本 - 测试 PDF 生成"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_export.collectors import DashboardDataCollector
from src.pdf_export.reportlab_renderer import ReportLabExporter
from src.pdf_export.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("PDF 导出演示")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("步骤 1/3: 加载实验结果...")
    collector = DashboardDataCollector(results_dir=PROJECT_ROOT / "results")
    all_results = collector.load_all_results()

    if not all_results:
        logger.error("未找到任何结果文件，请先运行实验")
        print("\n错误: 未找到结果文件")
        print("请先运行以下命令生成数据:")
        print("  python experiments/benchmark_suite.py --all")
        return 1

    logger.info("找到 %d 个结果文件", len(all_results))

    # 2. 构建报告上下文
    logger.info("步骤 2/3: 构建报告上下文...")
    first_result = all_results[0]
    context = collector.build_report_context(first_result, all_results)

    logger.info("报告标题: %s", context.title)
    logger.info("章节数量: %d", len(context.sections))
    for section in context.sections:
        logger.info(
            "  - %s (%d 图表, %d 表格)",
            section.title,
            len(section.figures),
            len(section.tables),
        )

    # 3. 生成 PDF
    logger.info("步骤 3/3: 生成 PDF...")
    output_path = PROJECT_ROOT / "reports" / "example_report.pdf"
    exporter = ReportLabExporter(output_path)

    try:
        exporter.generate(context)
        logger.info("=" * 60)
        logger.info("成功! PDF 已生成: %s", output_path)
        logger.info("=" * 60)
        print(f"\n[OK] PDF 生成成功!")
        print(f"文件位置: {output_path}")
        print(f"包含 {len(context.sections)} 个章节")
        return 0
    except Exception as e:
        logger.error("PDF 生成失败: %s", e, exc_info=True)
        print(f"\n[ERROR] PDF 生成失败: {e}")
        print("详细日志请查看: logs/pdf_export.log")
        return 1


if __name__ == "__main__":
    # 确保日志目录存在
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    sys.exit(main())

