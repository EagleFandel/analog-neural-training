"""PDF 导出日志工具"""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

LOG_CONFIG_PATH = Path("config/logging/pdf_logging.yaml")
DEFAULT_LOGGER_NAME = "pdf_export"


def setup_logging(config_path: Optional[Path] = None) -> None:
    """初始化日志配置"""
    config_file = config_path or LOG_CONFIG_PATH
    if config_file.exists():
        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logging.getLogger(__name__).warning(
            "日志配置文件 %s 不存在，使用 basicConfig", config_file
        )


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    if not logging.getLogger(name).handlers:
        setup_logging()
    return logging.getLogger(name)


