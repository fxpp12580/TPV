# utils/logger.py
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """
    设置日志记录器，支持同时输出到控制台和文件

    Args:
        name: 日志记录器名称

    Returns:
        配置好的日志记录器实例
    """
    # 创建日志记录器
    logger = logging.getLogger(name)

    # 如果日志记录器已经有处理器，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 创建日志文件名（按日期）
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{name}_{today}.log"

    # 创建文件处理器（支持日志轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 示例用法
if __name__ == "__main__":
    # 创建测试日志记录器
    test_logger = setup_logger("test")

    # 记录不同级别的日志
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
