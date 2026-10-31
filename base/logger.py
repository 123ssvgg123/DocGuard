"""
    日志管理模块：base/logger.py
"""
import logging
# 导入路径操作库
import os
# 导入配置类
from base.config import config


def setup_logger(logger_name='FoodSafetyRAG', logger_file=config.LOG_FILE):
    # 1. 确保日志目录存在，不存在创建
    dirname = os.path.dirname(logger_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # 2. 创建日志记录器: Logger
    logger = logging.getLogger(logger_name)
    # 设置全局日志级别为 DEBUG
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 3. 创建控制台控制器：StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        # 4. 创建文件处理器: FileHandler
        file_handler = logging.FileHandler(logger_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        # 5. 定义并设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s')
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # 6. 把处理器添加到 logger 中
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


# 创建日志记录器
logger = setup_logger('FoodSafetyRAG')


if __name__ == '__main__':
    logger.info("hello world 你好，世界")
