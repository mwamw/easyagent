def enable_logging(level="INFO"):
    """便捷方法：快速启用 EasyAgent 日志输出"""
    import logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s | %(levelname)s | %(message)s'
    )