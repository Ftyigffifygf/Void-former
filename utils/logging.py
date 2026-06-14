import logging
import sys

_FMT = "[%(asctime)s] %(name)s %(levelname)s | %(message)s"


def get_logger(name: str = "voidformer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger
