import logging
import os
from typing import Any


class Item(object):
    def __init__(
            self,
            item: Any,
            index: int,
    ):
        self.item = item
        self.index = index


def init_logger(logger=None, filename='log.txt'):
    if logger is None:
        logger = logging.getLogger()

    format = logging.Formatter('%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s')
    logger.setLevel(os.getenv('LOG_LEVEL', 'WARNING'))

    handler = logging.FileHandler(filename)
    handler.setFormatter(format)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(format)
    logger.addHandler(handler)
