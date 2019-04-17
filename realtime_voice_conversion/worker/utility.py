import logging
import os
from typing import NamedTuple

import numpy


class AudioConfig(NamedTuple):
    in_rate: int
    out_rate: int
    frame_period: float
    in_audio_chunk: int
    out_audio_chunk: int
    vocoder_buffer_size: int
    in_norm: float
    out_norm: float
    input_silent_threshold: float
    silent_threshold: float


class Item(object):
    def __init__(
            self,
            original: numpy.ndarray,
            item: any,
            index: int,
            conversion_flag: bool,
    ):
        self.original = original
        self.item = item
        self.index = index
        self.conversion_flag = conversion_flag


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
