from typing import Iterable

import numpy

from ..segment.segment import BaseSegmentMethod


class WaveSegmentMethod(BaseSegmentMethod[numpy.ndarray]):
    def length(self, data: numpy.ndarray) -> int:
        return len(data)

    def pad(self, width: int):
        return numpy.zeros(shape=width, dtype=numpy.float32)

    def pick(self, data: numpy.ndarray, first: int, last: int):
        return data[first:last]

    def concat(self, datas: Iterable[numpy.ndarray]):
        return numpy.concatenate(datas)
