from typing import Iterable
from unittest import TestCase

from realtime_voice_conversion.segment.segment import BaseSegmentMethod, Segment


class DummySegmentMethod(BaseSegmentMethod[str]):
    def __init__(self, sampling_rate: int):
        super().__init__(sampling_rate=sampling_rate)

    def length(self, data: str) -> int:
        raise NotImplementedError()

    def pad(self, width: int) -> str:
        raise NotImplementedError()

    def pick(self, data: str, first: int, last: int) -> str:
        raise NotImplementedError()

    def concat(self, datas: Iterable[str]) -> str:
        raise NotImplementedError()


class SegmentTest(TestCase):
    def setUp(self):
        self.start_time = 1
        self.data = ''
        self.method = DummySegmentMethod(sampling_rate=1)
        self.segment = Segment(
            start_time=self.start_time,
            data=self.data,
            method=self.method,
        )

    def test(self):
        self.assertEqual(self.start_time, self.segment.start_time)
        self.assertEqual(self.data, self.segment.data)
        self.assertEqual(self.method, self.segment.method)
