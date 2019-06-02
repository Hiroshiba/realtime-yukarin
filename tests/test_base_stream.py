from typing import Iterable
from unittest import TestCase

from realtime_voice_conversion.segment.segment import BaseSegmentMethod
from realtime_voice_conversion.stream.base_stream import BaseStream


class TestSegmentMethod(BaseSegmentMethod[str]):
    def __init__(self, sampling_rate: int):
        super().__init__(sampling_rate=sampling_rate)

    def length(self, data: str) -> int:
        return len(data)

    def pad(self, width: int) -> str:
        return ' ' * width

    def pick(self, data: str, first: int, last: int) -> str:
        return data[first:last]

    def concat(self, datas: Iterable[str]) -> str:
        return ''.join(datas)


class Stream(BaseStream):
    def process(self, start_time: float, time_length: float, extra_time: float):
        raise NotImplementedError()


class BaseStreamTest(TestCase):
    def setUp(self):
        self.rate = 10
        self.stream = Stream(
            in_segment_method=TestSegmentMethod(sampling_rate=self.rate),
            out_segment_method=TestSegmentMethod(sampling_rate=self.rate),
        )

    def test_initialize(self):
        pass

    def test_add(self):
        self.stream.add(start_time=0, data='a' * self.rate)
        self.stream.add(start_time=1, data='b' * self.rate)
        self.stream.add(start_time=2, data='c' * self.rate)
        self.assertEqual(len(self.stream.stream), 3)

    def test_remove(self):
        self.stream.add(start_time=0, data='a' * self.rate)
        self.stream.add(start_time=1, data='b' * self.rate)
        self.stream.add(start_time=2, data='c' * self.rate)
        self.assertEqual(len(self.stream.stream), 3)

        self.stream.remove(end_time=0)
        self.assertEqual(len(self.stream.stream), 3)

        self.stream.remove(end_time=1)
        self.assertEqual(len(self.stream.stream), 2)

        self.stream.remove(end_time=2)
        self.assertEqual(len(self.stream.stream), 1)

        self.stream.remove(end_time=3)
        self.assertEqual(len(self.stream.stream), 0)

    def test_fetch(self):
        self.stream.add(start_time=0, data='a' * self.rate)
        self.stream.add(start_time=1, data='b' * self.rate)

        data = self.stream.fetch(start_time=0, time_length=1, extra_time=0)
        self.assertEqual(data, 'a' * self.rate)

        data = self.stream.fetch(start_time=0.5, time_length=1, extra_time=0)
        self.assertEqual(data, 'a' * (self.rate // 2) + 'b' * (self.rate // 2))

    def test_fetch_with_padding(self):
        self.stream.add(start_time=0, data='a' * self.rate)
        self.stream.add(start_time=1, data='b' * self.rate)

        data = self.stream.fetch(start_time=-0.5, time_length=1, extra_time=0)
        self.assertEqual(data, ' ' * (self.rate // 2) + 'a' * (self.rate // 2))

        data = self.stream.fetch(start_time=1.5, time_length=1, extra_time=0)
        self.assertEqual(data, 'b' * (self.rate // 2) + ' ' * (self.rate // 2))

    def test_fetch_with_extra(self):
        self.stream.add(start_time=0, data='a' * self.rate)
        self.stream.add(start_time=1, data='b' * self.rate)

        data = self.stream.fetch(start_time=0, time_length=1, extra_time=0.3)
        self.assertEqual(data, ' ' * 3 + 'a' * self.rate + 'b' * 3)

        data = self.stream.fetch(start_time=0, time_length=2, extra_time=0.3)
        self.assertEqual(data, ' ' * 3 + 'a' * self.rate + 'b' * self.rate + ' ' * 3)
