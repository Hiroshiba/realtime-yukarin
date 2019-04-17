from typing import Iterable
from unittest import TestCase

import numpy
from yukarin import Wave

from realtime_voice_conversion.segment.feature_wrapper_segment import FeatureWrapperSegmentMethod
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FeatureWrapperSegmentMethodTest(TestCase):
    def setUp(self):
        self.frame_period = 10
        self.sampling_rate = 1000 // self.frame_period
        self.wave_sampling_rate = 10000
        self.order = 5
        self.segment_method = FeatureWrapperSegmentMethod(
            sampling_rate=self.sampling_rate,
            wave_sampling_rate=self.wave_sampling_rate,
            order=self.order,
            frame_period=self.frame_period,
        )

    def get_segments(self, values: Iterable[float], time_lengths: Iterable[float]):
        return AcousticFeatureWrapper(
            wave=Wave(
                wave=numpy.concatenate([
                    numpy.ones(round(time_length * self.wave_sampling_rate), dtype=numpy.float32) * value
                    for value, time_length in zip(values, time_lengths)
                ]),
                sampling_rate=self.wave_sampling_rate,
            ),
            f0=numpy.concatenate([
                numpy.ones((round(time_length * self.sampling_rate), 1), dtype=numpy.float32) * value
                for value, time_length in zip(values, time_lengths)
            ]),
        )

    def get_segment(self, value: float, time_length: float = 1):
        return self.get_segments([value], [time_length])

    def test_pad(self):
        data = self.segment_method.pad(width=self.sampling_rate)
        target = self.get_segment(0, time_length=1)
        self.assertEqual(data, target)

    def test_pick(self):
        data = self.get_segment(value=1, time_length=1)
        data = self.segment_method.pick(data, first=0, last=self.sampling_rate // 2)
        target = self.get_segment(value=1, time_length=0.5)
        self.assertEqual(data, target)

        data = self.get_segment(value=1, time_length=1)
        data = self.segment_method.pick(data, first=self.sampling_rate // 2, last=self.sampling_rate)
        target = self.get_segment(value=1, time_length=0.5)
        self.assertEqual(data, target)

    def test_concat(self):
        data1 = self.get_segment(value=0, time_length=1)
        data2 = self.get_segment(value=1, time_length=1)

        data = self.segment_method.concat([data1, data2])
        target = self.get_segments(values=[0, 1], time_lengths=[1, 1])
        self.assertEqual(data, target)
