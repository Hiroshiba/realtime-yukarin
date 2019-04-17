from typing import Iterable
from unittest import TestCase

import numpy
from become_yukarin.param import Param
from yukarin import Wave
from yukarin.param import AcousticParam

from realtime_voice_conversion.stream import ConvertStream
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger, AcousticFeatureWrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ConvertStreamTest(TestCase):
    def setUp(self):
        self.voice_changer: VoiceChanger = AttrDict(
            acoustic_converter=AttrDict(
                config=AttrDict(
                    dataset=AttrDict(
                        acoustic_param=AcousticParam(sampling_rate=16000),
                    ),
                ),
            ),
            super_resolution=AttrDict(
                config=AttrDict(
                    dataset=AttrDict(
                        param=Param(),
                    ),
                ),
            ),
            output_sampling_rate=24000,
        )
        self.stream = ConvertStream(voice_changer=self.voice_changer)
        self.stream.in_segment_method._keys = ['f0']

    @property
    def in_sampling_rate(self):
        return self.voice_changer.acoustic_converter.config.dataset.acoustic_param.sampling_rate

    @property
    def in_feature_rate(self):
        return 1000 // 5

    def get_feature_wrapper_segments(self, values: Iterable[float], time_lengths: Iterable[float]):
        return AcousticFeatureWrapper(
            wave=Wave(
                wave=numpy.concatenate([
                    numpy.ones(round(time_length * self.in_sampling_rate), dtype=numpy.float32) * value
                    for value, time_length in zip(values, time_lengths)
                ]),
                sampling_rate=self.in_sampling_rate,
            ),
            f0=numpy.concatenate([
                numpy.ones((round(time_length * self.in_feature_rate), 1), dtype=numpy.float32) * value
                for value, time_length in zip(values, time_lengths)
            ]),
        )

    def get_feature_wrapper_segment(self, value: float, time_length: float = 1):
        return self.get_feature_wrapper_segments([value], [time_length])

    def test_initialize(self):
        pass

    def test_fetch(self):
        self.stream.add(start_time=0, data=self.get_feature_wrapper_segment(1))
        self.stream.add(start_time=1, data=self.get_feature_wrapper_segment(2))

        data = self.stream.fetch(start_time=0, time_length=1, extra_time=0)
        target = self.get_feature_wrapper_segment(1)
        self.assertEqual(data, target)

        data = self.stream.fetch(start_time=0.5, time_length=1, extra_time=0)
        target = self.get_feature_wrapper_segments([1, 2], [0.5, 0.5])
        self.assertEqual(data, target)

    def test_fetch_with_padding(self):
        self.stream.add(start_time=0, data=self.get_feature_wrapper_segment(1))
        self.stream.add(start_time=1, data=self.get_feature_wrapper_segment(2))

        data = self.stream.fetch(start_time=-0.5, time_length=1, extra_time=0)
        target = self.get_feature_wrapper_segments([0, 1], [0.5, 0.5])
        self.assertEqual(data, target)

        data = self.stream.fetch(start_time=1.5, time_length=1, extra_time=0)
        target = self.get_feature_wrapper_segments([2, 0], [0.5, 0.5])
        self.assertEqual(data, target)

    def test_fetch_with_extra(self):
        self.stream.add(start_time=0, data=self.get_feature_wrapper_segment(1))
        self.stream.add(start_time=1, data=self.get_feature_wrapper_segment(2))

        data = self.stream.fetch(start_time=0, time_length=1, extra_time=0.3)
        target = self.get_feature_wrapper_segments([0, 1, 2], [0.3, 1, 0.3])
        self.assertEqual(data, target)

        data = self.stream.fetch(start_time=0, time_length=2, extra_time=0.3)
        target = self.get_feature_wrapper_segments([0, 1, 2, 0], [0.3, 1, 1, 0.3])
        self.assertEqual(data, target)
