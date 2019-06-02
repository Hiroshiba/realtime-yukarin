from typing import Iterable, NamedTuple
from unittest import TestCase

import numpy
from realtime_voice_conversion.yukarin_wrapper.vocoder import Vocoder
from yukarin.param import AcousticParam

from realtime_voice_conversion.segment.segment import BaseSegmentMethod
from realtime_voice_conversion.stream import EncodeStream
from realtime_voice_conversion.stream.base_stream import BaseStream


class VocoderMock(NamedTuple):
    acoustic_param: AcousticParam = AcousticParam()


class EncodeStreamTest(TestCase):
    def setUp(self):
        self.vocoder: Vocoder = VocoderMock()
        self.stream = EncodeStream(vocoder=self.vocoder)

    @property
    def in_sampling_rate(self):
        return self.vocoder.acoustic_param.sampling_rate

    def get_numpy_array(self, value: float, length=None):
        if length is None:
            length = self.in_sampling_rate
        return numpy.ones(length, dtype=numpy.float32) * value

    def test_initialize(self):
        pass

    def test_fetch(self):
        self.stream.add(start_time=0, data=self.get_numpy_array(1))
        self.stream.add(start_time=1, data=self.get_numpy_array(2))

        data = self.stream.fetch(start_time=0, time_length=1, extra_time=0)
        numpy.testing.assert_equal(data, self.get_numpy_array(1))

        data = self.stream.fetch(start_time=0.5, time_length=1, extra_time=0)
        target = numpy.concatenate([
            self.get_numpy_array(1, self.in_sampling_rate // 2),
            self.get_numpy_array(2, self.in_sampling_rate // 2),
        ])
        numpy.testing.assert_equal(data, target)

    def test_fetch_with_padding(self):
        self.stream.add(start_time=0, data=self.get_numpy_array(1))
        self.stream.add(start_time=1, data=self.get_numpy_array(2))

        data = self.stream.fetch(start_time=-0.5, time_length=1, extra_time=0)
        target = numpy.concatenate([
            self.get_numpy_array(0, self.in_sampling_rate // 2),
            self.get_numpy_array(1, self.in_sampling_rate // 2),
        ])
        numpy.testing.assert_equal(data, target)

        data = self.stream.fetch(start_time=1.5, time_length=1, extra_time=0)
        target = numpy.concatenate([
            self.get_numpy_array(2, self.in_sampling_rate // 2),
            self.get_numpy_array(0, self.in_sampling_rate // 2),
        ])
        numpy.testing.assert_equal(data, target)

    def test_fetch_with_extra(self):
        self.stream.add(start_time=0, data=self.get_numpy_array(1))
        self.stream.add(start_time=1, data=self.get_numpy_array(2))

        data = self.stream.fetch(start_time=0, time_length=1, extra_time=0.3)
        target = numpy.concatenate([
            self.get_numpy_array(0, self.in_sampling_rate // 10 * 3),
            self.get_numpy_array(1, self.in_sampling_rate),
            self.get_numpy_array(2, self.in_sampling_rate // 10 * 3),
        ])
        numpy.testing.assert_equal(data, target)

        data = self.stream.fetch(start_time=0, time_length=2, extra_time=0.3)
        target = numpy.concatenate([
            self.get_numpy_array(0, self.in_sampling_rate // 10 * 3),
            self.get_numpy_array(1, self.in_sampling_rate),
            self.get_numpy_array(2, self.in_sampling_rate),
            self.get_numpy_array(0, self.in_sampling_rate // 10 * 3),
        ])
        numpy.testing.assert_equal(data, target)
