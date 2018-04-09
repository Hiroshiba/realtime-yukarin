from abc import ABCMeta, abstractmethod
from typing import NamedTuple, List, Callable, Any

import numpy
from yukarin.acoustic_feature import AcousticFeature
from yukarin.wave import Wave

from .yukarin_wrapper.vocoder import Vocoder
from .yukarin_wrapper.voice_changer import AcousticFeatureWrapper
from .yukarin_wrapper.voice_changer import VoiceChanger


class BaseSegment(ABCMeta):
    start_time: float

    @property
    @abstractmethod
    def time_length(self) -> float:
        pass

    @property
    @abstractmethod
    def end_time(self) -> float:
        pass


class FeatureSegment(NamedTuple, BaseSegment):
    start_time: float
    feature: AcousticFeature
    frame_period: float

    @property
    def time_length(self):
        return len(self.feature.f0) * self.frame_period / 1000

    @property
    def end_time(self):
        return self.time_length + self.start_time


class FeatureWrapperSegment(NamedTuple, BaseSegment):
    start_time: float
    feature: AcousticFeatureWrapper
    frame_period: float

    @property
    def time_length(self):
        return len(self.feature.f0) * self.frame_period / 1000

    @property
    def end_time(self):
        return self.time_length + self.start_time


class WaveSegment(NamedTuple, BaseSegment):
    start_time: float
    wave: Wave

    @property
    def time_length(self):
        return len(self.wave.wave) / self.wave.sampling_rate

    @property
    def end_time(self):
        return self.time_length + self.start_time


class VoiceChangerStream(object):
    def __init__(
            self,
            sampling_rate: int,
            frame_period: float,
            order: int,
            in_dtype=numpy.float32,
    ):
        self.sampling_rate = sampling_rate
        self.frame_period = frame_period
        self.order = order
        self.in_dtype = in_dtype

        self.voice_changer: VoiceChanger = None
        self.vocoder: Vocoder = None
        self._data_stream: List[WaveSegment] = []
        self._in_feature_stream: List[FeatureWrapperSegment] = []
        self._out_feature_stream: List[FeatureSegment] = []

    def add_wave(self, start_time: float, wave: Wave):
        # validation
        assert wave.sampling_rate == self.sampling_rate
        assert wave.wave.dtype == self.in_dtype

        segment = WaveSegment(start_time=start_time, wave=wave)
        self._data_stream.append(segment)

    def add_in_feature(self, start_time: float, feature_wrapper: AcousticFeatureWrapper, frame_period: float):
        # validation
        assert frame_period == self.frame_period
        assert feature_wrapper.f0.dtype == self.in_dtype

        segment = FeatureWrapperSegment(start_time=start_time, feature=feature_wrapper, frame_period=self.frame_period)
        self._in_feature_stream.append(segment)

    def add_out_feature(self, start_time: float, feature: AcousticFeature, frame_period: float):
        # validation
        assert frame_period == self.frame_period

        segment = FeatureSegment(start_time=start_time, feature=feature, frame_period=self.frame_period)
        self._out_feature_stream.append(segment)

    def remove(self, end_time: float):
        self._data_stream = list(filter(lambda s: s.end_time > end_time, self._data_stream))
        self._in_feature_stream = list(filter(lambda s: s.end_time > end_time, self._in_feature_stream))
        self._out_feature_stream = list(filter(lambda s: s.end_time > end_time, self._out_feature_stream))

    @staticmethod
    def fetch(
            start_time: float,
            time_length: float,
            data_stream: List[BaseSegment],
            rate: float,
            pad_function: Callable[[int], Any],
            pick_function: Callable[[Any, int, int], Any],
            concat_function: Callable[[List], Any],
            extra_time: float = 0,
    ):
        start_time -= extra_time
        time_length += extra_time * 2

        end_time = start_time + time_length
        buffer_list = []
        stream = filter(lambda s: not (end_time < s.start_time or s.end_time < start_time), data_stream)

        start_time_buffer = start_time
        remaining_time = time_length
        for segment in stream:
            # padding
            if segment.start_time > start_time_buffer:
                length = round((segment.start_time - start_time_buffer) * rate)
                pad = pad_function(length)
                buffer_list.append(pad)
                remaining_time -= segment.start_time - start_time_buffer
                start_time_buffer = segment.start_time

            if remaining_time > segment.end_time - start_time_buffer:
                one_time_length = segment.end_time - start_time_buffer
            else:
                one_time_length = remaining_time

            first_index = round((start_time_buffer - segment.start_time) * rate)
            last_index = round(first_index + one_time_length * rate)
            one_buffer = pick_function(segment, first_index, last_index)
            buffer_list.append(one_buffer)

            start_time_buffer += one_time_length
            remaining_time -= one_time_length

            if start_time_buffer >= end_time:
                break
        else:
            # last padding
            length = round((end_time - start_time_buffer) * rate)
            pad = pad_function(length)
            buffer_list.append(pad)

        buffer = concat_function(buffer_list)
        return buffer

    def pre_convert(self, start_time: float, time_length: float, extra_time: float):
        keys = ['f0', 'ap', 'mc', 'voiced']
        wave = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
            data_stream=self._data_stream,
            rate=self.sampling_rate,
            pad_function=lambda length: numpy.zeros(shape=length, dtype=self.in_dtype),
            pick_function=lambda segment, first, last: segment.wave.wave[first:last],
            concat_function=numpy.concatenate,
        )
        in_wave = Wave(wave=wave, sampling_rate=self.sampling_rate)
        in_feature = self.vocoder.encode(in_wave)

        pad = round(extra_time * self.sampling_rate)
        in_wave.wave = in_wave.wave[pad:-pad]

        pad = round(extra_time / (self.vocoder.acoustic_param.frame_period / 1000))
        in_feature = in_feature.pick(pad, -pad, keys=keys)

        feature_wrapper = AcousticFeatureWrapper(wave=in_wave, **in_feature.__dict__)
        return feature_wrapper

    def convert(self, start_time: float, time_length: float, extra_time: float):
        sizes = AcousticFeature.get_sizes(sampling_rate=self.sampling_rate, order=self.order)
        keys = ['f0', 'ap', 'mc', 'voiced']

        def _pad_function(length):
            return AcousticFeatureWrapper.silent_wrapper(
                length,
                sizes=sizes,
                keys=keys,
                frame_period=self.frame_period,
                sampling_rate=self.sampling_rate,
                wave_dtype=self.in_dtype,
            ).astype_only_float_wrapper(self.in_dtype)

        def _pick_function(segment: FeatureWrapperSegment, first, last):
            return segment.feature.pick_wrapper(
                first,
                last,
                keys=keys,
                frame_period=self.frame_period,
            )

        in_feature = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
            data_stream=self._in_feature_stream,
            rate=1000 / self.frame_period,
            pad_function=_pad_function,
            pick_function=_pick_function,
            concat_function=lambda buffers: AcousticFeatureWrapper.concatenate_wrapper(buffers, keys=keys),
        )
        out_feature = self.voice_changer.convert_from_acoustic_feature(in_feature)

        pad = round(extra_time * 1000 / self.frame_period)
        out_feature = out_feature.pick(pad, -pad, keys=['f0', 'ap', 'sp', 'voiced'])
        return out_feature

    def post_convert(self, start_time: float, time_length: float):
        sizes = AcousticFeature.get_sizes(sampling_rate=self.sampling_rate, order=self.order)
        keys = ['f0', 'ap', 'sp', 'voiced']
        out_feature = self.fetch(
            start_time=start_time,
            time_length=time_length,
            data_stream=self._out_feature_stream,
            rate=1000 / self.frame_period,
            pad_function=lambda length: AcousticFeature.silent(length, sizes=sizes, keys=keys),
            pick_function=lambda segment, first, last: segment.feature.pick(first, last, keys=keys),
            concat_function=lambda buffers: AcousticFeature.concatenate(buffers, keys=keys),
        )

        out_wave = self.vocoder.decode(
            acoustic_feature=out_feature,
        )

        w = out_wave.wave
        w[numpy.isnan(w)] = 0
        out_wave = Wave(wave=w, sampling_rate=out_wave.sampling_rate)
        return out_wave


class VoiceChangerStreamWrapper(object):
    def __init__(
            self,
            voice_changer_stream: VoiceChangerStream,
            extra_time_pre: float = 0.0,
            extra_time: float = 0.0,
    ):
        self.voice_changer_stream = voice_changer_stream
        self.extra_time_pre = extra_time_pre
        self.extra_time = extra_time
        self._current_time_pre = 0
        self._current_time = 0
        self._current_time_post = 0

    def pre_convert_next(self, time_length: float):
        in_feature = self.voice_changer_stream.pre_convert(
            start_time=self._current_time_pre,
            time_length=time_length,
            extra_time=self.extra_time_pre,
        )
        self._current_time_pre += time_length
        return in_feature

    def convert_next(self, time_length: float):
        out_feature = self.voice_changer_stream.convert(
            start_time=self._current_time,
            time_length=time_length,
            extra_time=self.extra_time,
        )
        self._current_time += time_length
        return out_feature

    def post_convert_next(self, time_length: float):
        out_wave = self.voice_changer_stream.post_convert(
            start_time=self._current_time_post,
            time_length=time_length,
        )
        self._current_time_post += time_length
        return out_wave

    def remove_previous(self):
        end_time = min(
            self._current_time_pre - self.extra_time_pre,
            self._current_time - self.extra_time,
            self._current_time_post,
        )
        self.voice_changer_stream.remove(end_time=end_time)
