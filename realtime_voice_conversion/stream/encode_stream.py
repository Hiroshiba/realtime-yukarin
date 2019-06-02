import numpy
from yukarin.wave import Wave

from ..segment.feature_wrapper_segment import FeatureWrapperSegmentMethod
from ..segment.wave_segment import WaveSegmentMethod
from ..stream.base_stream import BaseStream
from ..yukarin_wrapper.vocoder import Vocoder
from ..yukarin_wrapper.voice_changer import AcousticFeatureWrapper


class EncodeStream(BaseStream[numpy.ndarray, AcousticFeatureWrapper]):
    def __init__(
            self,
            vocoder: Vocoder,
    ):
        super().__init__(
            in_segment_method=WaveSegmentMethod(
                sampling_rate=vocoder.acoustic_param.sampling_rate,
            ),
            out_segment_method=FeatureWrapperSegmentMethod(
                sampling_rate=1000 // vocoder.acoustic_param.frame_period,
                wave_sampling_rate=vocoder.acoustic_param.sampling_rate,
                order=vocoder.acoustic_param.order,
                frame_period=vocoder.acoustic_param.frame_period,
            ),
        )
        self.vocoder = vocoder

    def process(self, start_time: float, time_length: float, extra_time: float) -> AcousticFeatureWrapper:
        wave = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
        )
        wave = Wave(wave=wave, sampling_rate=self.in_segment_method.sampling_rate)
        feature_wrapper = self.vocoder.encode(wave)

        pad = round(extra_time * self.out_segment_method.sampling_rate)
        if pad > 0:
            feature_wrapper = self.out_segment_method.pick(feature_wrapper, pad, -pad)

        return feature_wrapper
