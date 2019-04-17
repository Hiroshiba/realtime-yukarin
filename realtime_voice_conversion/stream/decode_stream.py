import numpy
from yukarin.acoustic_feature import AcousticFeature

from ..segment.feature_segment import FeatureSegmentMethod
from ..segment.wave_segment import WaveSegmentMethod
from ..stream.base_stream import BaseStream
from ..yukarin_wrapper.vocoder import Vocoder


class DecodeStream(BaseStream[AcousticFeature, numpy.ndarray]):
    def __init__(
            self,
            vocoder: Vocoder,
    ):
        super().__init__(
            in_segment_method=FeatureSegmentMethod(
                sampling_rate=1000 // vocoder.acoustic_param.frame_period,
                wave_sampling_rate=vocoder.out_sampling_rate,
                order=vocoder.acoustic_param.order,
            ),
            out_segment_method=WaveSegmentMethod(
                sampling_rate=vocoder.out_sampling_rate,
            ),
        )
        self.vocoder = vocoder

    def process(self, start_time: float, time_length: float, extra_time: float) -> numpy.ndarray:
        out_feature = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
        )

        wave = self.vocoder.decode(
            acoustic_feature=out_feature,
        ).wave

        wave[numpy.isnan(wave)] = 0
        return wave
