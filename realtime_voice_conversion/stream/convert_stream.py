from yukarin.acoustic_feature import AcousticFeature

from ..segment.feature_segment import FeatureSegmentMethod
from ..segment.feature_wrapper_segment import FeatureWrapperSegmentMethod
from ..stream.base_stream import BaseStream
from ..yukarin_wrapper.voice_changer import AcousticFeatureWrapper
from ..yukarin_wrapper.voice_changer import VoiceChanger


class ConvertStream(BaseStream[AcousticFeatureWrapper, AcousticFeature]):
    def __init__(
            self,
            voice_changer: VoiceChanger,
    ):
        acoustic_converter_acoustic_param = voice_changer.acoustic_converter.config.dataset.acoustic_param
        super_resolution_acoustic_param = voice_changer.super_resolution.config.dataset.param.acoustic_feature_param
        super().__init__(
            in_segment_method=FeatureWrapperSegmentMethod(
                sampling_rate=1000 // acoustic_converter_acoustic_param.frame_period,
                wave_sampling_rate=acoustic_converter_acoustic_param.sampling_rate,
                order=acoustic_converter_acoustic_param.order,
                frame_period=acoustic_converter_acoustic_param.frame_period,
            ),
            out_segment_method=FeatureSegmentMethod(
                sampling_rate=1000 // super_resolution_acoustic_param.frame_period,
                wave_sampling_rate=voice_changer.output_sampling_rate,
                order=super_resolution_acoustic_param.order,
            ),
        )
        self.voice_changer = voice_changer

    def process(self, start_time: float, time_length: float, extra_time: float) -> AcousticFeature:
        in_feature = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
        )
        out_feature = self.voice_changer.convert_from_acoustic_feature(in_feature)

        pad = round(extra_time * self.in_segment_method.sampling_rate)
        if pad > 0:
            out_feature = self.out_segment_method.pick(out_feature, pad, -pad)

        return out_feature
