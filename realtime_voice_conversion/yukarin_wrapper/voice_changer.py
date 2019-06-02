import numpy
from become_yukarin import SuperResolution
from yukarin import AcousticConverter

from .acoustic_feature_wrapper import AcousticFeatureWrapper


class VoiceChanger(object):
    def __init__(
            self,
            acoustic_converter: AcousticConverter,
            super_resolution: SuperResolution,
            threshold: float = 60,
            output_sampling_rate: int = None,
    ) -> None:
        if output_sampling_rate is None:
            output_sampling_rate = super_resolution.config.dataset.param.voice_param.sample_rate

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution
        self.threshold = threshold
        self.output_sampling_rate = output_sampling_rate

    def convert_from_acoustic_feature(self, f_in: AcousticFeatureWrapper):
        w_in = f_in.wave

        f_in_effective, effective = self.acoustic_converter.separate_effective(
            wave=w_in,
            feature=f_in,
            threshold=self.threshold,
        )
        if numpy.any(effective):
            f_out = self.acoustic_converter.convert(f_in_effective)
        else:
            f_out = f_in_effective

        f_out = self.acoustic_converter.combine_silent(effective=effective, feature=f_out)
        f_out = self.acoustic_converter.decode_spectrogram(f_out)
        f_out.sp += 1e-16

        f_out.sp = self.super_resolution.convert(f_out.sp.astype(numpy.float32))
        return f_out
