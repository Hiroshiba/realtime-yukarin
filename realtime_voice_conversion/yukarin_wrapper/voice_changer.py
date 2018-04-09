from typing import List, Dict, Tuple

import numpy
from become_yukarin import SuperResolution
from yukarin import AcousticConverter
from yukarin import AcousticFeature
from yukarin.wave import Wave


class AcousticFeatureWrapper(AcousticFeature):
    def __init__(self, wave: Wave, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave = wave

    def astype_only_float_wrapper(self, dtype):
        return AcousticFeatureWrapper(
            wave=Wave(wave=self.wave.wave.astype(dtype), sampling_rate=self.wave.sampling_rate),
            **self.astype_only_float(dtype).__dict__,
        )

    @staticmethod
    def silent_wrapper(
            length: int,
            sizes: Dict[str, int],
            keys: Tuple[str],
            frame_period: float,
            sampling_rate: int,
            wave_dtype,
    ):
        length_wave = round(length * frame_period / 1000 * sampling_rate)
        return AcousticFeatureWrapper(
            wave=Wave(wave=numpy.zeros(shape=length_wave, dtype=wave_dtype), sampling_rate=sampling_rate),
            **AcousticFeatureWrapper.silent(length, sizes=sizes, keys=keys).__dict__,
        )

    @staticmethod
    def concatenate_wrapper(fs: List['AcousticFeatureWrapper'], keys: List[str]):
        return AcousticFeatureWrapper(
            wave=Wave(wave=numpy.concatenate([f.wave.wave for f in fs]), sampling_rate=fs[0].wave.sampling_rate),
            **AcousticFeatureWrapper.concatenate(fs, keys=keys).__dict__,
        )

    def pick_wrapper(self, first: int, last: int, keys: List[str], frame_period: float):
        first_wave = round(first * frame_period / 1000 * self.wave.sampling_rate)
        last_wave = round(last * frame_period / 1000 * self.wave.sampling_rate)
        return AcousticFeatureWrapper(
            wave=Wave(wave=self.wave.wave[first_wave:last_wave], sampling_rate=self.wave.sampling_rate),
            **self.pick(first, last, keys=keys).__dict__,
        )


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
