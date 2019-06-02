from typing import List, Dict, Iterable

import numpy
from yukarin import AcousticFeature
from yukarin.wave import Wave


class AcousticFeatureWrapper(AcousticFeature):
    def __init__(self, wave: Wave, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave = wave

    def __eq__(self, other):
        if not isinstance(other, AcousticFeatureWrapper):
            return NotImplemented
        return \
            numpy.all(other.wave.wave == self.wave.wave) and \
            other.wave.sampling_rate == self.wave.sampling_rate and \
            numpy.all(other.f0 == self.f0)

    def astype_only_float_wrapper(self, dtype):
        return AcousticFeatureWrapper(
            wave=Wave(wave=self.wave.wave.astype(dtype), sampling_rate=self.wave.sampling_rate),
            **self.astype_only_float(dtype).__dict__,
        )

    @classmethod
    def extract(cls, wave: Wave, *args, **kwargs):
        return cls(
            wave=wave,
            **super().extract(wave, *args, **kwargs).__dict__,
        )

    @staticmethod
    def silent_wrapper(
            length: int,
            sizes: Dict[str, int],
            keys: Iterable[str],
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
    def concatenate_wrapper(fs: List['AcousticFeatureWrapper'], keys: Iterable[str]):
        return AcousticFeatureWrapper(
            wave=Wave(wave=numpy.concatenate([f.wave.wave for f in fs]), sampling_rate=fs[0].wave.sampling_rate),
            **AcousticFeatureWrapper.concatenate(fs, keys=keys).__dict__,
        )

    def pick_wrapper(self, first: int, last: int, keys: Iterable[str], frame_period: float):
        first_wave = round(first * frame_period / 1000 * self.wave.sampling_rate)
        last_wave = round(last * frame_period / 1000 * self.wave.sampling_rate)
        return AcousticFeatureWrapper(
            wave=Wave(wave=self.wave.wave[first_wave:last_wave], sampling_rate=self.wave.sampling_rate),
            **self.pick(first, last, keys=keys).__dict__,
        )


class CrepeAcousticFeatureWrapper(AcousticFeatureWrapper):
    @classmethod
    def extract_f0(cls, x: numpy.ndarray, fs: int, frame_period: int, f0_floor: float, f0_ceil: float):
        import crepe
        t, f0, confidence, _ = crepe.predict(
            x,
            fs,
            viterbi=True,
            model_capacity='full',
            step_size=frame_period,
            verbose=0,
        )
        voiced = (crepe.predict_voicing(confidence) == 1) | (confidence > 0.1)
        f0[~voiced] = 0

        return f0, t
