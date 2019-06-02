from typing import Iterable, List

import numpy
from yukarin.acoustic_feature import AcousticFeature

from ..segment.segment import BaseSegmentMethod
from ..yukarin_wrapper.voice_changer import AcousticFeatureWrapper


class FeatureWrapperSegmentMethod(BaseSegmentMethod[AcousticFeatureWrapper]):
    def __init__(
            self,
            sampling_rate: int,
            wave_sampling_rate: int,
            order: int,
            frame_period: int,
            keys: List[str] = None,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.wave_sampling_rate = wave_sampling_rate
        self.order = order
        self.frame_period = frame_period

        self._keys = ['f0', 'ap', 'mc', 'voiced'] if keys is None else keys

    def length(self, data: AcousticFeatureWrapper) -> int:
        return len(data.f0)

    def pad(self, width: int):
        sizes = AcousticFeature.get_sizes(sampling_rate=self.wave_sampling_rate, order=self.order)
        return AcousticFeatureWrapper.silent_wrapper(
            width,
            sizes=sizes,
            keys=self._keys,
            frame_period=self.frame_period,
            sampling_rate=self.wave_sampling_rate,
            wave_dtype=numpy.float32,
        ).astype_only_float_wrapper(numpy.float32)

    def pick(self, data: AcousticFeatureWrapper, first: int, last: int):
        return data.pick_wrapper(
            first,
            last,
            keys=self._keys,
            frame_period=self.frame_period,
        )

    def concat(self, datas: Iterable[AcousticFeatureWrapper]):
        return AcousticFeatureWrapper.concatenate_wrapper(list(datas), keys=self._keys)
