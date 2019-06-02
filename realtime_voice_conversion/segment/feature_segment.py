from typing import Iterable

from yukarin.acoustic_feature import AcousticFeature

from ..segment.segment import BaseSegmentMethod
from ..yukarin_wrapper.voice_changer import AcousticFeatureWrapper


class FeatureSegmentMethod(BaseSegmentMethod[AcousticFeature]):
    def __init__(
            self,
            sampling_rate: int,
            wave_sampling_rate: int,
            order: int,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.wave_sampling_rate = wave_sampling_rate
        self.order = order

        self._keys = ['f0', 'ap', 'sp', 'voiced']

    def length(self, data: AcousticFeature) -> int:
        return len(data.f0)

    def pad(self, width: int):
        sizes = AcousticFeature.get_sizes(sampling_rate=self.wave_sampling_rate, order=self.order)
        return AcousticFeature.silent(width, sizes=sizes, keys=self._keys)

    def pick(self, data: AcousticFeatureWrapper, first: int, last: int):
        """

        :type data: object
        """
        return data.pick(first, last, keys=self._keys)

    def concat(self, datas: Iterable[AcousticFeatureWrapper]):
        return AcousticFeature.concatenate(list(datas), keys=self._keys)
