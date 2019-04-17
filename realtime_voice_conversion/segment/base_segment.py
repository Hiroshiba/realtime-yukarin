from abc import abstractmethod
from typing import NamedTuple, TypeVar, Generic, Iterable, GenericMeta, NamedTupleMeta

T = TypeVar('T')


class BaseSegmentMethod(Generic[T]):
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate

    @abstractmethod
    def length(self, data: T) -> int:
        raise NotImplementedError()

    @abstractmethod
    def pad(self, width: int) -> T:
        raise NotImplementedError()

    @abstractmethod
    def pick(self, data: T, first: int, last: int) -> T:
        raise NotImplementedError()

    @abstractmethod
    def concat(self, datas: Iterable[T]) -> T:
        raise NotImplementedError()


class NamedTupleGenericMeta(NamedTupleMeta, GenericMeta):
    pass


class Segment(NamedTuple, Generic[T], metaclass=NamedTupleGenericMeta):
    start_time: float
    data: T
    method: BaseSegmentMethod[T]

    @property
    def sampling_rate(self) -> int:
        return self.method.sampling_rate

    @property
    def length(self) -> int:
        return self.method.length(self.data)

    @property
    def time_length(self) -> float:
        return self.length / self.sampling_rate

    @property
    def end_time(self) -> float:
        return self.time_length + self.start_time
