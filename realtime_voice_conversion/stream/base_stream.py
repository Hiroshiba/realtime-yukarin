from abc import abstractmethod
from typing import List, TypeVar, Generic

from realtime_voice_conversion.segment.segment import BaseSegmentMethod, Segment

T_IN = TypeVar('T_IN')
T_OUT = TypeVar('T_OUT')


class BaseStream(Generic[T_IN, T_OUT]):
    def __init__(
            self,
            in_segment_method: BaseSegmentMethod[T_IN],
            out_segment_method: BaseSegmentMethod[T_OUT],
    ):
        self.in_segment_method = in_segment_method
        self.out_segment_method = out_segment_method

        self.stream: List[Segment[T_IN]] = []

    def add(self, start_time: float, data: T_IN):
        segment = Segment(
            start_time=start_time,
            data=data,
            method=self.in_segment_method,
        )
        self.stream.append(segment)

    def remove(self, end_time: float):
        self.stream = list(filter(lambda s: s.end_time > end_time, self.stream))

    def fetch(
            self,
            start_time: float,
            time_length: float,
            extra_time: float,
    ) -> T_IN:
        rate = self.in_segment_method.sampling_rate
        start_time -= extra_time
        time_length += extra_time * 2

        end_time = start_time + time_length
        stream = filter(lambda s: not (end_time < s.start_time or s.end_time < start_time), self.stream)

        start_time_buffer = start_time
        remaining_time = time_length
        buffer_list: List[T_IN] = []
        for segment in stream:
            # padding
            if segment.start_time > start_time_buffer:
                length = round((segment.start_time - start_time_buffer) * rate)
                pad = self.in_segment_method.pad(length)
                buffer_list.append(pad)
                remaining_time -= segment.start_time - start_time_buffer
                start_time_buffer = segment.start_time

            if remaining_time > segment.end_time - start_time_buffer:
                one_time_length = segment.end_time - start_time_buffer
            else:
                one_time_length = remaining_time

            first_index = round((start_time_buffer - segment.start_time) * rate)
            last_index = round(first_index + one_time_length * rate)
            one_buffer = self.in_segment_method.pick(segment.data, first_index, last_index)
            buffer_list.append(one_buffer)

            start_time_buffer += one_time_length
            remaining_time -= one_time_length

            if start_time_buffer >= end_time:
                break
        else:
            # last padding
            length = round((end_time - start_time_buffer) * rate)
            pad = self.in_segment_method.pad(length)
            buffer_list.append(pad)

        buffer = self.in_segment_method.concat(buffer_list)
        return buffer

    @abstractmethod
    def process(self, start_time: float, time_length: float, extra_time: float) -> T_OUT:
        raise NotImplementedError()
