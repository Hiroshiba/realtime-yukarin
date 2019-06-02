from ..stream.base_stream import BaseStream


class StreamWrapper(object):
    def __init__(self, stream: BaseStream, extra_time: float):
        self.stream = stream
        self.extra_time = extra_time

        self._current_time = 0.

    def process_next(self, time_length: float):
        data = self.stream.process(
            start_time=self._current_time,
            time_length=time_length,
            extra_time=self.extra_time,
        )
        self._current_time += time_length
        return data
