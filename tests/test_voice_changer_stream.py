import numpy

from realtime_voice_conversion.voice_changer_stream import VoiceChangerStream
from realtime_voice_conversion.voice_changer_stream import Wave
from realtime_voice_conversion.voice_changer_stream import WaveSegment


def voice_changer_stream_fetch():
    start_time = 0
    time_length = 1
    rate = 100
    pad_function = lambda length: numpy.zeros(shape=length)
    pick_function = lambda segment, first, last: segment.wave.wave[first:last]
    concat_function = numpy.concatenate
    extra_time = 0.1

    wave = Wave(wave=numpy.ones(time_length * rate), sampling_rate=rate)
    segment1 = WaveSegment(start_time=start_time, wave=wave)
    segment2 = WaveSegment(start_time=start_time + time_length, wave=wave)
    data_stream = [segment1, segment2]

    out = VoiceChangerStream.fetch(
        start_time=start_time,
        time_length=time_length,
        data_stream=data_stream,
        rate=rate,
        pad_function=pad_function,
        pick_function=pick_function,
        concat_function=concat_function,
        extra_time=extra_time,
    )
    assert len(out) == 120
    assert (out == 0).sum() == 10
    assert (out == 1).sum() == 110


if __name__ == '__main__':
    voice_changer_stream_fetch()
