import logging
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Lock

import librosa
import numpy
from yukarin.acoustic_feature import AcousticFeature

from realtime_voice_conversion.stream import DecodeStream
from realtime_voice_conversion.stream import StreamWrapper
from realtime_voice_conversion.worker.utility import init_logger, Item
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder


def decode_worker(
        realtime_vocoder: RealtimeVocoder,
        time_length: float,
        extra_time: float,
        vocoder_buffer_size: int,
        out_audio_chunk: int,
        output_silent_threshold: float,
        queue_input: Queue,
        queue_output: Queue,
        acquired_lock: Lock,
):
    logger = logging.getLogger('decode')
    init_logger(logger)
    logging.info('decode worker')

    realtime_vocoder.create_synthesizer(
        buffer_size=vocoder_buffer_size,
        number_of_pointers=16,
    )
    stream = DecodeStream(vocoder=realtime_vocoder)
    stream_wrapper = StreamWrapper(stream=stream, extra_time=extra_time)

    acquired_lock.release()
    start_time = extra_time
    wave_fragment = numpy.empty(0)
    while True:
        item: Item = queue_input.get()
        start = time.time()
        feature: AcousticFeature = item.item
        stream.add(
            start_time=start_time,
            data=feature,
        )
        start_time += time_length

        wave = stream_wrapper.process_next(time_length=time_length)

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        if len(wave_fragment) >= out_audio_chunk:
            wave, wave_fragment = wave_fragment[:out_audio_chunk], wave_fragment[out_audio_chunk:]

            power = librosa.core.power_to_db(numpy.abs(librosa.stft(wave)) ** 2).mean()
            if power < - output_silent_threshold:
                wave = None  # pass
        else:
            wave = None

        item.item = wave
        queue_output.put(item)

        logger.debug(f'{item.index}: {time.time() - start}')
