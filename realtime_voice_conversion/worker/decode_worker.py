import logging
import time
from multiprocessing import Queue

import librosa
import numpy
from yukarin.acoustic_feature import AcousticFeature
from yukarin.config import Config

from realtime_voice_conversion.stream import DecodeStream
from realtime_voice_conversion.stream import StreamWrapper
from realtime_voice_conversion.worker.utility import init_logger, Item, AudioConfig
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder


def decode_worker(
        config: Config,
        audio_config: AudioConfig,
        time_length: float,
        extra_time: float,
        queue_input: Queue,
        queue_output: Queue,
):
    logger = logging.getLogger('decode')
    init_logger(logger)
    logging.info('decode worker')

    stream = DecodeStream(
        vocoder=RealtimeVocoder(
            acoustic_param=config.dataset.acoustic_param,
            out_sampling_rate=audio_config.out_rate,
            buffer_size=audio_config.vocoder_buffer_size,
            number_of_pointers=16,
        )
    )
    stream_wrapper = StreamWrapper(stream=stream, extra_time=extra_time)

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
        if len(wave_fragment) >= audio_config.out_audio_chunk:
            wave, wave_fragment = \
                wave_fragment[:audio_config.out_audio_chunk], wave_fragment[audio_config.out_audio_chunk:]

            power = librosa.core.power_to_db(numpy.abs(librosa.stft(wave)) ** 2).mean()
            if power < audio_config.silent_threshold:
                wave = None  # pass
        else:
            wave = None

        item.item = wave
        queue_output.put(item)

        logger.debug(f'{item.index}: {time.time() - start}')
