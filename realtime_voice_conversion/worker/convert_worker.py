import logging
from multiprocessing import Queue

from become_yukarin import SuperResolution
from yukarin import AcousticConverter
from yukarin.config import Config

from realtime_voice_conversion.stream import ConvertStream
from realtime_voice_conversion.stream import StreamWrapper
from realtime_voice_conversion.worker.utility import init_logger, Item, AudioConfig
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger

import time


def convert_worker(
        config: Config,
        acoustic_converter: AcousticConverter,
        super_resolution: SuperResolution,
        audio_config: AudioConfig,
        time_length: float,
        extra_time: float,
        queue_input: Queue,
        queue_output: Queue,
):
    logger = logging.getLogger('convert')
    init_logger(logger)
    logging.info('convert worker')

    stream = ConvertStream(
        voice_changer=VoiceChanger(
            super_resolution=super_resolution,
            acoustic_converter=acoustic_converter,
            threshold=audio_config.input_silent_threshold,
        )
    )
    stream_wrapper = StreamWrapper(stream=stream, extra_time=extra_time)

    start_time = extra_time
    while True:
        item: Item = queue_input.get()
        start = time.time()
        in_feature: AcousticFeatureWrapper = item.item
        stream.add(
            start_time=start_time,
            data=in_feature,
        )
        start_time += time_length

        out_feature = stream_wrapper.process_next(time_length=time_length)
        item.item = out_feature
        queue_output.put(item)

        logger.debug(f'{item.index}: {time.time() - start}')
