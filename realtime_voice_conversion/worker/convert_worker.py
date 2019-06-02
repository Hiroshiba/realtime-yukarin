import logging
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Lock

import chainer
from become_yukarin import SuperResolution
from yukarin import AcousticConverter

from realtime_voice_conversion.stream import ConvertStream
from realtime_voice_conversion.stream import StreamWrapper
from realtime_voice_conversion.worker.utility import init_logger, Item
from realtime_voice_conversion.yukarin_wrapper.acoustic_feature_wrapper import AcousticFeatureWrapper
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger


def convert_worker(
        acoustic_converter: AcousticConverter,
        super_resolution: SuperResolution,
        time_length: float,
        extra_time: float,
        input_silent_threshold: float,
        queue_input: Queue,
        queue_output: Queue,
        acquired_lock: Lock,
):
    logger = logging.getLogger('convert')
    init_logger(logger)
    logging.info('convert worker')

    chainer.global_config.enable_backprop = False
    chainer.global_config.train = False

    stream = ConvertStream(
        voice_changer=VoiceChanger(
            super_resolution=super_resolution,
            acoustic_converter=acoustic_converter,
            threshold=input_silent_threshold,
        )
    )
    stream_wrapper = StreamWrapper(stream=stream, extra_time=extra_time)

    acquired_lock.release()
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
