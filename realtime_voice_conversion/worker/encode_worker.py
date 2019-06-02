import logging
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Lock

import numpy

from realtime_voice_conversion.stream import EncodeStream
from realtime_voice_conversion.stream import StreamWrapper
from realtime_voice_conversion.worker.utility import init_logger, Item
from realtime_voice_conversion.yukarin_wrapper.acoustic_feature_wrapper import AcousticFeatureWrapper
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder


def encode_worker(
        realtime_vocoder: RealtimeVocoder,
        time_length: float,
        extra_time: float,
        queue_input: Queue,
        queue_output: Queue,
        acquired_lock: Lock,
):
    logger = logging.getLogger('encode')
    init_logger(logger)
    logger.info('encode worker')

    stream = EncodeStream(vocoder=realtime_vocoder)
    stream_wrapper = StreamWrapper(stream=stream, extra_time=extra_time)

    acquired_lock.release()
    start_time = extra_time
    while True:
        item: Item = queue_input.get()
        start = time.time()
        wave: numpy.ndarray = item.item

        stream.add(start_time=start_time, data=wave)
        start_time += time_length

        feature_wrapper: AcousticFeatureWrapper = stream_wrapper.process_next(time_length=time_length)
        item.item = feature_wrapper
        queue_output.put(item)

        logger.debug(f'{item.index}: {time.time() - start}')
