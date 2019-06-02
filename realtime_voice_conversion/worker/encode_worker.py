import logging
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Lock

import numpy
from yukarin import AcousticFeature

from realtime_voice_conversion.stream import EncodeStream
from realtime_voice_conversion.stream import StreamWrapper
from realtime_voice_conversion.worker.utility import init_logger, Item
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper


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

    def _extract_f0(cls, x: numpy.ndarray, fs: int, frame_period: int, f0_floor: float, f0_ceil: float):
        import crepe
        t, f0, confidence, _ = crepe.predict(
            x,
            fs,
            viterbi=True,
            model_capacity='full',
            step_size=frame_period,
            verbose=0,
        )
        voiced = (crepe.predict_voicing(confidence) == 1) | (confidence > 0.1)
        f0[~voiced] = 0

        return f0, t

    import types
    AcousticFeature.extract_f0 = types.MethodType(_extract_f0, AcousticFeature)

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
