import argparse
import logging
import queue
import signal
import sys
from multiprocessing import Process, Lock
from multiprocessing import Queue
from pathlib import Path
from typing import List, Optional

import numpy
import pyaudio

from realtime_voice_conversion.config import Config
from realtime_voice_conversion.converter.yukarin_converter import YukarinConverter
from realtime_voice_conversion.worker import encode_worker, convert_worker, decode_worker
from realtime_voice_conversion.worker.utility import init_logger, Item
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder


def run(
        config_path: Path,
):
    logger = logging.getLogger('root')
    init_logger(logger)

    logger.info('model loading...')

    config = Config.from_yaml(config_path)

    converter = YukarinConverter.make_yukarin_converter(
        input_statistics_path=config.input_statistics_path,
        target_statistics_path=config.target_statistics_path,
        stage1_model_path=config.stage1_model_path,
        stage1_config_path=config.stage1_config_path,
        stage2_model_path=config.stage2_model_path,
        stage2_config_path=config.stage2_config_path,
    )

    realtime_vocoder = RealtimeVocoder(
        acoustic_param=converter.acoustic_converter.config.dataset.acoustic_param,
        out_sampling_rate=config.output_rate,
        extract_f0_mode=config.extract_f0_mode,
    )

    audio_instance = pyaudio.PyAudio()

    queue_input_wave: Queue[Item] = Queue()
    queue_input_feature: Queue[Item] = Queue()
    queue_output_feature: Queue[Item] = Queue()
    queue_output_wave: Queue[Item] = Queue()

    lock_encoder = Lock()
    lock_converter = Lock()
    lock_decoder = Lock()

    lock_encoder.acquire()
    process_encoder = Process(target=encode_worker, kwargs=dict(
        realtime_vocoder=realtime_vocoder,
        time_length=config.buffer_time,
        extra_time=config.encode_extra_time,
        queue_input=queue_input_wave,
        queue_output=queue_input_feature,
        acquired_lock=lock_encoder,
    ))
    process_encoder.start()

    lock_converter.acquire()
    process_converter = Process(target=convert_worker, kwargs=dict(
        acoustic_converter=converter.acoustic_converter,
        super_resolution=converter.super_resolution,
        time_length=config.buffer_time,
        extra_time=config.convert_extra_time,
        input_silent_threshold=config.input_silent_threshold,
        queue_input=queue_input_feature,
        queue_output=queue_output_feature,
        acquired_lock=lock_converter,
    ))
    process_converter.start()

    lock_decoder.acquire()
    process_decoder = Process(target=decode_worker, kwargs=dict(
        realtime_vocoder=realtime_vocoder,
        time_length=config.buffer_time,
        extra_time=config.decode_extra_time,
        vocoder_buffer_size=config.vocoder_buffer_size,
        out_audio_chunk=config.out_audio_chunk,
        output_silent_threshold=config.output_silent_threshold,
        queue_input=queue_output_feature,
        queue_output=queue_output_wave,
        acquired_lock=lock_decoder,
    ))
    process_decoder.start()

    with lock_encoder, lock_converter, lock_decoder:
        pass  # wait

    # input device
    if config.input_device_name is None:
        input_device_index = audio_instance.get_default_input_device_info()['index']

    else:
        for i in range(audio_instance.get_device_count()):
            if config.input_device_name in str(audio_instance.get_device_info_by_index(i)['name']):
                input_device_index = i
                break
        else:
            raise ValueError('input device not found')

    # output device
    if config.output_device_name is None:
        output_device_index = audio_instance.get_default_output_device_info()['index']

    else:
        for i in range(audio_instance.get_device_count()):
            if config.output_device_name in str(audio_instance.get_device_info_by_index(i)['name']):
                output_device_index = i
                break
        else:
            raise ValueError('output device not found')

    # audio stream
    audio_input_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=config.input_rate,
        frames_per_buffer=config.in_audio_chunk,
        input=True,
        input_device_index=input_device_index,
    )

    audio_output_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=config.output_rate,
        frames_per_buffer=config.out_audio_chunk,
        output=True,
        output_device_index=output_device_index,
    )

    # signal
    def signal_handler(s, f):
        process_encoder.terminate()
        process_converter.terminate()
        process_decoder.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    logger.debug('audio loop')

    index_input = 0
    index_output = 0
    popped_list: List[Item] = []
    while True:
        # input audio
        in_data = audio_input_stream.read(config.in_audio_chunk)
        in_wave = numpy.frombuffer(in_data, dtype=numpy.float32) * config.input_scale

        in_item = Item(
            item=in_wave,
            index=index_input,
        )
        queue_input_wave.put(in_item)

        logger.debug(f'input {index_input}')
        index_input += 1

        # output
        out_wave: Optional[numpy.ndarray] = None
        while True:
            try:
                while True:  # get all item in queue, for "cut in line"
                    item: Item = queue_output_wave.get_nowait()
                    popped_list.append(item)
            except queue.Empty:
                pass

            out_item = next(filter(lambda ii: ii.index == index_output, popped_list), None)
            if out_item is None:
                break

            popped_list.remove(out_item)

            logger.debug(f'output {index_output}')
            index_output += 1

            out_wave = out_item.item
            if out_wave is None:  # silence wave
                continue

            break

        if out_wave is None:
            out_wave = numpy.zeros(config.out_audio_chunk)
        out_wave *= config.output_scale

        b = out_wave[:config.out_audio_chunk].astype(numpy.float32).tobytes()
        audio_output_stream.write(b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, default=Path('./config.yaml'))
    args = parser.parse_args()

    run(
        config_path=args.config_path,
    )
