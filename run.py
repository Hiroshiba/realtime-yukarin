import argparse
import logging
import queue
import signal
import sys
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from typing import List

import chainer
import numpy
import pyaudio
import pynput
import world4py

from realtime_voice_conversion.worker import encode_worker, convert_worker, decode_worker
from realtime_voice_conversion.worker.utility import init_logger, AudioConfig, Item

world4py._WORLD_LIBRARY_PATH = 'x64_world.dll'

from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter


def main():
    logger = logging.getLogger('root')
    init_logger(logger)

    parser = argparse.ArgumentParser()
    parser.add_argument('-idn', '--input_device_name')
    parser.add_argument('-odn', '--output_device_name')
    args = parser.parse_args()

    logger.info('model loading...')

    chainer.global_config.enable_backprop = False
    chainer.global_config.train = False

    queue_input_wave = Queue()
    queue_input_feature = Queue()
    queue_output_feature = Queue()
    queue_output_wave = Queue()

    input_statistics_path = Path('./trained/f0_statistics/hiho_f0stat.npy')
    target_statistics_path = Path('./trained/f0_statistics/yukari_f0stat.npy')
    f0_converter = F0Converter(input_statistics=input_statistics_path, target_statistics=target_statistics_path)
    # model_path = Path('./trained/f0trans-wmc-multi-ref-el8-woD/predictor_13840000.npz')
    # config_path = Path('./trained/f0trans-wmc-multi-ref-el8-woD/config.json')
    # f0_converter = AcousticConverter(create_config(config_path), model_path, gpu=0)

    model_path = Path('./trained/multi-16k-ref24k-el8-woD-gbc8/predictor_2910000.npz')
    config_path = Path('./trained/multi-16k-ref24k-el8-woD-gbc8/config.json')
    # model_path = Path('./trained/akane-multi-ref-el8-woD-gbc8/predictor_5130000.npz')
    # config_path = Path('./trained/akane-multi-ref-el8-woD-gbc8/config.json')
    # model_path = Path('./trained/aoi-multi-ref-el8-woD-gbc8/predictor_5720000.npz')
    # config_path = Path('./trained/aoi-multi-ref-el8-woD-gbc8/config.json')
    # model_path = Path('./trained/zunko-multi-ref-el8-woD-gbc8/predictor_5710000.npz')
    # config_path = Path('./trained/zunko-multi-ref-el8-woD-gbc8/config.json')
    config = create_config(config_path)
    acoustic_converter = AcousticConverter(
        config,
        model_path,
        gpu=0,
        f0_converter=f0_converter,
        out_sampling_rate=24000,
    )
    logger.info('model 1 loaded!')

    # model_path = Path('./trained/sr-noise3/predictor_180000.npz')
    # config_path = Path('./trained/sr-noise3/config.json')
    model_path = Path('./trained/sr-noise3-more-gbc8/predictor_400000.npz')
    config_path = Path('./trained/sr-noise3-more-gbc8/config.json')
    # model_path = Path('./trained/akane-super-resolution/predictor_240000.npz')
    # config_path = Path('./trained/akane-super-resolution/config.json')
    sr_config = create_sr_config(config_path)
    super_resolution = SuperResolution(sr_config, model_path, gpu=0)
    logger.info('model 2 loaded!')

    buffer_time = 0.5

    audio_instance = pyaudio.PyAudio()
    audio_config = AudioConfig(
        in_rate=config.dataset.acoustic_param.sampling_rate,
        out_rate=24000,
        frame_period=config.dataset.acoustic_param.frame_period,
        in_audio_chunk=round(config.dataset.acoustic_param.sampling_rate * buffer_time),
        out_audio_chunk=round(24000 * buffer_time),
        vocoder_buffer_size=config.dataset.acoustic_param.sampling_rate // 16,
        in_norm=1 / 8,
        out_norm=2.0,
        input_silent_threshold=80,
        silent_threshold=-80.0,
    )

    encode_extra_time = 0.0
    encode_time_length = buffer_time

    convert_extra_time = 0.5
    convert_time_length = buffer_time

    decode_extra_time = 0.0
    decode_time_length = buffer_time

    conversion_flag = True

    process_encoder = Process(target=encode_worker, kwargs=dict(
        config=config,
        audio_config=audio_config,
        time_length=encode_time_length,
        extra_time=encode_extra_time,
        queue_input=queue_input_wave,
        queue_output=queue_input_feature,
    ))
    process_encoder.start()

    process_converter = Process(target=convert_worker, kwargs=dict(
        config=config,
        acoustic_converter=acoustic_converter,
        super_resolution=super_resolution,
        audio_config=audio_config,
        time_length=convert_time_length,
        extra_time=convert_extra_time,
        queue_input=queue_input_feature,
        queue_output=queue_output_feature,
    ))
    process_converter.start()

    process_decoder = Process(target=decode_worker, kwargs=dict(
        config=config,
        audio_config=audio_config,
        time_length=decode_time_length,
        extra_time=decode_extra_time,
        queue_input=queue_output_feature,
        queue_output=queue_output_wave,
    ))
    process_decoder.start()

    # input device
    name = args.input_device_name
    if name is None:
        input_device_index = audio_instance.get_default_input_device_info()['index']

    else:
        for i in range(audio_instance.get_device_count()):
            if name in str(audio_instance.get_device_info_by_index(i)['name']):
                input_device_index = i
                break
        else:
            logger.info('input device not found')
            exit(1)

    # output device
    name = args.output_device_name
    if name is None:
        output_device_index = audio_instance.get_default_output_device_info()['index']

    else:
        for i in range(audio_instance.get_device_count()):
            if name in str(audio_instance.get_device_info_by_index(i)['name']):
                output_device_index = i
                break
        else:
            logger.info('output device not found')
            exit(1)

    # audio stream
    audio_input_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.in_rate,
        frames_per_buffer=audio_config.in_audio_chunk,
        input=True,
        input_device_index=input_device_index,
    )

    audio_output_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.out_rate,
        frames_per_buffer=audio_config.out_audio_chunk // 2,  # if without divide, the latency will be twice. Why?
        output=True,
        output_device_index=output_device_index,
    )

    # signal
    def signal_handler(*args, **kwargs):
        process_encoder.terminate()
        process_converter.terminate()
        process_decoder.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # key event
    def key_handler(key):
        nonlocal conversion_flag
        if key == pynput.keyboard.Key.space:  # switch
            conversion_flag = not conversion_flag

    key_listener = pynput.keyboard.Listener(on_press=key_handler)
    key_listener.start()

    logger.debug('audio loop')

    index_input = 0
    index_output = 0
    popped_list: List[Item] = []
    while True:
        # input audio
        in_data = audio_input_stream.read(audio_config.in_audio_chunk)
        wave = numpy.frombuffer(in_data, dtype=numpy.float32) * audio_config.in_norm

        item = Item(
            original=wave * 5,
            item=wave,
            index=index_input,
            conversion_flag=conversion_flag,
        )
        queue_input_wave.put(item)

        logger.debug(f'input {index_input}')
        index_input += 1

        # logger.debug(f'queue_input_wave: {queue_input_wave.qsize()}', )
        # logger.debug(f'queue_input_feature: {queue_input_feature.qsize()}', )
        # logger.debug(f'queue_output_feature: {queue_output_feature.qsize()}', )
        # logger.debug(f'queue_output_wave: {queue_output_wave.qsize()}', )

        # output
        wave: numpy.ndarray = None

        while True:
            try:
                while True:  # get all item in queue, for "cut in line"
                    item: Item = queue_output_wave.get_nowait()
                    popped_list.append(item)
            except queue.Empty:
                pass

            item = next(filter(lambda ii: ii.index == index_output, popped_list), None)
            if item is None:
                break

            popped_list.remove(item)

            logger.debug(f'output {index_output}')
            index_output += 1
            if item.item is None:  # silence wave
                continue

            wave = item.item if item.conversion_flag else item.original
            break

        if wave is None:
            wave = numpy.zeros(audio_config.out_audio_chunk)
        wave *= audio_config.out_norm

        b = wave[:audio_config.out_audio_chunk // 2].astype(numpy.float32).tobytes()
        audio_output_stream.write(b)

        b = wave[audio_config.out_audio_chunk // 2:].astype(numpy.float32).tobytes()
        audio_output_stream.write(b)


if __name__ == '__main__':
    main()
