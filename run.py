import queue
import signal
import sys
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from typing import NamedTuple, List

import librosa
import numpy
import pyaudio
import pynput
import world4py

world4py._WORLD_LIBRARY_PATH = 'x64_world.dll'

from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter
from yukarin.acoustic_feature import AcousticFeature
from yukarin.config import Config
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter
from yukarin.wave import Wave

from realtime_voice_conversion.voice_changer_stream import VoiceChangerStream
from realtime_voice_conversion.voice_changer_stream import VoiceChangerStreamWrapper
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder
from realtime_voice_conversion.yukarin_wrapper.vocoder import Vocoder
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger


class AudioConfig(NamedTuple):
    rate: int
    frame_period: float
    audio_chunk: int
    convert_chunk: int
    vocoder_buffer_size: int
    in_norm: float
    out_norm: float
    silent_threshold: float


class Item(object):
    def __init__(
            self,
            original: numpy.ndarray,
            item: any,
            index: int,
            conversion_flag: bool,
    ):
        self.original = original
        self.item = item
        self.index = index
        self.conversion_flag = conversion_flag


def encode_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.vocoder = Vocoder(
        acoustic_param=config.dataset.acoustic_param,
        out_sampling_rate=audio_config.rate,
    )

    start_time = 0
    time_length = audio_config.convert_chunk / audio_config.rate

    # padding 1s
    prev_original = numpy.zeros(round(time_length * audio_config.rate), dtype=numpy.float32)
    w = Wave(wave=prev_original, sampling_rate=audio_config.rate)
    wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
    start_time += time_length

    while True:
        item: Item = queue_input.get()
        item.original, prev_original = prev_original, item.original
        wave = item.item

        w = Wave(wave=wave, sampling_rate=audio_config.rate)
        wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
        start_time += time_length

        feature_wrapper = wrapper.pre_convert_next(time_length=time_length)
        item.item = feature_wrapper
        queue_output.put(item)


def convert_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        acoustic_converter: AcousticConverter,
        super_resolution: SuperResolution,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.voice_changer = VoiceChanger(
        super_resolution=super_resolution,
        acoustic_converter=acoustic_converter,
    )

    start_time = 0
    time_length = audio_config.convert_chunk / audio_config.rate
    while True:
        item: Item = queue_input.get()
        in_feature: AcousticFeatureWrapper = item.item
        wrapper.voice_changer_stream.add_in_feature(
            start_time=start_time,
            feature_wrapper=in_feature,
            frame_period=audio_config.frame_period,
        )
        start_time += time_length

        out_feature = wrapper.convert_next(time_length=time_length)
        item.item = out_feature
        queue_output.put(item)


def decode_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.vocoder = RealtimeVocoder(
        acoustic_param=config.dataset.acoustic_param,
        out_sampling_rate=audio_config.rate,
        buffer_size=audio_config.vocoder_buffer_size,
        number_of_pointers=16,
    )

    start_time = 0
    time_length = audio_config.convert_chunk / audio_config.rate
    wave_fragment = numpy.empty(0)
    while True:
        item: Item = queue_input.get()
        feature: AcousticFeature = item.item
        wrapper.voice_changer_stream.add_out_feature(
            start_time=start_time,
            feature=feature,
            frame_period=audio_config.frame_period,
        )
        start_time += time_length

        wave = wrapper.post_convert_next(time_length=time_length).wave

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        if len(wave_fragment) >= audio_config.audio_chunk:
            wave, wave_fragment = wave_fragment[:audio_config.audio_chunk], wave_fragment[audio_config.audio_chunk:]

            power = librosa.core.power_to_db(numpy.abs(librosa.stft(wave)) ** 2).mean()
            if power < audio_config.silent_threshold:
                wave = None  # pass
        else:
            wave = None

        item.item = wave
        queue_output.put(item)


def main():
    print('model loading...', flush=True)

    queue_input_wave = Queue()
    queue_input_feature = Queue()
    queue_output_feature = Queue()
    queue_output_wave = Queue()

    input_statistics_path = Path('./trained/f0_statistics/hiho_f0stat.npy')
    target_statistics_path = Path('./trained/f0_statistics/yukari_f0stat.npy')
    f0_converter = F0Converter(input_statistics=input_statistics_path, target_statistics=target_statistics_path)

    model_path = Path('./trained/pp-el8-wof0/predictor_2260000.npz')
    config_path = Path('./trained/pp-el8-wof0/config.json')
    config = create_config(config_path)
    acoustic_converter = AcousticConverter(config, model_path, gpu=0, f0_converter=f0_converter)
    print('model 1 loaded!', flush=True)

    model_path = Path('./trained/sr-noise3/predictor_180000.npz')
    config_path = Path('./trained/sr-noise3/config.json')
    sr_config = create_sr_config(config_path)
    super_resolution = SuperResolution(sr_config, model_path, gpu=0)
    print('model 2 loaded!', flush=True)

    audio_instance = pyaudio.PyAudio()
    audio_config = AudioConfig(
        rate=config.dataset.acoustic_param.sampling_rate,
        frame_period=config.dataset.acoustic_param.frame_period,
        audio_chunk=config.dataset.acoustic_param.sampling_rate,
        convert_chunk=config.dataset.acoustic_param.sampling_rate,
        vocoder_buffer_size=config.dataset.acoustic_param.sampling_rate // 16,
        in_norm=1 / 8,
        out_norm=4.0,
        silent_threshold=-80.0,
    )

    conversion_flag = True

    voice_changer_stream = VoiceChangerStream(
        sampling_rate=audio_config.rate,
        frame_period=config.dataset.acoustic_param.frame_period,
        order=config.dataset.acoustic_param.order,
        in_dtype=numpy.float32,
    )

    wrapper = VoiceChangerStreamWrapper(
        voice_changer_stream=voice_changer_stream,
        extra_time_pre=0.2,
        extra_time=0.5,
    )

    process_encoder = Process(target=encode_worker, kwargs=dict(
        config=config,
        wrapper=wrapper,
        audio_config=audio_config,
        queue_input=queue_input_wave,
        queue_output=queue_input_feature,
    ))
    process_encoder.start()

    process_converter = Process(target=convert_worker, kwargs=dict(
        config=config,
        wrapper=wrapper,
        acoustic_converter=acoustic_converter,
        super_resolution=super_resolution,
        audio_config=audio_config,
        queue_input=queue_input_feature,
        queue_output=queue_output_feature,
    ))
    process_converter.start()

    process_decoder = Process(target=decode_worker, kwargs=dict(
        config=config,
        wrapper=wrapper,
        audio_config=audio_config,
        queue_input=queue_output_feature,
        queue_output=queue_output_wave,
    ))
    process_decoder.start()

    audio_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.rate,
        frames_per_buffer=audio_config.audio_chunk,
        input=True,
        output=True,
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

    index_input = 0
    index_output = 0
    while True:
        # input audio
        in_data = audio_stream.read(audio_config.audio_chunk)
        wave = numpy.fromstring(in_data, dtype=numpy.float32) * audio_config.in_norm

        item = Item(
            original=wave * 5,
            item=wave,
            index=index_input,
            conversion_flag=conversion_flag,
        )
        queue_input_wave.put(item)
        index_input += 1

        print('queue_input_wave', queue_input_wave.qsize(), flush=True)
        print('queue_input_feature', queue_input_feature.qsize(), flush=True)
        print('queue_output_feature', queue_output_feature.qsize(), flush=True)
        print('queue_output_wave', queue_output_wave.qsize(), flush=True)

        # output
        wave: numpy.ndarray = None
        popped_list: List[Item] = []

        while True:
            try:
                while True:
                    item: Item = queue_output_wave.get_nowait()
                    popped_list.append(item)
            except queue.Empty:
                pass

            print('index_output', index_output)
            item = next(filter(lambda ii: ii.index == index_output, popped_list), None)
            if item is None:
                break

            popped_list.remove(item)

            index_output += 1
            if item.item is None:
                continue

            wave = item.item if item.conversion_flag else item.original
            break

        if wave is not None:
            wave *= audio_config.out_norm
            b = wave.astype(numpy.float32).tobytes()
            audio_stream.write(b)


if __name__ == '__main__':
    main()
