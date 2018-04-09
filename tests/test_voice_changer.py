import world4py

world4py._WORLD_LIBRARY_PATH = 'x64_world.dll'

from pathlib import Path
from typing import NamedTuple

import librosa
import numpy
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.wave import Wave
from yukarin.f0_converter import F0Converter

from realtime_voice_conversion.voice_changer_stream import VoiceChangerStream
from realtime_voice_conversion.voice_changer_stream import VoiceChangerStreamWrapper
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger


class AudioConfig(NamedTuple):
    rate: int
    chunk: int
    vocoder_buffer_size: int
    out_norm: float


model_base_path = Path('./trained/').expanduser()
test_data_path = Path('tests/test-deep-learning-yuduki-yukari.wav')
test_output_path = Path('output.wav')
input_statistics_path = model_base_path / 'f0_statistics/hiho_f0stat.npy'
target_statistics_path = model_base_path / 'f0_statistics/yukari_f0stat.npy'

print('model loading...', flush=True)

f0_converter = F0Converter(input_statistics=input_statistics_path, target_statistics=target_statistics_path)

model_path = model_base_path / Path('pp-el8-wof0/predictor_2260000.npz')
config_path = model_base_path / Path('pp-el8-wof0/config.json')
config = create_config(config_path)
acoustic_converter = AcousticConverter(config, model_path, f0_converter=f0_converter)
print('model 1 loaded!', flush=True)

model_path = model_base_path / Path('sr-noise3/predictor_180000.npz')
config_path = model_base_path / Path('sr-noise3/config.json')
sr_config = create_sr_config(config_path)
super_resolution = SuperResolution(sr_config, model_path)
print('model 2 loaded!', flush=True)

audio_config = AudioConfig(
    rate=config.dataset.acoustic_param.sampling_rate,
    chunk=config.dataset.acoustic_param.sampling_rate,
    vocoder_buffer_size=config.dataset.acoustic_param.sampling_rate // 16,
    out_norm=4.5,
)
frame_period = config.dataset.acoustic_param.frame_period

vocoder = RealtimeVocoder(
    acoustic_param=config.dataset.acoustic_param,
    out_sampling_rate=audio_config.rate,
    buffer_size=audio_config.vocoder_buffer_size,
    number_of_pointers=16,
)

voice_changer = VoiceChanger(
    super_resolution=super_resolution,
    acoustic_converter=acoustic_converter,
)

voice_changer_stream = VoiceChangerStream(
    sampling_rate=audio_config.rate,
    frame_period=acoustic_converter._param.frame_period,
    order=acoustic_converter._param.order,
    in_dtype=numpy.float32,
)

voice_changer_stream.voice_changer = voice_changer
voice_changer_stream.vocoder = vocoder

wrapper = VoiceChangerStreamWrapper(
    voice_changer_stream=voice_changer_stream,
    extra_time_pre=0.2,
    extra_time=0.1,
)

raw_wave, _ = librosa.load(str(test_data_path), sr=audio_config.rate)
wave_out_list = []

start_time = 0
for i in range(0, len(raw_wave), audio_config.chunk):
    wave_in = Wave(wave=raw_wave[i:i + audio_config.chunk], sampling_rate=audio_config.rate)
    wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=wave_in)
    start_time += len(wave_in.wave) / wave_in.sampling_rate

start_time = 0
for i in range(len(raw_wave) // audio_config.chunk + 1):
    feature_in = wrapper.pre_convert_next(time_length=audio_config.chunk / audio_config.rate)
    wrapper.voice_changer_stream.add_in_feature(
        start_time=start_time,
        feature_wrapper=feature_in,
        frame_period=frame_period,
    )
    start_time += audio_config.chunk / audio_config.rate
    print('pre', i, flush=True)

start_time = 0
for i in range(len(raw_wave) // audio_config.chunk + 1):
    feature_out = wrapper.convert_next(time_length=audio_config.chunk / audio_config.rate)
    wrapper.voice_changer_stream.add_out_feature(start_time=start_time, feature=feature_out, frame_period=frame_period)
    start_time += audio_config.chunk / audio_config.rate
    print('cent', i, flush=True)

start_time = 0
for i in range(len(raw_wave) // audio_config.chunk + 1):
    wave_out = wrapper.post_convert_next(time_length=audio_config.chunk / audio_config.rate)
    wave_out_list.append(wave_out)
    start_time += audio_config.chunk / audio_config.rate
    print('post', i, flush=True)

out_wave = numpy.concatenate([w.wave for w in wave_out_list]).astype(numpy.float32)
librosa.output.write_wav(str(test_output_path), out_wave, sr=audio_config.rate)
