import os
from pathlib import Path

import librosa
import numpy
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter, Wave, AcousticFeature
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter

from realtime_voice_conversion.config import VocodeMode
from realtime_voice_conversion.stream import EncodeStream, ConvertStream
from realtime_voice_conversion.yukarin_wrapper.vocoder import Vocoder
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper, VoiceChanger


def equal_feature(a: AcousticFeature, b: AcousticFeature):
    return numpy.all(a.f0 == b.f0)


class StreamHelper(object):
    def __init__(self):
        self.input_statistics_path = os.getenv('INPUT_STATISTICS')
        self.target_statistics_path = os.getenv('TARGET_STATISTICS')
        self.acoustic_convert_model_path = os.getenv('ACOUSTIC_CONVERT_MODEL')
        self.acoustic_convert_config_path = os.getenv('ACOUSTIC_CONVERT_CONFIG')
        self.super_resolution_model_path = os.getenv('SUPER_RESOLUTION_MODEL')
        self.super_resolution_config_path = os.getenv('SUPER_RESOLUTION_CONFIG')

        if self.input_statistics_path is None: raise ValueError('INPUT_STATISTICS is not found.')
        if self.target_statistics_path is None: raise ValueError('TARGET_STATISTICS is not found.')
        if self.acoustic_convert_model_path is None: raise ValueError('ACOUSTIC_CONVERT_MODEL is not found.')
        if self.acoustic_convert_config_path is None: raise ValueError('ACOUSTIC_CONVERT_CONFIG is not found.')
        if self.super_resolution_model_path is None: raise ValueError('SUPER_RESOLUTION_MODEL is not found.')
        if self.super_resolution_config_path is None: raise ValueError('SUPER_RESOLUTION_CONFIG is not found.')

        self._ac_config = None
        self._sr_config = None
        self._in_rate = None
        self._out_sampling_rate = None
        self._vocoder = None
        self._models = None
        self._voice_changer = None
        self._encode_stream = None
        self._convert_stream = None

    @property
    def ac_config(self):
        if self._ac_config is None:
            self._ac_config = create_config(self.acoustic_convert_config_path)
        return self._ac_config

    @property
    def sr_config(self):
        if self._sr_config is None:
            self._sr_config = create_sr_config(self.super_resolution_config_path)
        return self._sr_config

    @property
    def in_rate(self):
        if self._in_rate is None:
            self._in_rate = self.ac_config.dataset.acoustic_param.sampling_rate
        return self._in_rate

    @property
    def out_sampling_rate(self):
        if self._out_sampling_rate is None:
            self._out_sampling_rate = self.sr_config.dataset.param.voice_param.sample_rate
        return self._out_sampling_rate

    @property
    def vocoder(self):
        if self._vocoder is None:
            self._vocoder = Vocoder(
                acoustic_param=self.ac_config.dataset.acoustic_param,
                out_sampling_rate=self.out_sampling_rate,
                extract_f0_mode=VocodeMode.CREPE,
            )
        return self._vocoder

    @property
    def models(self):
        if self._models is None:
            f0_converter = F0Converter(
                input_statistics=self.input_statistics_path,
                target_statistics=self.target_statistics_path,
            )

            ac_config = self.ac_config
            sr_config = self.sr_config

            acoustic_converter = AcousticConverter(
                ac_config,
                self.acoustic_convert_model_path,
                f0_converter=f0_converter,
                out_sampling_rate=self.out_sampling_rate,
            )
            super_resolution = SuperResolution(
                sr_config,
                self.super_resolution_model_path,
            )
            self._models = acoustic_converter, super_resolution
        return self._models

    @property
    def voice_changer(self):
        if self._voice_changer is None:
            acoustic_converter, super_resolution = self.models
            self._voice_changer = VoiceChanger(
                acoustic_converter=acoustic_converter,
                super_resolution=super_resolution,
                output_sampling_rate=self.out_sampling_rate,
            )
        return self._voice_changer

    @property
    def encode_stream(self):
        if self._encode_stream is None:
            self._encode_stream = EncodeStream(vocoder=self.vocoder)
        return self._encode_stream

    @property
    def convert_stream(self):
        if self._convert_stream is None:
            self._convert_stream = ConvertStream(
                voice_changer=self.voice_changer,
            )
        return self._convert_stream

    def _load_wave_and_split(self, time_length: float = 1):
        rate = self.ac_config.dataset.acoustic_param.sampling_rate
        length = round(time_length * rate)
        wave, _ = librosa.load(Path('data/audioA.wav'), sr=rate)
        return [wave[i * length:(i + 1) * length] for i in range(len(wave) // length)]

    def _encode(self, w: numpy.ndarray):
        wave = Wave(wave=w, sampling_rate=self.in_rate)
        feature = self.vocoder.encode(wave)
        feature_wrapper = AcousticFeatureWrapper(wave=wave, **feature.__dict__)
        return feature_wrapper

    def _convert(self, feature_wrapper: AcousticFeatureWrapper):
        feature = self.voice_changer.convert_from_acoustic_feature(feature_wrapper)
        return feature

    def test_initialize(self):
        pass

    def test_load_model(self):
        acoustic_converter, super_resolution = self.models
        self.assertNotEqual(acoustic_converter, None)
        self.assertNotEqual(super_resolution, None)

    def test_load_wave(self):
        wave_segments = self._load_wave_and_split()
        self.assertEqual(len(wave_segments[0]), self.ac_config.dataset.acoustic_param.sampling_rate)

    def test_encode_stream(self):
        waves = self._load_wave_and_split()
        encode_stream = self.encode_stream

        encode_stream.add(start_time=0, data=waves[0])
        encode_stream.add(start_time=1, data=waves[1])

        # pick
        output = encode_stream.process(start_time=0, time_length=1, extra_time=0)
        target = self._encode(waves[0])
        self.assertEqual(output, target)

        # concat
        output = encode_stream.process(start_time=0.3, time_length=1, extra_time=0)
        target = self._encode(numpy.concatenate([
            waves[0][self.in_rate * 3 // 10:],
            waves[1][:self.in_rate * 3 // 10],
        ]))
        self.assertEqual(output, target)

        # pad
        output = encode_stream.process(start_time=1.3, time_length=1, extra_time=0)
        target = self._encode(numpy.concatenate([
            waves[1][self.in_rate * 3 // 10:],
            numpy.zeros(self.in_rate * 3 // 10),
        ]))
        self.assertEqual(output, target)

    def test_convert_stream(self):
        waves = self._load_wave_and_split()
        convert_stream = self.convert_stream

        convert_stream.add(start_time=0, data=self._encode(waves[0]))
        convert_stream.add(start_time=1, data=self._encode(waves[1]))

        # pick
        output = convert_stream.process(start_time=0, time_length=1, extra_time=0)
        target = self._convert(self._encode(waves[0]))
        self.assertTrue(equal_feature(output, target))

        # concat
        output = convert_stream.process(start_time=0.3, time_length=1, extra_time=0)
        target = self._convert(self._encode(numpy.concatenate([
            waves[0][self.in_rate * 3 // 10:],
            waves[1][:self.in_rate * 3 // 10],
        ])))
        print(output.f0[:5], output.f0[-5:])
        print(target.f0[:5], target.f0[-5:])
        self.assertTrue(equal_feature(output, target))
