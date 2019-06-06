import os
from pathlib import Path
from typing import Tuple
from unittest import TestCase

import librosa
import numpy
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter, Wave, AcousticFeature
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter

from realtime_voice_conversion.config import VocodeMode
from realtime_voice_conversion.stream import EncodeStream, ConvertStream, DecodeStream
from realtime_voice_conversion.stream.base_stream import BaseStream
from realtime_voice_conversion.yukarin_wrapper.vocoder import Vocoder, RealtimeVocoder
from realtime_voice_conversion.yukarin_wrapper.voice_changer import AcousticFeatureWrapper, VoiceChanger


def equal_feature(a: AcousticFeature, b: AcousticFeature):
    return numpy.all(a.f0 == b.f0)


class AllStreamTest(TestCase):
    def setUp(self):
        self.input_statistics_path = os.getenv('INPUT_STATISTICS')
        self.target_statistics_path = os.getenv('TARGET_STATISTICS')
        self.stage1_model_path = os.getenv('ACOUSTIC_CONVERT_MODEL')
        self.stage1_config_path = os.getenv('ACOUSTIC_CONVERT_CONFIG')
        self.stage2_model_path = os.getenv('SUPER_RESOLUTION_MODEL')
        self.stage2_config_path = os.getenv('SUPER_RESOLUTION_CONFIG')

        if self.input_statistics_path is None: raise ValueError('INPUT_STATISTICS is not found.')
        if self.target_statistics_path is None: raise ValueError('TARGET_STATISTICS is not found.')
        if self.stage1_model_path is None: raise ValueError('ACOUSTIC_CONVERT_MODEL is not found.')
        if self.stage1_config_path is None: raise ValueError('ACOUSTIC_CONVERT_CONFIG is not found.')
        if self.stage2_model_path is None: raise ValueError('SUPER_RESOLUTION_MODEL is not found.')
        if self.stage2_config_path is None: raise ValueError('SUPER_RESOLUTION_CONFIG is not found.')

        self._ac_config = None
        self._sr_config = None
        self._input_rate = None
        self._out_sampling_rate = None
        self._vocoder = None
        self._realtime_vocoder = None
        self._models = None
        self._voice_changer = None
        self._encode_stream = None
        self._convert_stream = None
        self._decode_stream = None

    @property
    def ac_config(self):
        if self._ac_config is None:
            self._ac_config = create_config(self.stage1_config_path)
        return self._ac_config

    @property
    def sr_config(self):
        if self._sr_config is None:
            self._sr_config = create_sr_config(self.stage2_config_path)
        return self._sr_config

    @property
    def input_rate(self):
        if self._input_rate is None:
            self._input_rate = self.ac_config.dataset.acoustic_param.sampling_rate
        return self._input_rate

    @property
    def out_sampling_rate(self):
        if self._out_sampling_rate is None:
            self._out_sampling_rate = self.sr_config.dataset.param.voice_param.sample_rate
        return self._out_sampling_rate

    @property
    def realtime_vocoder(self):
        if self._realtime_vocoder is None:
            self._realtime_vocoder = RealtimeVocoder(
                acoustic_param=self.ac_config.dataset.acoustic_param,
                out_sampling_rate=self.out_sampling_rate,
                extract_f0_mode=VocodeMode.WORLD,
            )
            self._realtime_vocoder.create_synthesizer(
                buffer_size=1024,
                number_of_pointers=16,
            )
        return self._realtime_vocoder

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
                self.stage1_model_path,
                f0_converter=f0_converter,
                out_sampling_rate=self.out_sampling_rate,
            )
            super_resolution = SuperResolution(
                sr_config,
                self.stage2_model_path,
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
            self._encode_stream = EncodeStream(vocoder=self.realtime_vocoder)
        return self._encode_stream

    @property
    def convert_stream(self):
        if self._convert_stream is None:
            self._convert_stream = ConvertStream(
                voice_changer=self.voice_changer,
            )
        return self._convert_stream

    @property
    def decode_stream(self):
        if self._decode_stream is None:
            self._decode_stream = DecodeStream(
                vocoder=self.realtime_vocoder,
            )
        return self._decode_stream

    def _load_wave_and_split(self, time_length: float = 1):
        rate = self.ac_config.dataset.acoustic_param.sampling_rate
        length = round(time_length * rate)
        wave, _ = librosa.load(Path('tests/data/audioA.wav'), sr=rate)
        return [wave[i * length:(i + 1) * length] for i in range(len(wave) // length)]

    def _encode(self, w: numpy.ndarray):
        wave = Wave(wave=w, sampling_rate=self.input_rate)
        feature_wrapper = self.realtime_vocoder.encode(wave)
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
            waves[0][self.input_rate * 3 // 10:],
            waves[1][:self.input_rate * 3 // 10],
        ]))
        self.assertEqual(output, target)

        # pad
        output = encode_stream.process(start_time=1.3, time_length=1, extra_time=0)
        target = self._encode(numpy.concatenate([
            waves[1][self.input_rate * 3 // 10:],
            numpy.zeros(self.input_rate * 3 // 10),
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

    def test_all_stream(self):
        num_data = 10
        time_length = 0.3

        def _add(_stream: BaseStream, _datas):
            for i, data in zip(range(num_data), _datas):
                _stream.add(start_time=i * time_length, data=data)

        def _split_process(_stream: BaseStream, _extra_time: float):
            return [
                _stream.process(start_time=i * time_length, time_length=time_length, extra_time=_extra_time)
                for i in range(num_data)
            ]

        def _join_process(_stream: BaseStream, _extra_time: float):
            return _stream.process(start_time=0, time_length=time_length * num_data, extra_time=_extra_time)

        def _process_all_stream(
                _streams: Tuple[BaseStream, BaseStream, BaseStream],
                _datas,
                _split_flags: Tuple[bool, bool, bool],
                _extra_times: Tuple[float, float, float],
        ):
            for stream, split_flag, extra_time in zip(_streams, _split_flags, _extra_times):
                _add(stream, _datas)
                if split_flag:
                    _datas = _split_process(stream, _extra_time=extra_time)
                else:
                    _datas = [_join_process(stream, _extra_time=extra_time)]
            return _datas

        def _concat_and_save(_waves, _path: str):
            wave = numpy.concatenate(_waves).astype(numpy.float32)
            librosa.output.write_wav(_path, wave, self.out_sampling_rate)

        def _remove(_streams: Tuple[BaseStream, BaseStream, BaseStream]):
            for stream in _streams:
                stream.remove(end_time=num_data)

        waves = self._load_wave_and_split(time_length=time_length)[:num_data]
        encode_stream = self.encode_stream
        convert_stream = self.convert_stream
        decode_stream = self.decode_stream

        streams = (encode_stream, convert_stream, decode_stream)

        # datas = _process_all_stream(streams, waves, _split_flags=(True, True, True), _extra_times=(0, 0, 0))
        # _concat_and_save(datas, '../test_all_split.wav')
        # _remove(streams)
        #
        # datas= _process_all_stream(streams, waves, _split_flags=(False, True, True), _extra_times=(0, 0, 0))
        # _concat_and_save(datas, '../test_encode_join.wav')
        # _remove(streams)
        #
        # datas = _process_all_stream(streams, waves, _split_flags=(True, False, True), _extra_times=(0, 0, 0))
        # _concat_and_save(datas, '../test_convert_join.wav')
        # _remove(streams)
        #
        datas = _process_all_stream(streams, waves, _split_flags=(True, True, True), _extra_times=(0, 1, 0))
        _concat_and_save(datas, '../test_convert_extra05.wav')
        _remove(streams)

        # datas = _process_all_stream(streams, waves, _split_flags=(True, True, False), _extra_times=(0, 0, 0))
        # _concat_and_save(datas, '../test_decode_join.wav')
        # _remove(streams)
        #
        # datas = _process_all_stream(streams, waves, _split_flags=(False, False, True), _extra_times=(0, 0, 0))
        # _concat_and_save(datas, '../test_encode_convert_join.wav')
        # _remove(streams)
        #
        # datas = _process_all_stream(streams, waves, _split_flags=(False, False, False), _extra_times=(0, 0, 0))
        # _concat_and_save(datas, '../test_all_join.wav')
        # _remove(streams)
