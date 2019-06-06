import argparse
from pathlib import Path
from typing import Tuple

import librosa
import numpy
from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter

from realtime_voice_conversion.config import VocodeMode
from realtime_voice_conversion.stream import EncodeStream, ConvertStream, DecodeStream
from realtime_voice_conversion.stream.base_stream import BaseStream
from realtime_voice_conversion.yukarin_wrapper.vocoder import RealtimeVocoder
from realtime_voice_conversion.yukarin_wrapper.voice_changer import VoiceChanger


def check(
        input_path: Path,
        input_time_length: int,
        output_path: Path,
        input_statistics_path: Path,
        target_statistics_path: Path,
        stage1_model_path: Path,
        stage1_config_path: Path,
        stage2_model_path: Path,
        stage2_config_path: Path,
):
    ac_config = create_config(stage1_config_path)
    sr_config = create_sr_config(stage2_config_path)
    input_rate = ac_config.dataset.acoustic_param.sampling_rate
    output_rate = sr_config.dataset.param.voice_param.sample_rate

    realtime_vocoder = RealtimeVocoder(
        acoustic_param=ac_config.dataset.acoustic_param,
        out_sampling_rate=output_rate,
        extract_f0_mode=VocodeMode.WORLD,
    )
    realtime_vocoder.create_synthesizer(
        buffer_size=1024,
        number_of_pointers=16,
    )

    f0_converter = F0Converter(
        input_statistics=input_statistics_path,
        target_statistics=target_statistics_path,
    )

    ac_config = ac_config
    sr_config = sr_config

    acoustic_converter = AcousticConverter(
        ac_config,
        stage1_model_path,
        f0_converter=f0_converter,
        out_sampling_rate=output_rate,
    )
    super_resolution = SuperResolution(
        sr_config,
        stage2_model_path,
    )

    voice_changer = VoiceChanger(
        acoustic_converter=acoustic_converter,
        super_resolution=super_resolution,
        output_sampling_rate=output_rate,
    )

    encode_stream = EncodeStream(vocoder=realtime_vocoder)
    convert_stream = ConvertStream(voice_changer=voice_changer)
    decode_stream = DecodeStream(vocoder=realtime_vocoder)

    num_data = input_time_length
    time_length = 1

    def _load_wave_and_split(time_length: float = 1):
        length = round(time_length * input_rate)
        wave, _ = librosa.load(str(input_path), sr=input_rate)
        return [wave[i * length:(i + 1) * length] for i in range(len(wave) // length)]

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

    def _concat_and_save(_waves, _path: Path):
        wave = numpy.concatenate(_waves).astype(numpy.float32)
        librosa.output.write_wav(str(_path), wave, output_rate)

    def _remove(_streams: Tuple[BaseStream, BaseStream, BaseStream]):
        for stream in _streams:
            stream.remove(end_time=num_data)

    waves = _load_wave_and_split(time_length=time_length)[:num_data]
    encode_stream = encode_stream
    convert_stream = convert_stream
    decode_stream = decode_stream

    streams = (encode_stream, convert_stream, decode_stream)

    datas = _process_all_stream(streams, waves, _split_flags=(True, True, True), _extra_times=(0, 1, 0))
    _concat_and_save(datas, output_path)
    _remove(streams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=Path)
    parser.add_argument('--input_time_length', type=int)
    parser.add_argument('--output_path', type=Path, default=Path('output.wav'))
    parser.add_argument('--input_statistics_path', type=Path, default=Path('./sample/model_stage1/predictor.npz'))
    parser.add_argument('--target_statistics_path', type=Path, default=Path('./sample/model_stage1/config.json'))
    parser.add_argument('--stage1_model_path', type=Path, default=Path('./sample/model_stage2/predictor.npz'))
    parser.add_argument('--stage1_config_path', type=Path, default=Path('./sample/model_stage2/config.json'))
    parser.add_argument('--stage2_model_path', type=Path, default=Path('./sample/input_statistics.npy'))
    parser.add_argument('--stage2_config_path', type=Path, default=Path('./sample/tareget_statistics.npy'))
    args = parser.parse_args()

    check(
        input_path=args.input_path,
        input_time_length=args.input_time_length,
        output_path=args.output_path,
        input_statistics_path=args.input_statistics_path,
        target_statistics_path=args.target_statistics_path,
        stage1_model_path=args.stage1_model_path,
        stage1_config_path=args.stage1_config_path,
        stage2_model_path=args.stage2_model_path,
        stage2_config_path=args.stage2_config_path,
    )
