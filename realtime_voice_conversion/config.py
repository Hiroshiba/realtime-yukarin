from pathlib import Path
from typing import NamedTuple, Dict, Any

import yaml


class Config(NamedTuple):
    in_rate: int
    out_rate: int
    frame_period: float
    buffer_time: float
    vocoder_buffer_size: int
    in_norm: float
    out_norm: float
    input_silent_threshold: float
    silent_threshold: float
    encode_extra_time: float
    convert_extra_time: float
    decode_extra_time: float

    f0_converter_input_statistics_path: Path
    f0_converter_target_statistics_path: Path
    acoustic_converter_model_path: Path
    acoustic_converter_config_path: Path
    super_resolution_model_path: Path
    super_resolution_config_path: Path

    @property
    def in_audio_chunk(self):
        return round(self.in_rate * self.buffer_time)

    @property
    def out_audio_chunk(self):
        return round(self.out_rate * self.buffer_time)

    @staticmethod
    def from_yaml(path: Path):
        d: Dict[str, Any] = yaml.safe_load(path.open())
        return Config(
            in_rate=d['in_rate'],
            out_rate=d['out_rate'],
            frame_period=d['frame_period'],
            buffer_time=d['buffer_time'],
            vocoder_buffer_size=d['vocoder_buffer_size'],
            in_norm=d['in_norm'],
            out_norm=d['out_norm'],
            input_silent_threshold=d['input_silent_threshold'],
            silent_threshold=d['silent_threshold'],
            encode_extra_time=d['encode_extra_time'],
            convert_extra_time=d['convert_extra_time'],
            decode_extra_time=d['decode_extra_time'],

            f0_converter_input_statistics_path=Path(d['f0_converter_input_statistics_path']),
            f0_converter_target_statistics_path=Path(d['f0_converter_target_statistics_path']),
            acoustic_converter_model_path=Path(d['acoustic_converter_model_path']),
            acoustic_converter_config_path=Path(d['acoustic_converter_config_path']),
            super_resolution_model_path=Path(d['super_resolution_model_path']),
            super_resolution_config_path=Path(d['super_resolution_config_path']),
        )
