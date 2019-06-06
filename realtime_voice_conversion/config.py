from enum import Enum
from pathlib import Path
from typing import NamedTuple, Dict, Any

import yaml


class VocodeMode(Enum):
    WORLD = 'world'
    CREPE = 'crepe'


class Config(NamedTuple):
    input_device_name: str
    output_device_name: str
    input_rate: int
    output_rate: int
    frame_period: float
    buffer_time: float
    extract_f0_mode: VocodeMode
    vocoder_buffer_size: int
    input_scale: float
    output_scale: float
    input_silent_threshold: float
    output_silent_threshold: float
    encode_extra_time: float
    convert_extra_time: float
    decode_extra_time: float

    input_statistics_path: Path
    target_statistics_path: Path
    stage1_model_path: Path
    stage1_config_path: Path
    stage2_model_path: Path
    stage2_config_path: Path

    @property
    def in_audio_chunk(self):
        return round(self.input_rate * self.buffer_time)

    @property
    def out_audio_chunk(self):
        return round(self.output_rate * self.buffer_time)

    @staticmethod
    def from_yaml(path: Path):
        d: Dict[str, Any] = yaml.safe_load(path.open())
        return Config(
            input_device_name=d['input_device_name'],
            output_device_name=d['output_device_name'],
            input_rate=d['input_rate'],
            output_rate=d['output_rate'],
            frame_period=d['frame_period'],
            buffer_time=d['buffer_time'],
            extract_f0_mode=VocodeMode(d['extract_f0_mode']),
            vocoder_buffer_size=d['vocoder_buffer_size'],
            input_scale=d['input_scale'],
            output_scale=d['output_scale'],
            input_silent_threshold=d['input_silent_threshold'],
            output_silent_threshold=d['output_silent_threshold'],
            encode_extra_time=d['encode_extra_time'],
            convert_extra_time=d['convert_extra_time'],
            decode_extra_time=d['decode_extra_time'],

            input_statistics_path=Path(d['input_statistics_path']),
            target_statistics_path=Path(d['target_statistics_path']),
            stage1_model_path=Path(d['stage1_model_path']),
            stage1_config_path=Path(d['stage1_config_path']),
            stage2_model_path=Path(d['stage2_model_path']),
            stage2_config_path=Path(d['stage2_config_path']),
        )
