import logging
from pathlib import Path

from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter

from realtime_voice_conversion.worker.utility import init_logger


class YukarinConverter(object):
    def __init__(
            self,
            acoustic_converter: AcousticConverter,
            super_resolution: SuperResolution,
    ):
        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution

    @staticmethod
    def make_yukarin_converter(
            f0_converter_input_statistics_path: Path,
            f0_converter_target_statistics_path: Path,
            acoustic_converter_config_path: Path,
            acoustic_converter_model_path: Path,
            super_resolution_config_path: Path,
            super_resolution_model_path: Path,
    ):
        logger = logging.getLogger('encode')
        init_logger(logger)
        logger.info('make_yukarin_converter')

        f0_converter = F0Converter(
            input_statistics=f0_converter_input_statistics_path,
            target_statistics=f0_converter_target_statistics_path,
        )

        config = create_config(acoustic_converter_config_path)
        acoustic_converter = AcousticConverter(
            config=config,
            model_path=acoustic_converter_model_path,
            gpu=0,
            f0_converter=f0_converter,
            out_sampling_rate=24000,
        )
        logger.info('model 1 loaded!')

        sr_config = create_sr_config(super_resolution_config_path)
        super_resolution = SuperResolution(
            config=sr_config,
            model_path=super_resolution_model_path,
            gpu=0,
        )
        logger.info('model 2 loaded!')
        return YukarinConverter(
            acoustic_converter=acoustic_converter,
            super_resolution=super_resolution,
        )
