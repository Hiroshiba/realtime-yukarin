# Realtime Yukarin: an application for real-time voice conversion
Realtime Yukarin is the application for real-time voice conversion with a single command.
This application needs trained deep learning models and a GPU computer.
The source code is an OSS and MIT license.
So you can modify this code, or use it for your applications whether commercial or non-commercial.

[Japanese README](./README_jp.md)

## Supported environment
* Windows
* GeForce GTX 1060
* 6GB GPU memory
* Intel Core i7-7700 CPU @ 3.60GHz
* Python 3.6

## Preparation
### Installation required libraries
```bash
pip install -r requirements.txt
```

### Prepare trained models
You need two trained models, a first stage model responsible for voice conversion
and a second stage model for enhancing the quality of the converted results.
You can create a first stage model with [Yukarin](https://github.com/Hiroshiba/yukarin)
and a second stage model with [Become Yukarin](https://github.com/Hiroshiba/become-yukarin).

Also, for voice pitch conversion, you need a file of frequency statistics
at [Yukarin](https://github.com/Hiroshiba/yukarin).

Here, each filename is as follows:

|  Content  |  Filename  |
| ---- | ---- |
|  Frequency statistics for input voice  |  `./sample/input_statistics.npy`  |
|  Frequency statistics for target voice  |  `./sample/tareget_statistics.npy`  |
|  First stage model from [Yukarin](https://github.com/Hiroshiba/yukarin)  |  `./sample/model_stage1/predictor.npz`  |
|  First stage's config file  |  `./sample/model_stage1/config.json`  |
|  Second stage model from [Become Yukarin](https://github.com/Hiroshiba/become-yukarin) |  `./sample/model_stage2/predictor.npz`  |
|  Second stage's config file  |  `./sample/model_stage2/config.json`  |

## Verification
You can verify prepared files with executing `./check.py`.
The following example converts 5 seconds voice data of `input.wav`, and save to `output.wav`.

```bash
python check.py \
    --input_path 'input.wav' \
    --input_time_length 5 \
    --output_path 'output.wav' \
    --input_statistics_path './sample/model_stage1/predictor.npz' \
    --target_statistics_path './sample/model_stage1/config.json' \
    --stage1_model_path './sample/model_stage2/predictor.npz' \
    --stage1_config_path './sample/model_stage2/config.json' \
    --stage2_model_path './sample/input_statistics.npy' \
    --stage2_config_path './sample/tareget_statistics.npy' \

```

If you have problems, you can ask questions
on [Github Issue](https://github.com/Hiroshiba/realtime-yukarin/issues).

## Run
To perform real-time voice conversion, create a config file `config.yaml` and run `./run.py`.

```bash
python run.py ./config.yaml
```

### Description of config file
```yaml
# Name of input sound device. Partial Match. Details are below.
input_device_name: str

# Name of output sound device. Partial Match. Details are below.
output_device_name: str

# Input sampling rate
input_rate: int

# Output sampling rate
output_rate: int

# frame_period for Acoustic feature
frame_period: int

# Length of voice to convert at one time (seconds).
# If it is too long, delay will increase, and if it is too short, processing will not catch up.
buffer_time: float

# Method to calclate the fundamental frequency. world ofr crepe.
# CREPE needs additional libraries, details are requirements.txt
extract_f0_mode: world

# Length of voice to be synthesized at one time (number of samples)
vocoder_buffer_size: int

# Amplitude scaling for input.
# When it is more than 1, the amplitude becomes large, and when it is less than 1, the amplitude becomes small.
input_scale: float

# Amplitude scaling for output.
# When it is more than 1, the amplitude becomes large, and when it is less than 1, the amplitude becomes small.
output_scale: float

# Silence threshold for input (db).
# The smaller the value, the easier it is to silence.
input_silent_threshold: float

# Silence threshold for output (db).
# The smaller the value, the easier it is to silence.
output_silent_threshold: float

# Overlap for encoding (seconds)
encode_extra_time: float

# Overlap for converting (seconds)
convert_extra_time: float

# Overlap for decoding (seconds)
decode_extra_time: float

# Path of frequency statistics file
input_statistics_path: str
target_statistics_path: str

# Path of trained model file
stage1_model_path: str
stage1_config_path: str
stage2_model_path: str
stage2_config_path: str
```

#### (preliminary knowledge) Name of sound device
In the example below, `Logitech Speaker` is the name of the sound device.
<img src='https://user-images.githubusercontent.com/4987327/59046047-2eaf9980-88bc-11e9-8732-0a7d80ef2d2e.png'>

## License
[MIT License](./LICENSE)
