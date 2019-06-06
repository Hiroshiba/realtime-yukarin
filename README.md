# realtime-yukarin
ディープラーニング声質変換を使って、リアルタイムで声質変換する。

## 推奨環境
* Windows
* GeForce GTX 1060
* 6GB 以上の GPU メモリ
* Intel Core i7-7700 CPU @ 3.60GHz
* Python 3.6.x

## 準備
### 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

### 学習済みモデルの準備
実行するために必要なデータは、[yukarin](https://github.com/Hiroshiba/yukarin)と[become-yukarin](https://github.com/Hiroshiba/become-yukarin)の学習済みモデルと設定ファイル、
yukarinで得られる、入力・目標音声の周波数の統計量のファイルです。

ここでは、次のようにファイル名を設定したとします。

|  説明  |  ファイル名  |
| ---- | ---- |
|  入力音声の周波数の統計量のファイル  |  `./sample/input_statistics.npy`  |
|  目標音声の周波数の統計量のファイル  |  `./sample/tareget_statistics.npy`  |
|  yukarin用のモデル  |  `./sample/model_stage1/predictor.npz`  |
|  yukarin用の設定ファイル  |  `./sample/model_stage1/config.json`  |
|  become-yukarin用のモデル  |  `./sample/model_stage2/predictor.npz`  |
|  become-yukarin用の設定ファイル  |  `./sample/model_stage2/config.json`  |

## 確認
`./check.py`を実行して、用意したファイルで正しく動くか確認します。
次の例では、`input.wav`の音声データ5秒を声質変換し、音声データを`output.wav`に出力します。

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

動かない場合は[GithubのIssue](https://github.com/Hiroshiba/realtime-yukarin/issues)で質問することができます。

## 実行
リアルタイム声質変換を実行するには、設定ファイル`config.yaml`を作成し、`./run.py`を実行します。

```bash
python run.py ./config.yml
```

### 設定ファイルの説明
```yaml
# 入力サウンドデバイスの名前。部分一致。詳細は下記
input_device_name: str

# 出力サウンドデバイスの名前。部分一致。詳細は下記
output_device_name: str

# 入力音声サンプリングレート
input_rate: int

# 出力音声サンプリングレート
output_rate: int

# 音響特徴量のframe_period
frame_period: int

# 一度に変換する音声の長さ（秒）。長すぎると遅延が増え、短すぎると処理が追いつかない
buffer_time: float

or crepe  # 基本周波数を求める手法。CREPEは別途ライブラリが必要、詳細はrequirements.txt
extract_f0_mode: worl

# 一度に合成する音声の長さ（サンプル数）
vocoder_buffer_size: int

# 入力音声データの振幅スケーリング。1以上だと音を大きくし、1未満だと小さくする
input_scale: float

# 出力音声データの振幅スケーリング。1以上だと音を大きくし、1未満だと小さくする
output_scale: float

# 入力音声データが無音だとみなされる閾値（db）。小さいほど無音に判定されやすい
input_silent_threshold: float

# 出力音声データが無音だとみなされる閾値（db）。小さいほど無音に判定されやすい
output_silent_threshold: float

# エンコード時の前後オーバーラップ（秒）
encode_extra_time: float

# コンバート時の前後オーバーラップ（秒）
convert_extra_time: float

# デコード時の前後オーバーラップ（秒）
decode_extra_time: float

# 学習済みモデルのファイル
input_statistics_path: str
target_statistics_path: str
stage1_model_path: str
stage1_config_path: str
stage2_model_path: str
stage2_config_path: str
```

#### サウンドデバイスの名前
下の例だと、`Logicool Speaker`がサウンドデバイスの名前。

<img src='https://user-images.githubusercontent.com/4987327/59046047-2eaf9980-88bc-11e9-8732-0a7d80ef2d2e.png'>

## License
[MIT License](./LICENSE)
