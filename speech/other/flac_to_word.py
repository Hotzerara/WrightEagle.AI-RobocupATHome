import os.path
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import sys

from config import VOSK_en, FLAC_PATH

audio_path = os.path.join(FLAC_PATH,"61-70968-0000.flac")

# 检查文件是否存在
if not os.path.exists(audio_path):
    print(f"音频文件不存在: {audio_path}")
    sys.exit(1)

try:
    audio = AudioSegment.from_file(audio_path, format="flac")
    print(f"成功加载音频文件，时长: {len(audio)/1000.0} 秒")
except Exception as e:
    print(f"加载音频文件时出错: {e}")
    sys.exit(1)

# 设置单声道和采样率
audio = audio.set_channels(1).set_frame_rate(16000)
print(f"音频参数 - 采样率: {audio.frame_rate}, 声道数: {audio.channels}")

model = Model(VOSK_en)
rec = KaldiRecognizer(model, audio.frame_rate)

# 分块处理音频
results = []
for i in range(0, len(audio), 4000):
    chunk = audio[i:i+4000]
    data = chunk.raw_data
    if rec.AcceptWaveform(data):
        result = rec.Result()
        results.append(result)
        print(result)

# 打印最终结果
final_result = rec.FinalResult()
if final_result:
    results.append(final_result)
    print(final_result)

if not results:
    print("没有识别到任何内容，请检查音频文件是否包含语音或模型是否正确配置")
