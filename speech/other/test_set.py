import os
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import json
# 禁用Vosk的日志输出
from vosk import SetLogLevel
SetLogLevel(-1)
from config import *

def transcribe_audio(audio_path, model):
    """转录音频文件"""
    try:
        audio = AudioSegment.from_file(audio_path, format="flac")
        audio = audio.set_channels(1).set_frame_rate(16000)

        rec = KaldiRecognizer(model, audio.frame_rate)

        results = []
        for i in range(0, len(audio), 4000):
            chunk = audio[i:i+4000]
            data = chunk.raw_data
            if rec.AcceptWaveform(data):
                result = rec.Result()
                results.append(result)

        final_result = rec.FinalResult()
        if final_result:
            results.append(final_result)

        # 解析JSON结果并提取文本
        full_text = ""
        for result_str in results:
            result_json = json.loads(result_str)
            if 'text' in result_json:
                full_text += result_json['text'] + " "

        return full_text.strip()
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {e}")
        return ""

def load_transcriptions(trans_file):
    """加载转录文件"""
    transcriptions = {}
    with open(trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id = parts[0]
                text = parts[1]
                transcriptions[file_id] = text
    return transcriptions

def calculate_wer(reference, hypothesis):
    """计算词错误率 (WER)"""
    # 简单的单词分割和比较
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # 这里只是一个简单的实现，实际WER计算更复杂
    if len(ref_words) == 0:
        return 0 if len(hyp_words) == 0 else 100

    # 计算匹配度（简化版）
    matches = sum(1 for rw, hw in zip(ref_words, hyp_words) if rw == hw)
    accuracy = matches / len(ref_words) * 100 if len(ref_words) > 0 else 0
    return 100 - accuracy

def batch_test(audio_dir, trans_file):
    """批量测试函数"""
    # 加载参考转录
    reference_transcriptions = load_transcriptions(trans_file)

    # 只加载一次模型
    print("正在加载语音识别模型...")
    model = Model(VOSK_en)
    print("模型加载完成")
    total_files = 0
    total_accuracy = 0

    # 遍历所有音频文件
    for file_id in reference_transcriptions.keys():
        audio_filename = f"{file_id}.flac"
        audio_path = os.path.join(audio_dir, audio_filename)

        if os.path.exists(audio_path):
            print(f"正在处理: {audio_filename}")

            # 转录音频
            recognized_text = transcribe_audio(audio_path, model)
            reference_text = reference_transcriptions[file_id]

            # 计算准确率
            wer = calculate_wer(reference_text, recognized_text)
            accuracy = 100 - wer

            print(f"文件: {file_id}")
            print(f"参考文本: {reference_text}")
            print(f"识别文本: {recognized_text}")
            print(f"准确率: {accuracy:.2f}%")
            print("-" * 50)

            total_files += 1
            total_accuracy += accuracy
        else:
            print(f"找不到音频文件: {audio_path}")

    if total_files > 0:
        average_accuracy = total_accuracy / total_files
        print(f"\n平均准确率: {average_accuracy:.2f}% (基于 {total_files} 个文件)")
    else:
        print("没有找到可处理的文件")

# 使用示例
if __name__ == "__main__":
    # 音频文件和转录文件在同一目录下
    audio_directory = TEST_PATH
    transcription_file = TEST_FILE_PATH

    batch_test(audio_directory, transcription_file)
