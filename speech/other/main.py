"""
调用speech_to_word.py把语音转文本
并调用SparkPythondemo.py让模型解释文本
"""
import json

import speech_to_word
from api import SparkPythondemo, SparkApi
from api.SparkPythondemo import checklen, getText, appid, api_key, api_secret, Spark_url, domain


def convert_speech_to_text():
    """
    简单的语音转文字函数，捕获5秒的语音输入并转换为文本
    :return: 识别到的文本
    """
    import time

    # 创建语音识别器实例
    recognizer_inner = speech_to_word.SpeechRecognizer()

    result_text = []

    # 设置回调函数来收集识别结果
    def collect_result(text):
        if text.strip():
            result_text.append(text)

    recognizer_inner.set_result_callback(collect_result)

    # 开始录音5秒钟
    print("请说话... (5秒内)")
    start_time = time.time()

    try:
        while time.time() - start_time < 5:
            # 从麦克风获取音频数据
            data = recognizer_inner.stream.read(4000, exception_on_overflow=False)

            # 如果没有数据，跳过
            if len(data) == 0:
                continue

            # 进行语音识别
            if recognizer_inner.rec.AcceptWaveform(data):
                # 获取识别结果
                result = recognizer_inner.rec.Result()
                result_json = json.loads(result)
                text = result_json.get('text', '')
                if text.strip():
                    print(f"识别结果: {text}")
                    collect_result(text)

    except Exception as e:
        print(f"录音过程中出错: {e}")
    finally:
        recognizer_inner.cleanup()

    # 返回识别到的文本
    return ' '.join(result_text)

def main():
    while True:
        # 调用语音转文字功能
        text = convert_speech_to_text()

        # 如果成功获取到文本，则调用模型进行解释
        if text:
            question = checklen(getText("user", text))
            SparkApi.answer = ""
            print("星火:", end="")
            SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
            # print(SparkApi.answer)
            getText("assistant", SparkApi.answer)
        else:
            print("未能识别到有效语音")

if __name__ == "__main__":
    main()
