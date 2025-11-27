import sherpa_onnx
import sounddevice as sd
import sys


def TTS(text):
    # 配置模型路径 (你需要先下载这些文件)
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                # model="./models/vits-zh-aishell3/vits-aishell3.onnx",
                # tokens="./models/vits-zh-aishell3/tokens.txt",
                # lexicon="./models/vits-zh-aishell3/lexicon.txt",# 这个文件只有中文模型需要

                model="./models/vits-ljs/vits-ljs.onnx",
                tokens="./models/vits-ljs/tokens.txt",
                lexicon="./models/vits-ljs/lexicon.txt",
                # 注意：没有 lexicon！

                # model="./models/vits-vctk/vits-vctk.onnx",
                # tokens="./models/vits-vctk/tokens.txt",
                # # 同样不需要 lexicon
            )
        )
    )

    tts = sherpa_onnx.OfflineTts(tts_config)

    # 生成音频, sid是模型中的说话人ID，默认为0
    # vits-vctk这个模型sid可以从0到107
    # vits-zh-aishell3模型的sid可以从0到217
    # vits-ljs这个模型只有一个说话人, sid=0
    # audio = tts.generate("机器人已就绪，随时准备出发。", sid=0, speed=1.0)
    audio = tts.generate(text, sid=0, speed=1.0)

    # 播放音频 (使用 sounddevice)
    sd.play(audio.samples, audio.sample_rate)
    sd.wait()


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python tts_script.py \"要转换的文字\"")
        print("示例: python tts_script.py \"Hello, how are you?\"")
        sys.exit(1)
    
    # 获取命令行参数中的文字
    text = sys.argv[1]
    print(f"正在转换文字: {text}")
    TTS(text)