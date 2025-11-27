import wave
import numpy as np
import pyaudio
import samplerate
from faster_whisper import WhisperModel
import torch
import torchaudio
from silero_vad import load_silero_vad, VADIterator
import collections

from config import WHISPER_MEDIUM, WHISPER_LARGE


# ============= 查找指定麦克风 =============
def find_input_device(name_part: str):
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if name_part.lower() in info["name"].lower() and info["maxInputChannels"] > 0:
            print(f"✓ 找到麦克风: {info['name']} (index={i})")
            rate = int(info.get("defaultSampleRate", 48000))
            return i, rate
    print("⚠ 未找到指定麦克风，使用默认设备")
    index = p.get_default_input_device_info()["index"]
    info = p.get_device_info_by_index(index)
    return index, int(info.get("defaultSampleRate", 48000))


class SpeechRecognizer:
    def __init__(self, mic_name="Newmine"):
        self.model_path = WHISPER_MEDIUM

        print("加载 Whisper 模型中...")
        self.model = WhisperModel(
            self.model_path,
            device="cpu",
            compute_type="int8"
        )
        print("✓ Whisper 模型加载完成！")

        # 加载 Silero VAD 模型
        print("加载 Silero VAD 模型中...")
        self.vad_model = load_silero_vad()
        print("✓ Silero VAD 模型加载完成！")

        # Whisper 固定 16k 采样率
        self.target_rate = 16000

        # ===== 查找麦克风 =====
        self.p = pyaudio.PyAudio()
        self.device_index, self.device_rate = find_input_device(mic_name)
        print(f"🎤 麦克风采样率: {self.device_rate} Hz")

        # 每帧 32ms 保证在16k 采样下每帧有512个采样点满足vad模型的需求
        self.frame_duration_ms = 32
        # 设备侧每帧采样点数
        self.frame_samples_device = int(self.device_rate * self.frame_duration_ms / 1000)
        # 16k 侧每帧采样点数
        self.frame_samples_16k = int(self.target_rate * self.frame_duration_ms / 1000)

        # ===== 打开麦克风流 =====
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.device_rate,
            input=True,
            frames_per_buffer=self.frame_samples_device,
            input_device_index=self.device_index
        )

        # 录音保存（可选）
        self.wav_file = wave.open("recorded_audio.wav", "wb")
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(self.device_rate)

        # ===== 语音缓冲区（16k）=====
        self.buffer_16k = np.array([], dtype=np.float32)

        # VAD Iterator 用于流式检测
        self.vad_iterator = VADIterator(self.vad_model, threshold=0.5)

        # 状态跟踪
        self.is_speaking = False

        # 启动时简单检测环境噪声
        self.noise_floor = self.detect_noise_floor()
        print(f"环境噪声能量估计: {self.noise_floor:.6f}")

    # ============= 辅助函数 =============
    @staticmethod
    def rms(audio):
        """计算短时能量"""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def detect_noise_floor(self):
        """启动时采几帧估计噪声能量"""
        print("正在检测环境噪声（请暂时不要说话）...")
        samples = []
        num_frames = 10  # 10 * 30ms = 300ms
        for _ in range(num_frames):
            data = self.stream.read(self.frame_samples_device, exception_on_overflow=False)
            # 顺便写进 wav，避免丢头几帧
            self.wav_file.writeframes(data)

            audio_dev = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_16k = samplerate.resample(
                audio_dev,
                self.target_rate / self.device_rate,
                "sinc_best"
            )
            samples.append(self.rms(audio_16k))

        floor = float(np.mean(samples)) if samples else 0.0
        return floor

    # ============= 主循环 =============
    def start_listening(self):
        print("开始实时语音识别 (Silero VAD)...\n")

        try:
            while True:
                # 按帧读取
                data = self.stream.read(self.frame_samples_device, exception_on_overflow=False)
                # 保存到 wav
                self.wav_file.writeframes(data)

                # 转成 float32 [-1, 1]（设备采样率）
                audio_dev = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # 重采样到 16k，用于 VAD + Whisper
                audio_16k = samplerate.resample(
                    audio_dev,
                    self.target_rate / self.device_rate,
                    "sinc_best"
                )

                # 确保长度是我们期望的帧大小
                if len(audio_16k) > self.frame_samples_16k:
                    audio_16k = audio_16k[:self.frame_samples_16k]
                elif len(audio_16k) < self.frame_samples_16k:
                    pad_len = self.frame_samples_16k - len(audio_16k)
                    audio_16k = np.concatenate(
                        [audio_16k, np.zeros(pad_len, dtype=np.float32)]
                    )

                # 使用 Silero VAD 检测
                speech_dict = self.vad_iterator(audio_16k, return_seconds=False)

                if speech_dict is not None:
                    if 'start' in speech_dict:
                        # 检测到语音开始
                        print("开始说话...")
                        self.is_speaking = True
                        self.buffer_16k = np.array([], dtype=np.float32)  # 重置缓冲区

                    if 'end' in speech_dict:
                        # 检测到语音结束
                        print("说话结束，开始识别...")
                        self.is_speaking = False
                        if len(self.buffer_16k) > int(self.target_rate * 0.1):  # 至少0.1秒
                            self.transcribe_buffer()

                # 如果正在说话中，将音频加入缓冲区
                if self.is_speaking:
                    self.buffer_16k = np.concatenate([self.buffer_16k, audio_16k])

        except KeyboardInterrupt:
            print("识别已停止")
            # 停止前把最后一段说完的话也识别一下
            if self.is_speaking and len(self.buffer_16k) > int(self.target_rate * 0.1):
                self.transcribe_buffer()
        finally:
            self.cleanup()

    # ============= 送给 Whisper 识别 =============
    def transcribe_buffer(self):
        audio = self.buffer_16k.astype(np.float32)
        if len(audio) == 0:
            return

        # 这里保持你原来的设置：英语，beam search，不要时间戳
        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            language="en",
            without_timestamps=True
        )

        text = "".join(seg.text.strip() + " " for seg in segments).strip()
        if text:
            print("识别:", text)

    # ============= 清理资源 =============
    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.wav_file.close()
        print("录音已保存为 recorded_audio.wav")


if __name__ == "__main__":
    recognizer = SpeechRecognizer("Newmine")
    recognizer.start_listening()
