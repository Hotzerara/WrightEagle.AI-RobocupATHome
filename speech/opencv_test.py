import wave
import numpy as np
import pyaudio
import samplerate
from faster_whisper import WhisperModel
import cv2
from silero_vad import load_silero_vad, VADIterator
import threading
import queue
import time

from config import WHISPER_MEDIUM


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
        self.last_transcription = ""
        self.current_status = "Listening..."
        self.current_energy = 0.0

        # 启动时简单检测环境噪声
        self.noise_floor = self.detect_noise_floor()
        print(f"环境噪声能量估计: {self.noise_floor:.6f}")

        # OpenCV可视化参数
        self.window_name = "Real-time Speech Recognition"
        self.window_width = 800
        self.window_height = 600

        # 识别队列和线程
        self.recognition_queue = queue.Queue()
        self.recognition_thread = None
        self.recognition_active = False

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
        num_frames = 10  # 10 * 32ms = 320ms
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

    def recognition_worker(self):
        """后台识别工作线程"""
        while self.recognition_active:
            try:
                # 从队列获取需要识别的音频数据
                audio_data = self.recognition_queue.get(timeout=0.1)
                if audio_data is None:  # 停止信号
                    break

                # 执行识别
                segments, _ = self.model.transcribe(
                    audio_data,
                    beam_size=5,
                    language="en",
                    without_timestamps=True
                )

                text = "".join(seg.text.strip() + " " for seg in segments).strip()
                if text:
                    print("识别:", text)
                    self.last_transcription = text
                    self.current_status = f"Result: {text[:50]}..."
                    if len(text) > 50:
                        self.current_status += "..."
                else:
                    self.current_status = "No speech detected"

            except queue.Empty:
                continue
            except Exception as e:
                print(f"识别出错: {e}")
                self.current_status = "Recognition error"

        print("识别线程已退出")

    def start_recognition_thread(self):
        """启动识别线程"""
        self.recognition_active = True
        self.recognition_thread = threading.Thread(target=self.recognition_worker, daemon=True)
        self.recognition_thread.start()

    def submit_for_recognition(self, audio_data):
        """提交音频数据进行识别"""
        # 清空队列，避免积压旧数据
        with self.recognition_queue.mutex:
            self.recognition_queue.queue.clear()
        self.recognition_queue.put(audio_data.copy())

    def draw_visualization(self):
        """绘制OpenCV可视化界面"""
        # 创建深色背景 (现代感)
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        img.fill(20)  # 深灰色背景

        # 添加渐变背景效果
        for i in range(self.window_height):
            alpha = i / self.window_height
            color = int(20 + 10 * alpha)
            cv2.line(img, (0, i), (self.window_width, i), (color, color, color), 1)

        # === 标题区域 ===
        title_bg = np.zeros((80, self.window_width, 3), dtype=np.uint8)
        title_bg.fill(30)
        img[0:80, 0:self.window_width] = title_bg

        # 主标题
        cv2.putText(img, "Real-time Speech Recognition", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)

        # 副标题
        cv2.putText(img, "Silero VAD + Whisper Medium", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)

        # === 状态卡片 ===
        card_y = 100
        card_height = 65
        card_bg = np.zeros((card_height, self.window_width - 60, 3), dtype=np.uint8)
        card_bg.fill(40)

        # 添加卡片边框
        cv2.rectangle(img, (30, card_y), (self.window_width - 30, card_y + card_height),
                      (60, 60, 60), 1)
        img[card_y:card_y + card_height, 30:self.window_width - 30] = card_bg

        # 状态指示器 (动态颜色)
        if self.is_speaking:
            status_color = (0, 255, 0)  # 绿色 - 说话中
            status_text = "SPEAKING"
            # 添加脉动效果
            pulse = int(10 * abs(np.sin(time.time() * 8)))
            cv2.circle(img, (60, card_y + 40), 12,
                       (0, 255 - pulse, 0), -1)
        elif self.current_status.startswith("Recognizing"):
            status_color = (255, 255, 0)  # 黄色 - 识别中
            status_text = "PROCESSING"
            cv2.circle(img, (60, card_y + 40), 12,
                       (255, 255, 0), 2)
        else:
            status_color = (100, 100, 255)  # 蓝色 - 监听中
            status_text = "LISTENING"
            cv2.circle(img, (60, card_y + 40), 10,
                       status_color, -1)

        # 状态文本
        cv2.putText(img, status_text, (90, card_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 详细状态
        cv2.putText(img, self.current_status, (90, card_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # === 音量仪表 ===
        meter_y = 200
        cv2.putText(img, "VOLUME LEVEL", (30, meter_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1)

        # 计算音量百分比 (更平滑的显示)
        energy_ratio = max(0.0, (self.current_energy - self.noise_floor) / (0.1 - self.noise_floor))
        energy_percent = min(100.0, energy_ratio * 100)

        # 音量条背景
        cv2.rectangle(img, (30, meter_y), (self.window_width - 30, meter_y + 25),
                      (50, 50, 50), -1)

        # 动态音量条 (颜色渐变)
        bar_width = int((self.window_width - 60) * energy_percent / 100)
        if energy_percent < 30:
            bar_color = (0, 200, 0)  # 绿色
        elif energy_percent < 70:
            bar_color = (0, 200, 200)  # 黄色
        else:
            bar_color = (0, 100, 255)  # 红色

        if bar_width > 0:
            cv2.rectangle(img, (30, meter_y), (30 + bar_width, meter_y + 25),
                          bar_color, -1)

        # 音量数值
        cv2.putText(img, f"{energy_percent:.1f}%", (self.window_width - 80, meter_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 刻度标记
        for i in range(0, 101, 25):
            x_pos = 30 + int((self.window_width - 60) * i / 100)
            cv2.line(img, (x_pos, meter_y + 25), (x_pos, meter_y + 30),
                     (150, 150, 150), 1)
            cv2.putText(img, str(i), (x_pos - 10, meter_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # === 识别结果区域 ===
        result_y = 280
        cv2.putText(img, "TRANSCRIPTION", (30, result_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)

        # 结果背景框
        result_bg = np.zeros((180, self.window_width - 60, 3), dtype=np.uint8)
        result_bg.fill(25)
        cv2.rectangle(img, (30, result_y + 20), (self.window_width - 30, result_y + 180),
                      (40, 40, 40), -1)
        cv2.rectangle(img, (30, result_y + 20), (self.window_width - 30, result_y + 180),
                      (80, 80, 80), 1)

        # 显示识别文本 (多行支持)
        y_offset = result_y + 50
        max_line_width = 60  # 字符数限制

        if self.last_transcription:
            lines = []
            words = self.last_transcription.split()
            current_line = ""

            for word in words:
                test_line = current_line + word + " "
                if len(test_line) <= max_line_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = word + " "

            if current_line:
                lines.append(current_line.strip())

            # 限制显示行数
            lines = lines[-4:]  # 只显示最后4行

            for i, line in enumerate(lines):
                # 文字阴影效果
                cv2.putText(img, line, (52, y_offset + i * 35 + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                # 主文字
                cv2.putText(img, line, (50, y_offset + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (173, 216, 230), 2)
        else:
            # 无内容时的提示
            placeholder = "Speak to see transcription here..."
            cv2.putText(img, placeholder, (50, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 150), 1)

        # === 底部信息栏 ===
        footer_y = self.window_height - 40
        footer_bg = np.zeros((40, self.window_width, 3), dtype=np.uint8)
        footer_bg.fill(15)
        img[footer_y:footer_y + 40, 0:self.window_width] = footer_bg

        # 退出提示
        cv2.putText(img, "Press 'q' to exit",
                    (self.window_width // 2 - 100, self.window_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow(self.window_name, img)

        # 检查退出键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt

    # ============= 主循环 =============
    def start_listening(self):
        print("开始实时语音识别 (Silero VAD + OpenCV 可视化)...\n")
        print("OpenCV窗口已打开，请关注界面。按 'q' 键退出。")

        # 创建OpenCV窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        # 启动识别线程
        self.start_recognition_thread()

        try:
            while True:
                # 按帧读取
                data = self.stream.read(self.frame_samples_device, exception_on_overflow=False)
                # 保存到 wav
                self.wav_file.writeframes(data)

                # 转成 float32 [-1, 1]（设备采样率）
                audio_dev = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # 更新当前能量值用于可视化
                self.current_energy = self.rms(audio_dev)

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
                        self.current_status = "Speaking..."
                        self.buffer_16k = np.array([], dtype=np.float32)  # 重置缓冲区

                    if 'end' in speech_dict:
                        # 检测到语音结束
                        print("说话结束，开始识别...")
                        self.is_speaking = False
                        self.current_status = "Recognizing..."
                        if len(self.buffer_16k) > int(self.target_rate * 0.1):  # 至少0.1秒
                            # 提交音频数据到后台线程进行识别
                            self.submit_for_recognition(self.buffer_16k)
                        else:
                            self.current_status = "Too short"

                # 如果正在说话中，将音频加入缓冲区
                if self.is_speaking:
                    self.buffer_16k = np.concatenate([self.buffer_16k, audio_16k])

                # 更新状态
                if not self.is_speaking and self.current_status != "Recognizing...":
                    self.current_status = "Listening..."

                # 绘制可视化界面
                self.draw_visualization()

        except KeyboardInterrupt:
            print("识别已停止")
            # 停止前把最后一段说完的话也识别一下
            if self.is_speaking and len(self.buffer_16k) > int(self.target_rate * 0.1):
                self.submit_for_recognition(self.buffer_16k)
        finally:
            # 停止识别线程
            self.recognition_active = False
            # 发送停止信号
            try:
                self.recognition_queue.put(None, timeout=0.1)
            except:
                pass
            if self.recognition_thread and self.recognition_thread.is_alive():
                self.recognition_thread.join(timeout=2)
            self.cleanup()

    # ============= 送给 Whisper 识别 =============
    def transcribe_buffer(self):
        """保持这个方法用于兼容性，实际使用后台线程"""
        pass

    # ============= 清理资源 =============
    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.wav_file.close()
        cv2.destroyAllWindows()
        print("录音已保存为 recorded_audio.wav")


if __name__ == "__main__":
    recognizer = SpeechRecognizer("Newmine")
    recognizer.start_listening()