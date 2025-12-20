import sys
import threading

from config import SAMPLE_RATE
from input_recorder import InputRecorder
from stt.realtime_monitoring import WakeMonitor
from command_processor import CommandProcessor
from stt.vad import SileroVAD
from stt.stt_recognizer import STTRecognizer
from gpt.yandex_gpt_client import YandexGptClient
from scipy.io import wavfile
from dotenv import load_dotenv

load_dotenv()

vad = SileroVAD()
whisper_small_sst = STTRecognizer("whisper/whisper-small")
whisper_large_sst = STTRecognizer("whisper/whisper-large-v3-ct2")
openai_ogg_gpt = YandexGptClient("gpt-oss-120b/latest")

wake_monitor = WakeMonitor(stt=whisper_small_sst, vad=vad)
command_processor = CommandProcessor(stt=whisper_large_sst, gpt=openai_ogg_gpt)

recorder = InputRecorder()
recorder.subscribe(wake_monitor.on_audio_available)
recorder.subscribe(vad.on_audio_available)
recorder.start()

while True:
    listen_result = wake_monitor.process()
    # print(listen_result)
    if listen_result.recorded is not None:
        print("Команда записана, сохраняем в файл...")
        wavfile.write('audio.wav', SAMPLE_RATE, listen_result.recorded)
        command_recognition_thread = threading.Thread(target=command_processor.process_command(listen_result.recorded))
        command_recognition_thread.start()

    #
    # transcribed = command_recognizer.recognize_command()
    # command_text = ' '.join(map(lambda x: x.text, transcribed))
    # print(command_text)

    # command_processor.decode_command(command_text)

