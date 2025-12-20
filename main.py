import logging
import threading
from datetime import datetime

from config import SAMPLE_RATE
from input_recorder import InputRecorder
from stt.realtime_monitoring import WakeMonitor
from command_processor import CommandProcessor
from stt.vad import SileroVAD
from stt.stt_recognizer import STTRecognizer
from gpt.yandex_gpt_client import YandexGptClient
from gpt.groq_gpt_client import GroqGtpClient
from scipy.io import wavfile
from dotenv import load_dotenv


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def main():
    load_dotenv()
    init_logging()

    vad = SileroVAD()
    whisper_small_sst = STTRecognizer("whisper/whisper-small")
    whisper_large_sst = STTRecognizer("whisper/whisper-large-v3-ct2")
    # yandex_gpt_client = YandexGptClient("gpt-oss-120b/latest")
    groq_gpt_client = GroqGtpClient(model_name="openai/gpt-oss-20b")

    wake_monitor = WakeMonitor(stt=whisper_small_sst, vad=vad)
    command_processor = CommandProcessor(stt=whisper_large_sst, gpt=groq_gpt_client)

    recorder = InputRecorder()
    recorder.subscribe(wake_monitor.on_audio_available)
    recorder.subscribe(vad.on_audio_available)
    recorder.start()

    while True:
        listen_result = wake_monitor.process()
        # print(listen_result)
        if listen_result.recorded is not None:
            print("Команда записана, сохраняем в файл...")
            wavfile.write(get_filename(), SAMPLE_RATE, listen_result.recorded)
            command_recognition_thread = threading.Thread(
                target=command_processor.process_command(listen_result.recorded)
            )
            command_recognition_thread.start()

        #
        # transcribed = command_recognizer.recognize_command()
        # command_text = ' '.join(map(lambda x: x.text, transcribed))
        # print(command_text)

        # command_processor.decode_command(command_text)


def get_filename():
    return f'records/command-record-{datetime.now().strftime("%H-%M-%S_%d-%m-%Y")}.wav'


if __name__ == "__main__":
    main()
