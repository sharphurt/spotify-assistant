import os

from config import SAMPLE_RATE
from input_recorder import InputRecorder
from whisper_transcriber import LocalWhisper
from command_recognizer import CommandRecognizer

from scipy.io import wavfile
from dotenv import load_dotenv

load_dotenv()

recorder = InputRecorder()
whisper_small = LocalWhisper("whisper-small")
command_recognizer = CommandRecognizer()


recorder.subscribe(whisper_small.on_audio_available)
recorder.start()

while True:
    listen_result = whisper_small.process()
    print(listen_result)
    if listen_result.recorded is not None:
        print("Команда записана, сохраняем в файл...")
        wavfile.write('audio.wav', SAMPLE_RATE, listen_result.recorded)

    #
    # transcribed = command_recognizer.recognize_command()
    # command_text = ' '.join(map(lambda x: x.text, transcribed))
    # print(command_text)

    # command_processor.decode_command(command_text)

