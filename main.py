from config import SAMPLE_RATE
from input_recorder import InputRecorder
from whisper_transcriber import WhisperTranscriber
from scipy.io import wavfile


recorder = InputRecorder()
whisper = WhisperTranscriber("whisper-small")

recorder.subscribe(whisper.on_audio_available)
recorder.start()

while True:
    print('Waiting wake word...')
    whisper.wait_wake_word()

    print('Recording command...')
    command_audio = whisper.record_command()
    wavfile.write('audio.wav', SAMPLE_RATE, command_audio)
    print('Command recorded')



