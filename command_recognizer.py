from faster_whisper import WhisperModel
from config import *


class CommandRecognizer:
    def __init__(self):
        self.model = WhisperModel(
            model_size_or_path="whisper-large-v3-ct2",
            device="cuda",
            compute_type="float16"
        )

    def recognize_command(self):
        transcribed_segments, _ = self.model.transcribe(
            "audio.wav",
            vad_filter=True,
            no_speech_threshold=SPEECH_RECOGNITION_THRESHOLD
        )

        return transcribed_segments
