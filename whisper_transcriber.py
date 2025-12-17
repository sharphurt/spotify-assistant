from time import time as current_time, sleep
from collections import deque
from faster_whisper import WhisperModel
from config import *
import re
import numpy as np

wake_re = re.compile(WAKE_WORD_REGEX)


class WhisperTranscriber:
    def __init__(self, model_name: str):
        self.hop_queue = deque(maxlen=WINDOW)
        self.large_queue = deque(maxlen=LARGE_AUDIO_BUFFER_SIZE)

        self.last_transcription_time = 0.0
        self.recording_start_time = 0.0
        self.last_speech_time = 0.0

        self.model = WhisperModel(
            model_size_or_path=model_name,
            device="cuda",
            compute_type="float16"
        )

    def wait_wake_word(self):
        while True:
            if self._is_time_to_recognize():
                self.last_transcription_time = current_time()
                audio_chunk = np.array(self.hop_queue, dtype=np.float32)

                transcribed = self._transcribe(audio_chunk)
                if transcribed is not None and re.search(wake_re, transcribed.lower()):
                    return

            sleep(0.01)

    def record_command(self):
        self.recording_start_time = current_time()
        self.last_speech_time = current_time()
        self.large_queue.clear()

        while self._can_listen():
            if self._is_time_to_recognize():
                self.last_transcription_time = current_time()
                if self._is_active_voice(np.array(self.hop_queue, dtype=np.float32)):
                    self.last_speech_time = current_time()

            sleep(0.01)

        int16_audio = np.array(self.large_queue, dtype=np.float32).clip(-1.0, 1.0) * 32767
        return int16_audio.astype(np.int16)

    def _can_listen(self):
        return current_time() - self.recording_start_time <= MAX_COMMAND_DURATION_SECONDS \
            and current_time() - self.last_speech_time <= MAX_SILENCE_DURATION_SECONDS

    def _is_time_to_recognize(self):
        return (current_time() - self.last_transcription_time >= CHECK_WAKE_EVERY_SECONDS
                and len(self.hop_queue) == WINDOW)

    def _transcribe(self, samples: np.array):
        transcribed_segments, _ = self.model.transcribe(
            samples,
            language="ru",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=SPEECH_RECOGNITION_THRESHOLD
        )

        for segment in transcribed_segments:
            if segment.no_speech_prob > SPEECH_RECOGNITION_THRESHOLD:
                continue

            print(f"Try: {segment.text} | {segment.temperature} | {segment.no_speech_prob}")
            return segment.text

    def _is_active_voice(self, samples: np.array):
        segments, _ = self.model.transcribe(
            samples,
            language="ru",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=1.0
        )

        for seg in segments:
            if seg.no_speech_prob is not None and seg.no_speech_prob < SPEECH_RECOGNITION_THRESHOLD:
                return True

        return False

    def _get_window(self):
        if len(self.hop_queue) == WINDOW:
            return np.array(self.hop_queue, dtype=np.float32)

        return None

    def on_audio_available(self, audio_data):
        self.hop_queue.extend(audio_data)
        self.large_queue.extend(audio_data)
