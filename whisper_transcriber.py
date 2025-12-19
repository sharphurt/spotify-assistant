from dataclasses import dataclass
from time import time as current_time, sleep
from collections import deque
from faster_whisper import WhisperModel
import numpy as np
import re

from scipy.signal import ellip

from config import *

wake_re = re.compile(WAKE_WORD_REGEX, re.IGNORECASE)
hallucination_phrases_score = [
    ('спасибо', 1),
    ('субтитр', 5),
    ('редакт', 3),
    ('коррект', 3)
]


@dataclass
class ListenResult:
    state: str
    recorded: np.array
    last_recognized: str


class LocalWhisper:
    def __init__(self, model_name: str):
        self.realtime_buffer = deque(maxlen=WINDOW)
        self.record_buffer = deque(maxlen=LARGE_AUDIO_BUFFER_SIZE)

        self.is_recording = False
        self.state = "IDLE"

        self.last_transcription_time = 0.0
        self.recording_start_time = 0.0
        self.last_speech_time = 0.0

        self.last_recognized = ''

        self.model = WhisperModel(
            model_size_or_path=model_name,
            device="cuda",
            compute_type="float16"
        )

    def on_audio_available(self, audio_data):
        self.realtime_buffer.extend(audio_data)

        if self.state == "RECORDING":
            self.record_buffer.extend(audio_data)

    def process(self) -> ListenResult:
        sleep(0.01)
        if self._is_time_to_recognize():
            self.last_transcription_time = current_time()
            audio_chunk = np.array(self.realtime_buffer, dtype=np.float32)
            self.last_recognized = self._transcribe(audio_chunk)

        if self.state == "IDLE":
            if re.search(wake_re, self.last_recognized):
                self._start_recording()

        elif self.state == "RECORDING":
            if _is_active_voice(np.array(self.realtime_buffer, dtype=np.float32)):
                self.last_speech_time = current_time()

            if current_time() - self.recording_start_time > COMMAND_TIMEOUT:
                # or current_time() - self.last_speech_time > SILENCE_TIMEOUT:
                self._stop_recording()
                return ListenResult(state=self.state, recorded=np.array(self.record_buffer, dtype=np.float32),
                                    last_recognized=self.last_recognized)

        return ListenResult(state=self.state, recorded=None, last_recognized=self.last_recognized)

    def _start_recording(self):
        print('RECORDING')
        self.state = "RECORDING"
        self.is_recording = True
        self.recording_start_time = current_time()
        self.last_speech_time = current_time()
        self.record_buffer.clear()
        self.record_buffer.extend(self.realtime_buffer)

    def _stop_recording(self):
        self.state = "IDLE"
        self.is_recording = False
        self.realtime_buffer.clear()

    def _is_time_to_recognize(self):
        is_buffer_enough = len(self.realtime_buffer) >= WINDOW
        is_time_to_listen = current_time() - self.last_transcription_time >= CHECK_WAKE_EVERY_SECONDS
        return is_buffer_enough and is_time_to_listen

    def _transcribe(self, samples: np.array):
        segments, _ = self.model.transcribe(
            samples,
            language="ru",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=0.4
        )

        transcribed = ' '.join(map(lambda x: x.text, segments))
        return transcribed if not _is_probably_hallucination(transcribed) else ''


def _is_active_voice(samples: np.array):
    if len(samples) == 0:
        return False

    rms = np.sqrt(np.mean(samples ** 2))
    return rms > VAD_THRESHOLD


def _is_probably_hallucination(input: str):
    score = 5

    if input.isupper():
        score -= 2

    for hallucination_phrase, phrase_score in hallucination_phrases_score:
        if hallucination_phrase in input.lower():
            score -= phrase_score

    words = input.lower().split()
    score -= len(words) - len(set(words))

    return score < 0
