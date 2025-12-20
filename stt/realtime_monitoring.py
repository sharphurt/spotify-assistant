from dataclasses import dataclass
from time import time as current_time, sleep
from collections import deque
import numpy as np
import re

from stt.stt_recognizer import STTRecognizer
from stt.vad import SileroVAD

from config import *

wake_re = re.compile(WAKE_WORD_REGEX, re.IGNORECASE)


@dataclass
class MonitorInfo:
    state: str
    recorded: np.array
    last_recognized: str


class WakeMonitor:
    def __init__(self, stt: STTRecognizer, vad: SileroVAD):
        self.realtime_buffer = deque(maxlen=WINDOW)
        self.record_buffer = deque(maxlen=LARGE_AUDIO_BUFFER_SIZE)

        self.is_recording = False
        self.state = "IDLE"

        self.last_transcription_time = 0.0
        self.recording_start_time = 0.0
        self.last_speech_time = 0.0

        self.last_recognized = ''

        self.stt = stt
        self.vad = vad

    def on_audio_available(self, audio_data):
        self.realtime_buffer.extend(audio_data)

        if self.state == "RECORDING":
            self.record_buffer.extend(audio_data)

    def process(self) -> MonitorInfo:
        sleep(0.01)

        if self.vad.check_voice_activity():
            self.last_speech_time = current_time()

        if self._is_time_to_recognize():
            self.last_transcription_time = current_time()
            audio_chunk = np.array(self.realtime_buffer, dtype=np.float32)
            stt_result = self.stt.transcribe(audio_chunk)
            self.last_recognized = stt_result.result if not stt_result.is_probably_hallucination else ''

        if self.state == "IDLE":
            if re.search(wake_re, self.last_recognized):
                self._start_recording()

        elif self.state == "RECORDING":
            if current_time() - self.recording_start_time > COMMAND_TIMEOUT:
                print("Finish recording by timeout")
                self._stop_recording()
                return MonitorInfo(state=self.state, recorded=np.array(self.record_buffer, dtype=np.float32),
                                   last_recognized=self.last_recognized)

            if current_time() - self.last_speech_time > SILENCE_TIMEOUT:
                print('Finish recording by silence')
                self._stop_recording()
                return MonitorInfo(state=self.state, recorded=np.array(self.record_buffer, dtype=np.float32),
                                   last_recognized=self.last_recognized)

        return MonitorInfo(state=self.state, recorded=None, last_recognized=self.last_recognized)

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
        self.last_recognized = ''
        self.realtime_buffer.clear()

    def _is_time_to_recognize(self):
        is_buffer_enough = len(self.realtime_buffer) >= WINDOW
        is_time_to_listen = current_time() - self.last_transcription_time >= CHECK_WAKE_EVERY_SECONDS
        return is_buffer_enough and is_time_to_listen
