import logging
from collections import deque
from time import time as current_time, sleep

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from config import *


class SileroVAD:
    def __init__(self):
        self.model = load_silero_vad()
        self.realtime_buffer = deque(maxlen=int(WINDOW / 4))
        self.last_transcription_time = 0.0
        self.last_voice_info = None

    def on_audio_available(self, audio_data):
        self.realtime_buffer.extend(audio_data)

    def check_voice_activity(self):
        if self._is_time_to_recognize():
            self.last_transcription_time = current_time()
            audio_chunk = np.array(self.realtime_buffer, dtype=np.float32)
            self.last_voice_info = self.check_voice(audio_chunk)

        # print(self.last_voice_info)
        return self.last_voice_info and len(self.last_voice_info) > 0

    def _is_time_to_recognize(self):
        is_buffer_enough = len(self.realtime_buffer) == self.realtime_buffer.maxlen
        is_time_to_listen = current_time() - self.last_transcription_time >= CHECK_WAKE_EVERY_SECONDS / 4
        return is_buffer_enough and is_time_to_listen

    def check_voice(self, audio_chunk):
        wav = torch.from_numpy(audio_chunk)
        speech_timestamps = get_speech_timestamps(
            wav,
            self.model,
            return_seconds=True
        )

        return speech_timestamps
