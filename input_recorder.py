from collections.abc import Callable

import numpy as np
import sounddevice as sd
from config import *

class InputRecorder:
    def __init__(self):
        self._stream = None
        self._subscribers = []

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Статус: {status}")

        audio_data = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
        audio_data /= 32768.0

        for subscriber in self._subscribers:
            subscriber(audio_data)


    def start(self):
        if self._stream is not None:
            print('Input already listening. Call stop before')
            return

        self._stream = sd.RawInputStream(samplerate=SAMPLE_RATE,
                                         blocksize=BLOCK_SIZE,
                                         dtype="int16",
                                         channels=1,
                                         device=DEVICE,
                                         callback=self._callback)
        self._stream.start()
        print('Listening...')


    def subscribe(self, on_audio_available_handler: Callable):
        self._subscribers.append(on_audio_available_handler)