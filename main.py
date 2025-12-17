from config import *
from collections import deque
import sounddevice as sd
import numpy as np
import time
import re





ring = deque(maxlen=WINDOW)
last_run = 0.0
wake_re = re.compile(WAKE_WORD_REGEX)


model = WhisperModel(
    "whisper-small",
    device="cuda",
    compute_type="float16"
)


def feed_audio(samples):
    ring.extend(samples)


def get_window():
    if len(ring) == WINDOW:
        return np.array(ring, dtype=np.float32)
    return None


def callback(indata, frames, time_info, status):
    if status:
        print(status)

    samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    samples *= 2.0
    ring.extend(samples)


def detect_wake(samples: np.array):
    transcribed_segments, _ = model.transcribe(
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
        if re.search(wake_re, segment.text.lower()):
            print(f'Triggered on {segment.text}')


with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        device=DEVICE,
        callback=callback
):
    print("Listening... Ctrl+C to stop")

    try:
        while True:
            now = time.time()
            if now - last_run >= HOP_SECS and len(ring) == WINDOW:
                last_run = now
                audio = np.array(ring, dtype=np.float32)
                detect_wake(audio)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped")
