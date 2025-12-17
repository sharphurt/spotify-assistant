from faster_whisper import WhisperModel

WAKE_WORD_REGEX = "леха|лёха|алеха|алёха"

SAMPLE_RATE = 16000  # Количество бит на 1 секунду
BLOCK_SIZE = 1024
CHUNK_SECONDS = 5
DEVICE = "Микрофон (5- fifine Microphone), MME"

SPEECH_RECOGNITION_THRESHOLD = 0.7

WINDOW_LENGTH_SECS = 1.0
WINDOW = int(SAMPLE_RATE * WINDOW_LENGTH_SECS)

HOP_SECS = 0.25
HOP = int(SAMPLE_RATE * HOP_SECS)
