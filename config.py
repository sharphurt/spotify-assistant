import re

# Параметры рекордера
SAMPLE_RATE = 16000  # Количество бит на 1 секунду
BLOCK_SIZE = 1024
DEVICE = "Микрофон (5- fifine Microphone), MME"

# Параметры транскрибера
WAKE_RE = re.compile("леха|лёха|алеха|алёха|лоха|лех|лёх|йохан", re.IGNORECASE)
PUNCTUATION_RE = re.compile(r'[^\w\s]', re.IGNORECASE)
WHITESPACES_RE = re.compile(r'\s+', re.IGNORECASE)

SILENCE_TIMEOUT = 1.5
COMMAND_TIMEOUT = 7

SPEECH_RECOGNITION_THRESHOLD = 0.6

WINDOW_LENGTH_SECONDS = 2
WINDOW = int(SAMPLE_RATE * WINDOW_LENGTH_SECONDS)

CHECK_WAKE_EVERY_SECONDS = 0.25
HOP = int(SAMPLE_RATE * CHECK_WAKE_EVERY_SECONDS)

LARGE_AUDIO_BUFFER_SIZE = int(SAMPLE_RATE * COMMAND_TIMEOUT + WINDOW_LENGTH_SECONDS) * 2
