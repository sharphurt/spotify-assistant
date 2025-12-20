from dataclasses import dataclass
from typing import Union
import logging

import numpy as np
from faster_whisper import WhisperModel
from config import *

logger = logging.getLogger(__name__)
hallucination_phrases_score = [
    ('спасибо', 1),
    ('субтитр', 5),
    ('редакт', 3),
    ('коррект', 3)
]

@dataclass
class STTResult:
    result: str
    is_probably_hallucination: bool


class STTRecognizer:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio: Union[str, np.ndarray]):
        segments, _ = self.model.transcribe(
            audio,
            language="ru",
            vad_filter=True,
            no_speech_threshold=SPEECH_RECOGNITION_THRESHOLD
        )

        transcribed = ' '.join(map(lambda x: x.text, segments))
        is_probably_hallucination = self._is_probably_hallucination(transcribed)

        logger.info(f"WhisperRecognizer [{self.model_name}]: recognized '{transcribed}'")
        return STTResult(transcribed, is_probably_hallucination)

    @staticmethod
    def _is_probably_hallucination(input: str):
        score = 5

        if input.isupper():
            score -= 2
            logger.debug(f"Input '{input}' is entirely in caps, hallucination score: {score}")

        for hallucination_phrase, phrase_score in hallucination_phrases_score:
            if hallucination_phrase in input.lower():
                score -= phrase_score
                logger.debug(f"Hallicination marker '{hallucination_phrase}' exists in '{input}', hallucination score: {score}")

        words = input.lower().split()
        phrase_length_difference = len(words) - len(set(words))
        score -= phrase_length_difference
        logger.debug(f"Difference between all words count and unique words count is '{phrase_length_difference}', hallucination score: {score}")

        return score < 0
