import logging
import re
from typing import Union

import numpy as np
from ddgs import DDGS

from config import *
from gpt.abstract_gpt_client import AbstractGptClient
from stt.stt_recognizer import STTRecognizer

logger = logging.getLogger(__name__)

wake_re = re.compile(WAKE_WORD_REGEX, re.IGNORECASE)
punctuation_re = re.compile(r'[^\w\s]', re.IGNORECASE)
whitespaces_re = re.compile(r'\s+', re.IGNORECASE)

system_prompt = open('system_prompt.txt', 'r', encoding='utf-8').read()
duckduckgo_search_url = "https://html.duckduckgo.com/html/?q="


class CommandProcessor:
    def __init__(self, stt: STTRecognizer, gpt: AbstractGptClient):
        self.stt = stt
        self.gpt = gpt

    def process_command(self, audio: Union[str, np.ndarray]):
        stt_result = self.stt.transcribe(audio)
        if not stt_result.is_probably_hallucination:
            self.parse_command(stt_result.result)
        else:
            logger.warning('Recorded command is probably hallucination, will not be processed')
            # todo: ask to repeat

    def parse_command(self, command):
        sanitized = sanitize_text(command)
        web_search_result = self.get_web_search_info(sanitized)
        user_request = sanitized + ("\n SEARCH RESULTS:\n" + web_search_result if web_search_result else "")
        response = self.gpt.request_gpt(user_input=user_request, system_prompt=system_prompt)
        if response is None or response == '':
            logger.error("Got empty response from GPT")

        logger.info(response)

    def get_web_search_info(self, command):
        try:
            results = DDGS().text(command + " music", safesearch='off', page=1, backend="duckduckgo")
            sanitized = sanitize_text(' '.join(map(lambda r: r['body'], results)))
            logger.info(f"Web search results: {sanitized}")
            return sanitized
        except Exception as e:
            logger.error(e)
            return None


def sanitize_text(text):
    remove_wake = re.sub(pattern=wake_re, repl="", string=text)
    remove_punctuation = re.sub(pattern=punctuation_re, repl="", string=remove_wake)
    remove_whitespaces = re.sub(pattern=whitespaces_re, repl=" ", string=remove_punctuation)
    return remove_whitespaces.strip()
