import re
from string import punctuation
from typing import Union

import dotenv
import numpy as np
from ddgs import DDGS

from config import *
from gpt.yandex_gpt_client import YandexGptClient
from stt.stt_recognizer import STTRecognizer

dotenv.load_dotenv()

wake_re = re.compile(WAKE_WORD_REGEX, re.IGNORECASE)
punctuation_re = re.compile(r'[^\w\s]', re.IGNORECASE)
whitespaces_re = re.compile(r'\s+', re.IGNORECASE)

system_prompt = open('system_prompt.txt', 'r', encoding='utf-8').read()
duckduckgo_search_url = "https://html.duckduckgo.com/html/?q="


class CommandProcessor:
    def __init__(self, stt: STTRecognizer, gpt: YandexGptClient):
        self.stt = stt
        self.gpt = gpt

    def process_command(self, audio: Union[str, np.ndarray]):
        stt_result = self.stt.transcribe(audio)
        if not stt_result.is_probably_hallucination:
            self.parse_command(stt_result.result)
        else:
            print('Recorded command is probably hallucination, will not be processed')
            # todo: ask to repeat

    def parse_command(self, command):
        sanitized = sanitize_text(command)
        web_search_result = self.get_web_search_info(sanitized)
        user_request = sanitized + ("\n SEARCH RESULTS:\n" + web_search_result if web_search_result else "")
        response = self.gpt.request_gpt(user_input=user_request, system_prompt=system_prompt)

        print(response)

    def get_web_search_info(self, command):
        try:
            results = DDGS().text(command + " music", safesearch='off', page=1, backend="duckduckgo")
            return sanitize_text(' '.join(map(lambda r: r['body'], results)))[0:256]
        except Exception as e:
            print(e)
            return None

def sanitize_text(text):
    remove_wake = re.sub(pattern=wake_re, repl="", string=text)
    remove_punctuation = re.sub(pattern=punctuation_re, repl="", string=remove_wake)
    remove_whitespaces = re.sub(pattern=whitespaces_re, repl=" ", string=remove_punctuation)
    return remove_whitespaces.strip()
