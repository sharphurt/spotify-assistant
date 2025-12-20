import logging
from typing import Union

import numpy as np
from ddgs import DDGS
from pydantic import ValidationError

from config import *
from dto.gpt_response import GptRawResponse, GptResultResponse
from dto.spotify import SpotifySearchRequest, map_intent_to_spotify_type
from gpt.abstract_gpt_client import AbstractGptClient
from spotify_proxy_client import SpotifyProxyClient
from stt.stt_recognizer import STTRecognizer

logger = logging.getLogger(__name__)
system_prompt = open('system_prompt.txt', 'r', encoding='utf-8').read()


class CommandProcessor:
    def __init__(self, stt: STTRecognizer, gpt: AbstractGptClient, spotify_client: SpotifyProxyClient):
        self.stt = stt
        self.gpt = gpt
        self.spotify = spotify_client

    def process_command(self, audio: Union[str, np.ndarray]):
        stt_result = self.stt.transcribe(audio)
        if not stt_result.is_probably_hallucination:
            self.parse_command(stt_result.result)
        else:
            logger.warning('Recorded command is probably hallucination, will not be processed')
            # todo: ask to repeat

    def parse_command(self, command):
        sanitized = self._sanitize_text(command)
        web_search_result = self.get_web_search_info(sanitized)
        user_request = sanitized + ("\n SEARCH RESULTS:\n" + web_search_result if web_search_result else "")
        response = self.gpt.request_gpt(user_input=user_request, system_prompt=system_prompt)

        gpt_result = self._validate_gpt_response(response)
        if gpt_result is None:
            return

        logger.info(gpt_result)
        self.spotify.play_by_request(gpt_result.spotify_search_request)

    def get_web_search_info(self, command):
        try:
            results = DDGS().text(command + " music", safesearch='off', page=1, backend="duckduckgo")
            sanitized = self._sanitize_text(' '.join(map(lambda r: r['body'], results)))
            logger.info(f"Web search results: {sanitized}")
            return sanitized
        except Exception as e:
            logger.error(e)
            return None

    def _sanitize_text(self, text):
        remove_wake = re.sub(pattern=WAKE_RE, repl="", string=text)
        remove_punctuation = re.sub(pattern=PUNCTUATION_RE, repl="", string=remove_wake)
        remove_whitespaces = re.sub(pattern=WHITESPACES_RE, repl=" ", string=remove_punctuation)
        return remove_whitespaces.strip()

    def _validate_gpt_response(self, response):
        if response is None or response == '':
            logger.error("Got empty response from GPT")
            return None

        try:
            gpt_response = GptRawResponse.model_validate_json(response)
            spotify_request_dto = SpotifySearchRequest(
                type=map_intent_to_spotify_type(gpt_response.intent),
                q=gpt_response.query
            )

            return GptResultResponse(spotify_search_request=spotify_request_dto,
                                     humanfriendly_description=gpt_response.human_response)
        except ValidationError as e:
            logger.error(f"JSON {response} has invalid schema", e)
            return None
