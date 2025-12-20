import logging

import requests

from dto.spotify import SpotifySearchRequest

logger = logging.getLogger(__name__)

base_proxy_url = "http://77.110.122.194:8200/proxy/spotify"
play_by_request_url = f"{base_proxy_url}/play_request"
set_device_url = f"{base_proxy_url}/set_device"


class SpotifyProxyClient:

    def set_device(self, device_id):
        pass

    def play_by_request(self, search_request: SpotifySearchRequest):
        response = requests.post(play_by_request_url, json=search_request)
        if not response.ok:
            logger.error("Не удалось включить трек в Spotify", response)
            return False

        return True