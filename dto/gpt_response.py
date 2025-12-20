from dataclasses import dataclass

from pydantic import BaseModel

from dto.intent import Intent
from dto.spotify import SpotifySearchRequest


class GptRawResponse(BaseModel):
    intent: Intent
    query: str
    human_response: str

@dataclass
class GptResultResponse:
    spotify_search_request: SpotifySearchRequest
    humanfriendly_description: str