from dataclasses import dataclass
from enum import Enum

from dto.intent import Intent


class RequestType(str, Enum):
    play_track = "track"
    play_album = "album"
    play_artist = "artist"
    play_playlist = "playlist"
    search = "track"


@dataclass
class SpotifySearchRequest:
    type: RequestType
    q: str


def map_intent_to_spotify_type(intent: Intent):
    return RequestType[intent.value]
