from enum import Enum


class Intent(str, Enum):
    play_track = "play_track"
    play_album = "play_album"
    play_artist = "play_artist"
    play_playlist = "play_playlist"
    search = "search"
    unknown = "unknown"
