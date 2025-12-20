from dataclasses import dataclass

import numpy as np


@dataclass
class MonitorInfo:
    state: str
    recorded: np.array
    last_recognized: str