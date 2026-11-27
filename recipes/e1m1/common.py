import json
import os
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("E1M1_DATA", HERE / "data")).resolve()
BUTTONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
)
TICRATE = 35
REPEAT = 4
MAX_TICS = 120 * TICRATE
SIZE = 196
HEIGHT = 147
TOP = (SIZE - HEIGHT) // 2
XS = ((np.arange(SIZE) + 0.5) * 320 / SIZE).astype(int)
YS = ((np.arange(HEIGHT) + 0.5) * 240 / HEIGHT).astype(int)
LANDMARKS = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)


def start_episode(game, seed, demo_path=None):
    game.set_seed(int(seed))
    if demo_path is None:
        game.new_episode()
    else:
        game.new_episode(str(Path(demo_path).resolve()))


def frame(state):
    """Return RGB and raw uint8 engine depth letterboxed to the network dimensions."""
    rgb = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    depth = np.zeros((SIZE, SIZE), dtype=np.uint8)
    rgb[TOP : TOP + HEIGHT] = state.screen_buffer[YS[:, None], XS]
    depth[TOP : TOP + HEIGHT] = state.depth_buffer[YS[:, None], XS]
    return rgb, depth


def goal_indices(progress, reference_progress, reference_action_tics):
    """Look up the next ordered waypoint in a successful training trajectory."""
    ref = np.maximum.accumulate(reference_progress)
    waypoint_rows = np.searchsorted(ref, LANDMARKS, side="left").clip(0, len(ref) - 1)
    waypoint_tics = np.asarray(reference_action_tics)[waypoint_rows]
    next_waypoint = np.searchsorted(LANDMARKS, progress, side="right").clip(0, len(LANDMARKS) - 1)
    return waypoint_tics[next_waypoint]


def json_write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(path)
