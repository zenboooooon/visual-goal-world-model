"""Waypoint player for E1M1 Hangar, with the enemy layout of Ultra-Violence."""

import math
import random

import vizdoom as vzd

from common import BUTTONS, DATA


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


ASSETS = DATA / "e1m1_assets"
GAME_WAD = ASSETS / "doom1.wad"
MAP_WAD = ASSETS / "E1M1.wad"
TICRATE = 35
REPEAT = 4
TIMEOUT = 120 * TICRATE
# Start room, computer room, zigzag, south door, exit room and west-facing switch.
ROUTE = (
    (1110, -3140),
    (1270, -2990),
    (1280, -2520),
    (1460, -2496),
    (1630, -2496),
    (1830, -2510),
    (1980, -2650),
    (2430, -2620),
    (2550, -2620),
    (2680, -2680),
    (2820, -2715),
    (2935, -2820),
    (2960, -2900),
    (2870, -2995),
    (2930, -3080),
    (2910, -3260),
    (3090, -3370),
    (3210, -3440),
    (3150, -3550),
    (3020, -3670),
    (3010, -3910),
    (3008, -4110),
    (3010, -4330),
    (3008, -4530),
    (3008, -4700),
    (2990, -4768),
    (2920, -4768),
)
DOORS = ((1536, -2496, 0), (3008, -4016, -90), (3008, -4632, -90), (2912, -4768, 180))


def make_game(seed):
    game = vzd.DoomGame()
    game.set_doom_game_path(str(GAME_WAD))
    game.set_doom_scenario_path(str(MAP_WAD))
    game.set_doom_map("E1M1")
    game.set_doom_skill(4)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(False)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_available_buttons([getattr(vzd.Button, b) for b in BUTTONS])
    game.set_available_game_variables(
        [
            getattr(vzd.GameVariable, k)
            for k in ("POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "HEALTH", "KILLCOUNT")
        ]
    )
    game.set_ticrate(TICRATE)
    game.set_episode_timeout(TIMEOUT)
    game.set_map_exit_reward(1.0)
    game.set_seed(seed)
    game.init()
    return game


class HangarPlayer:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.route = [(x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)) for x, y in ROUTE[:-1]] + [ROUTE[-1]]
        self.node = 0
        self.previous = None
        self.step = 0
        self.last_use = -50

    def action(self, state):
        x, y, _, angle = state.game_variables[:4]
        while self.node < len(self.route) - 1 and math.dist((x, y), self.route[self.node]) < 35:
            self.node += 1
        gx, gy = self.route[self.node]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        aim = math.degrees(math.atan2(dy, dx))
        buttons = set()
        door = next((a for px, py, a in DOORS if math.hypot(x - px, y - py) < 63), None)
        if door is not None:
            aim = door
            if self.step - self.last_use >= 22 and abs(wrap(aim - angle)) < 38:
                self.last_use = self.step
                buttons.add("USE")
        else:
            enemies = [label for label in state.labels if label.object_category == "Monster"]
            if enemies:
                enemy = min(
                    enemies, key=lambda e: math.hypot(e.object_position_x - x, e.object_position_y - y)
                )
                target = math.degrees(math.atan2(enemy.object_position_y - y, enemy.object_position_x - x))
                if abs(wrap(target - angle)) < 55:
                    aim = target
                    buttons.add("ATTACK")
        if any(
            e.object_category == "Monster"
            and abs(wrap(math.degrees(math.atan2(e.object_position_y - y, e.object_position_x - x)) - angle))
            < 20
            for e in state.labels
        ):
            buttons.add("ATTACK")
        error = wrap(aim - angle)
        if abs(error) > 7:
            buttons.add("TURN_LEFT" if error > 0 else "TURN_RIGHT")
        vx, vy = (0.0, 0.0) if self.previous is None else (x - self.previous[0], y - self.previous[1])
        self.previous = (x, y)
        scale = min(0.50, 29 / max(1.0, dist))
        dx, dy = dx * scale - 0.69 * vx, dy * scale - 0.69 * vy
        cs, sn = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        forward, left = dx * cs + dy * sn, dy * cs - dx * sn
        if abs(forward) > 3:
            buttons.add("MOVE_FORWARD" if forward > 0 else "MOVE_BACKWARD")
        if abs(left) > 3:
            buttons.add("MOVE_LEFT" if left > 0 else "MOVE_RIGHT")
        self.step += 1
        return [int(b in buttons) for b in BUTTONS]
