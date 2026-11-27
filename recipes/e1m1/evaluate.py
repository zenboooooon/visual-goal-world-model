"""Evaluate an E1M1 RGB policy on held-out seeds using the training visual-goal schedule."""

import argparse
from collections import Counter, deque
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from common import BUTTONS, LANDMARKS, REPEAT, TICRATE, frame, goal_indices, json_write, start_episode
from sampling import file_hash
from player import GAME_WAD, MAP_WAD, ROUTE, TIMEOUT, HangarPlayer, make_game
from experiment import load_policy
from common import DATA

HERE = Path(__file__).resolve().parent


def reference(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    row = next(r for r in manifest["episodes"] if r["id"] == manifest["reference_id"])
    assert row["source"] == "success" and "dev_start" not in row
    root = manifest_path.parent
    bank = np.load(root / row["bank"], mmap_mode="r")
    nodes = np.arange(len(ROUTE), dtype=np.float32) / (len(ROUTE) - 1)
    with np.load(HERE / "reference/goal_reference.npz") as source, np.load(root / row["labels"]) as labels:
        goal_tics = goal_indices(nodes, source["progress"], source["action_tics"])
        assert np.array_equal(goal_tics[source["route_node"]], labels["goals"])
    goal_frames = {
        int(t): bank[np.arange(max(0, t - 12), t + 1, REPEAT)].transpose(0, 2, 3, 1)
        for t in np.unique(goal_tics)
    }
    return goal_tics, goal_frames


def rollout(policy, seed, goal_tics, images, dest):
    route = HangarPlayer(seed).route
    node = decisions = 0
    frames = deque(maxlen=4)
    game = make_game(seed)
    try:
        start_episode(game, seed, dest / "game.lmp")
        initial_pose = game.get_state().game_variables[:4].tolist()
        start = game.get_episode_time()
        with (dest / "decisions.jsonl").open("w") as log:
            while not game.is_episode_finished():
                s = game.get_state()
                x, y, _, angle, health, kills = s.game_variables
                while node < len(route) - 1 and math.dist((x, y), route[node]) < 35:
                    node += 1
                goal = int(goal_tics[node])
                rgb, _ = frame(s)
                frames.append(rgb)
                action = policy.act(frames, None, images[goal], None).tolist()
                log.write(
                    json.dumps(
                        dict(
                            tic=decisions * REPEAT,
                            xy=[x, y],
                            yaw=angle,
                            health=health,
                            kills=kills,
                            route_node=node,
                            goal_tic=goal,
                            action=action,
                        )
                    )
                    + "\n"
                )
                clock = game.get_episode_time()
                game.make_action(action, REPEAT)
                decisions += 1
        end = game.get_episode_time()
        dead = game.is_player_dead()
        exited = game.get_total_reward() > 0.5 and end < clock and not dead
        if not dead and not exited and decisions * REPEAT < TIMEOUT - REPEAT:
            raise ValueError("Unexpected episode end")
        return dict(
            seed=seed,
            initial_pose=initial_pose,
            decisions=decisions,
            reason="exit" if exited else "death" if dead else "timeout",
            seconds=(decisions * REPEAT if exited else end - start) / TICRATE,
            last_route_node=node,
            max_progress=node / (len(route) - 1),
        )
    finally:
        game.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first-seed", type=int, default=61000)
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()
    dest = args.out.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest = DATA / "e1m1_mixed_rgb/manifest.json"
    signature = dict(
        checkpoint_sha256=file_hash(args.checkpoint),
        manifest_sha256=file_hash(manifest),
        map_sha256=file_hash(MAP_WAD),
        game_sha256=file_hash(GAME_WAD),
        first_seed=args.first_seed,
        count=args.count,
        skill=4,
        timeout_tics=TIMEOUT,
        action_repeat=REPEAT,
        script_sha256=file_hash(Path(__file__)),
    )
    protocol = dest / "protocol.json"
    torch.set_num_threads(2)
    with (dest / "eval.lock").open("a") as lock, tempfile.TemporaryDirectory() as runtime:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if protocol.exists() and json.loads(protocol.read_text()) != signature:
            raise ValueError("Output contains a different evaluation")
        json_write(protocol, signature)
        policy = load_policy(args.checkpoint, "cuda:0")
        assert policy.rgb_only and policy.buttons == BUTTONS
        goal_tics, images = reference(manifest)
        used = {r["seed"] for r in json.loads((HERE / "reference/attempts.json").read_text())}
        seeds = range(args.first_seed, args.first_seed + args.count)
        assert args.count > 0 and not used.intersection(seeds)
        os.chdir(runtime)
        rows = []
        for seed in seeds:
            final = dest / f"seed_{seed}"
            if not final.exists():
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    staged = Path(tmp) / final.name
                    staged.mkdir()
                    row = rollout(policy, seed, goal_tics, images, staged)
                    json_write(staged / "result.json", row)
                    staged.rename(final)
            row = json.loads((final / "result.json").read_text())
            rows.append(row)
            assert row["seed"] == seed and row["initial_pose"] == rows[0]["initial_pose"]
            counts = Counter(r["reason"] for r in rows)
            reached = {str(int(p * 100)): int(sum(r["max_progress"] >= p for r in rows)) for p in LANDMARKS}
            summary = dict(
                starts=len(rows),
                exits=counts["exit"],
                deaths=counts["death"],
                timeouts=counts["timeout"],
                route_threshold_counts=reached,
                median_max_progress=float(np.median([r["max_progress"] for r in rows])),
                episodes=rows,
                signature=signature,
            )
            json_write(dest / "summary.json", summary)
            if len(rows) % 5 == 0:
                print(len(rows), dict(counts), flush=True)


if __name__ == "__main__":
    main()
