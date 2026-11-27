"""Rebuild the recorded RGB banks, or audit a transferred copy without re-rendering."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from common import DATA, HERE, frame, goal_indices, start_episode
from dataset import MixedDataset, resize_rgb
from player import GAME_WAD, MAP_WAD, HangarPlayer, ROUTE, make_game
from sampling import file_hash

REF = HERE / "reference"
BANK = DATA / "e1m1_mixed_rgb"
WADS = {
    GAME_WAD: "1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771",
    MAP_WAD: "2ee20cc35794d6a26995d70e001a856387b570c49e79d8e28a08235a22728aee",
}


def require_hash(path, digest):
    if file_hash(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def auto(row, seed, bank, tmp):
    game = make_game(seed)
    player = HangarPlayer(seed)
    actions, nodes = [], []
    try:
        start_episode(game, seed, tmp / "demo.lmp")
        while not game.is_episode_finished():
            action = player.action(game.get_state())
            actions.append(action)
            nodes.append(player.node)
            game.make_action(action, 4)

        def check_end():
            if row["source"] == "death":
                assert game.is_player_dead()
            else:
                assert game.get_total_reward() > 0.5 and not game.is_player_dead()

        check_end()
        game.replay_episode(str(tmp / "demo.lmp"))
        i = 0
        while not game.is_episode_finished():
            bank[i, :, 24:168] = resize_rgb(frame(game.get_state())[0])
            game.advance_action(1)
            i += 1
        assert i == len(bank)
        check_end()
    finally:
        game.close()
    if row["source"] == "success":
        with np.load(REF / "goal_reference.npz") as ref:
            progress = np.asarray(nodes, np.float32) / (len(ROUTE) - 1)
            goals = goal_indices(progress, ref["progress"], ref["action_tics"]).astype(np.int32)
        actions = np.asarray(actions, np.uint8)
        assert hashlib.sha256(actions.tobytes() + goals.tobytes()).hexdigest() == row["label_digest"]
        np.savez(tmp / row["labels"], actions=actions, goals=goals)


def human(path, bank):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) != 35:
            raise ValueError(f"Expected 35 fps video: {path}")
        for dest in bank:
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Truncated: {path}")
            dest[:, 24:168] = resize_rgb(bgr[:, :, ::-1])
        if cap.read()[0]:
            raise ValueError(f"Extra frames: {path}")
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--videos", type=Path, default=DATA / "human")
    args = p.parse_args()
    videos = args.videos.resolve()
    manifest = (REF / "manifest.json").read_bytes()
    sources = json.loads((REF / "sources.json").read_text())
    hashes = json.loads((REF / "bank_sha256.json").read_text())
    BANK.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    with (BANK / ".prepare.lock").open("a") as lock, tempfile.TemporaryDirectory() as runtime:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.chdir(runtime)
        if not args.verify:
            for path, digest in WADS.items():
                require_hash(path, digest)
        for row in json.loads(manifest)["episodes"]:
            key = row["id"]
            out = BANK / row["bank"]
            if args.verify or (out.exists() and ("labels" not in row or (BANK / row["labels"]).exists())):
                require_hash(out, hashes[key])
                continue
            with tempfile.TemporaryDirectory(dir=BANK) as staging:
                tmp = Path(staging)
                dest = tmp / out.name
                bank = np.lib.format.open_memmap(
                    dest, mode="w+", dtype=np.uint8, shape=(row["frames"], 3, 192, 192)
                )
                if row["source"] == "human":
                    human(videos / f"{sources[key]['id']}.mp4", bank)
                else:
                    auto(row, sources[key]["seed"], bank, tmp)
                bank.flush()
                del bank
                require_hash(dest, hashes[key])
                if "labels" in row:
                    (tmp / row["labels"]).replace(BANK / row["labels"])
                dest.replace(out)
            print(key, flush=True)
        if args.verify:
            require_hash(BANK / "manifest.json", hashlib.sha256(manifest).hexdigest())
        else:
            temp = BANK / ".manifest.tmp"
            temp.write_bytes(manifest)
            temp.replace(BANK / "manifest.json")
        data = MixedDataset(BANK / "manifest.json", 192)
        assert len(data.train) == 239 and len(data.dev) == 15
        assert len(data.rows(data.train, labels=True)) == 11807
        assert len(data.rows(data.dev, labels=True)) == 530
        print("Verified 239 banks; 11,807 train and 530 validation action labels.")


if __name__ == "__main__":
    main()
