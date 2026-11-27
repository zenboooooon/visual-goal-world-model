"""Import the original seed-17 AdamW checkpoint at exactly 25k without re-training."""

import argparse
import fcntl
from pathlib import Path

from common import BUTTONS, HERE, json_write
from experiment import RUNS, load, save
from sampling import file_hash

ORIGINAL = "f596b7562588e4ea1c4954826c4582d88620226558f04e062ea3958f8a625107"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    path = parser.parse_args().checkpoint
    if file_hash(path) != ORIGINAL:
        raise ValueError("Expected the original resume_at_25000.pt including AdamW and sampler state")
    data = load(path)
    fingerprint = file_hash(HERE / "reference/manifest.json")
    if data["step"] != 25000 or data["config"]["dataset"] != fingerprint:
        raise ValueError("Incorrect parent experiment")
    run = RUNS / "base"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((run / f).exists() for f in ("resume.pt", "exact.pt", "previous.pt")):
            raise FileExistsError("Import into a fresh E1M1_RUNS directory")
        state = {
            k: data[k]
            for k in (
                "step",
                "policy",
                "optimizer",
                "best_score",
                "best_step",
                "window_positions",
                "transition_position",
                "action_position",
            )
        }
        state.update(phase="base", dataset=fingerprint, best=data["best"]["policy"])
        save(run / "resume.pt", state)
        save(run / "best.pt", dict(step=data["best_step"], policy=state["best"]))
        json_write(run / "validation_action.json", dict(step=25000, bce=0.9028775772958432))
        json_write(run / "progress.json", dict(phase="base", step=25000, train_wm=0.37781, dev_wm=0.66967))
        save(run / "exact.pt", dict(step=25000, policy=data["policy"], buttons=list(BUTTONS)))


if __name__ == "__main__":
    main()
