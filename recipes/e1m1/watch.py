"""Run the E1M1 recipe sequentially under a persistent systemd user timer."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import BadZipFile, ZipFile

from common import DATA, HERE, json_write
from experiment import BUDGET, RUNS, load
from sampling import file_hash

UNIT = "e1m1-seed17-" + hashlib.sha256(os.fsencode(RUNS)).hexdigest()[:10]
WORKER = UNIT + ".service"
TIMER = UNIT + "-watch.timer"
PAUSE = RUNS / "paused"
STATE = RUNS / "watch.json"
LOG = RUNS / "worker.log"


def systemctl(*args, check=True):
    return subprocess.run(
        ["systemctl", "--user", *args], check=check, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def task():
    for phase in BUDGET:
        if not (RUNS / phase / "exact.pt").exists():
            return phase
        if phase == "base" and read(RUNS / "eval25000/summary.json").get("starts") != 100:
            return "evaluate"
    return "complete"


def worker():
    if PAUSE.exists():
        return
    with LOG.open("a", buffering=1) as out:
        while not PAUSE.exists():
            phase = task()
            if phase == "complete":
                return
            args = ["experiment.py", phase]
            if phase == "evaluate":
                args = [
                    "evaluate.py",
                    "--checkpoint",
                    str(RUNS / "base/exact.pt"),
                    "--out",
                    str(RUNS / "eval25000"),
                ]
            print(datetime.now().isoformat(), args, file=out, flush=True)
            subprocess.run(
                [sys.executable, "-u", *args], cwd=HERE, stdout=out, stderr=subprocess.STDOUT, check=True
            )


def valid(path, phase):
    try:
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
        state = load(path)
        step = state["step"]
        return (
            state["phase"] == phase
            and state["dataset"] == file_hash(HERE / "reference/manifest.json")
            and 0 < step <= BUDGET[phase]
            and bool(state["policy"])
            and bool(state["optimizer"]["state"])
            and all(state["window_positions"][h] == 2 * step for h in range(1, 11))
            and state["action_position"] == 128 * step
            and state["transition_position"] == 20 * step
        )
    except (OSError, ValueError, RuntimeError, KeyError, EOFError, TypeError, BadZipFile):
        return False


def recover(phase):
    directory = RUNS / phase
    last, previous = directory / "resume.pt", directory / "previous.pt"
    if not last.exists() and not previous.exists():
        if (directory / "progress.json").exists():
            raise RuntimeError("Progress exists but both checkpoints are missing")
        return
    if last.exists() and valid(last, phase):
        return
    if not valid(previous, phase):
        raise RuntimeError("Missing or damaged checkpoint; no verified previous checkpoint")
    if last.exists():
        os.link(last, directory / f"damaged-{time.time_ns()}.pt")
    tmp = directory / "recovery.tmp"
    shutil.copyfile(previous, tmp)
    os.replace(tmp, last)


def check():
    now = time.time()
    phase = task()
    progress = read(RUNS / phase / "progress.json")
    step = progress.get("step", 0)
    if phase == "evaluate":
        step = read(RUNS / "eval25000/summary.json").get("starts", 0)
    state = read(STATE)
    same = state.get("position") == [phase, step]
    since = state.get("since", now) if same else now
    retries = state.get("retries", 0) if same else 0
    status = systemctl("show", "-p", "ActiveState", "--value", WORKER)
    result = systemctl("show", "-p", "Result", "--value", WORKER)
    tail = ""
    if LOG.exists():
        with LOG.open("rb") as log:
            log.seek(max(0, LOG.stat().st_size - 4096))
            tail = log.read().decode(errors="replace")
    active = status in {"active", "activating", "deactivating"}
    if PAUSE.exists() or phase == "complete":
        action = "paused" if PAUSE.exists() else "complete"
    elif active and now - since < 3600:
        action = "running"
    elif retries >= 2:
        action = "repeated failure at the same position; inspect worker.log and journal"
    else:
        try:
            if active:
                systemctl("stop", WORKER)
            if "out of memory" in tail.lower() or result == "oom-kill":
                import torch

                if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 2**30:
                    raise RuntimeError("waiting for GPU memory")
            if phase != "evaluate":
                recover(phase)
            systemctl("reset-failed", WORKER, check=False)
            systemctl("start", "--no-block", WORKER)
            retries += 1
            since = now
            action = "started from verified state" if step else "started"
        except (RuntimeError, OSError, subprocess.CalledProcessError) as err:
            action = str(err)
    json_write(
        STATE,
        dict(
            position=[phase, step],
            since=since,
            retries=retries,
            checked=now,
            service=status,
            result=result,
            action=action,
            log_tail=tail,
        ),
    )
    lines = [
        "# E1M1 seed 17",
        "",
        f"{datetime.now().astimezone():%F %T %Z}  {phase} · {action}",
        "",
        "|Phase|Step|Train WM|Dev WM|Dev action BCE (step)|",
        "|---|---:|---:|---:|---:|",
    ]
    for name in BUDGET:
        row = read(RUNS / name / "progress.json")
        val = read(RUNS / name / "validation_action.json")
        if row:
            lines.append(
                f"|{name}|{row['step']}|{row['train_wm']:.5f}|{row['dev_wm']:.5f}|"
                + (f"{val['bce']:.5f} ({val['step']})" if val else "—")
                + "|"
            )
    (RUNS / "progress.md").write_text("\n".join(lines) + "\n")
    print(phase, step, action)


def unit_text(value):
    if "\n" in value or "\r" in value:
        raise ValueError("Unit paths must be single-line")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def install():
    directory = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(E1M1_RUNS=str(RUNS), E1M1_DATA=str(DATA))
    for key in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]
    settings = "".join("Environment=" + unit_text(k + "=" + v) + "\n" for k, v in env.items())
    for name, mode, kind in ((WORKER, "worker", "simple"), (UNIT + "-watch.service", "check", "oneshot")):
        cmd = " ".join(unit_text(str(s)) for s in [sys.executable, Path(__file__).resolve(), mode])
        (directory / name).write_text(
            f"[Unit]\nDescription=E1M1 {mode}\n[Service]\nType={kind}\n" + settings + f"ExecStart={cmd}\n"
        )
    (directory / TIMER).write_text(
        "[Unit]\nDescription=Check E1M1 progress every 15 minutes\n"
        "[Timer]\nOnCalendar=*:0/15\nPersistent=true\n[Install]\nWantedBy=timers.target\n"
    )
    systemctl("daemon-reload")
    if PAUSE.exists():
        state = read(STATE)
        if state:
            state.update(retries=0, since=time.time())
            json_write(STATE, state)
        PAUSE.unlink()
    systemctl("enable", "--now", TIMER)
    systemctl("start", UNIT + "-watch.service")
    print(systemctl("list-timers", TIMER))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "pause", "check", "worker"))
    command = parser.parse_args().command
    RUNS.mkdir(parents=True, exist_ok=True)
    if command == "install":
        install()
    elif command == "pause":
        PAUSE.touch()
        systemctl("disable", "--now", TIMER)
        systemctl("stop", UNIT + "-watch.service", WORKER)
    elif command == "check":
        check()
    else:
        worker()
