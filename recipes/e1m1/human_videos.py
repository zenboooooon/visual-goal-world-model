"""Fetch the pinned DSDA demo members and render the original 35 Hz input videos."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

from common import DATA
from prepare import REF, WADS, require_hash
from player import GAME_WAD, MAP_WAD


def render(row, src, args):
    dest = args.out / f"{src['id']}.mp4"
    if dest.exists():
        require_hash(dest, row["sha256"])
        return
    with tempfile.TemporaryDirectory(dir=args.out) as tmpdir:
        tmp = Path(tmpdir)
        archive = tmp / "input.zip"
        local = args.archives / f"{src['id']}.zip" if args.archives else None
        if local and local.is_file():
            archive = local
        else:
            with (
                urlopen(
                    Request(src["url"], headers={"User-Agent": "E1M1-reproduction/1.0"}), timeout=90
                ) as r,
                archive.open("wb") as f,
            ):
                shutil.copyfileobj(r, f)
        with ZipFile(archive) as z:
            demo = z.read(src["source_member"])
        if hashlib.sha256(demo).hexdigest() != src["source_lmp_sha256"]:
            raise ValueError("LMP SHA-256 mismatch")
        (tmp / "replay.lmp").write_bytes(demo)
        capture = "ffmpeg -v error -f rawvideo -pixel_format rgb24 -framerate %r -video_size %wx%h -i - -an -filter_threads 1 -vf scale=320:240:flags=area -c:v libx264 -preset veryfast -crf 17 -threads 2 -pix_fmt yuv420p -movflags +faststart -y capture.mp4"
        settings = dict(
            cap_soundcommand='"cat > /dev/null"',
            cap_videocommand=f'"{capture}"',
            cap_muxcommand='"true"',
            cap_remove_tempfiles=0,
            cap_fps=35,
            hudadd_demoprogressbar=0,
            render_wipescreen=0,
            uncapped_framerate=0,
            screen_resolution='"320x240"',
            show_messages=0,
        )
        cfg = (REF / "dsda.cfg").read_text()
        for key, value in settings.items():
            cfg, count = re.subn(r"(?m)^" + key + r"[ \t]+.*$", f"{key} {value}", cfg)
            if count != 1:
                raise ValueError(f"Missing setting {key}")
        (tmp / "render.cfg").write_text(cfg)
        (tmp / "ffmpeg").symlink_to(args.ffmpeg)
        env = dict(
            os.environ,
            PATH=str(tmp) + os.pathsep + os.environ["PATH"],
            SDL_VIDEODRIVER="dummy",
            SDL_RENDER_DRIVER="software",
            SDL_AUDIODRIVER="dummy",
        )
        level = ["-file", str(MAP_WAD)] if src["geometry"] == "romero_DoomMapsSE" else []
        cmd = [
            args.dsda,
            "-iwad",
            GAME_WAD,
            *level,
            "-complevel",
            str(src["complevel"]),
            "-timedemo",
            "replay.lmp",
            "-levelstat",
            "-analysis",
            "-viddump",
            "discard.mkv",
            "-nomusic",
            "-nosfx",
            "-config",
            "render.cfg",
        ]
        subprocess.run(cmd, cwd=tmp, env=env, check=True, timeout=600, stdout=subprocess.DEVNULL)
        stat = (tmp / "levelstat.txt").read_text()
        match = re.search(r"E1M1 - (\d+:\d+\.\d+) .*?K:\s*(\d+)/(\d+).*?S:\s*(\d+)/(\d+)", stat)
        if (
            not match
            or match[1] != src["dsda_time"]
            or [int(x) for x in match.groups()[1:]]
            != [src[k] for k in ("kills", "total_kills", "secrets", "total_secrets")]
        ):
            raise ValueError(f"Replay desynchronized: {stat}")
        movie = tmp / "movie.mp4"
        subprocess.run(
            [
                args.ffmpeg,
                "-v",
                "error",
                "-i",
                tmp / "capture.mp4",
                "-frames:v",
                str(row["frames"]),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-crf",
                "16",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                movie,
            ],
            check=True,
        )
        require_hash(movie, row["sha256"])
        movie.replace(dest)
        print(dest.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, help="Optional cache of original <id>.zip files")
    p.add_argument("--out", type=Path, default=DATA / "human")
    p.add_argument("--dsda", default="dsda-doom")
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--id", type=int, help="Render only one DSDA id")
    args = p.parse_args()
    if args.ffmpeg is None:
        import imageio_ffmpeg

        args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.ffmpeg = Path(shutil.which(args.ffmpeg) or args.ffmpeg).resolve(strict=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for path, digest in WADS.items():
        require_hash(path, digest)
    sources = json.loads((REF / "sources.json").read_text())
    for row in json.loads((REF / "manifest.json").read_text())["episodes"]:
        src = sources[row["id"]]
        if row["source"] == "human" and args.id in (None, src["id"]):
            render(row, src, args)
