#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
dest=$(realpath -m "${E1M1_DATA:-$PWD/data}/e1m1_assets")
mkdir -p "$dest"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
python3 - <<'PY'
from urllib.request import urlretrieve
urlretrieve("https://deb.debian.org/debian/pool/non-free/d/doom-wad-shareware/doom-wad-shareware_1.9.fixed-4_all.deb","shareware.deb")
urlretrieve("https://raw.githubusercontent.com/TCRF/old-doom-stuff/bfc76f379e6967ae82007b6e61d256bf520ab6a1/DoomMapsSE/e1m1.dwd","E1M1.dwd")
PY
echo '6215bb9c694eef260bd6a857df2310bc53863c6b6f74605948d2b92cb36dd483  E1M1.dwd' | sha256sum -c -
dpkg-deb -x shareware.deb pkg
git clone -q https://github.com/teefoss/doombsp.git bsp
git -C bsp checkout -q 01969e4508a40e69b735df3362da308a0abb3d96
g++ -std=gnu++11 -include cstdint -w bsp/*.cpp -o doombsp -lSDL2
./doombsp E1M1.dwd .
echo '2ee20cc35794d6a26995d70e001a856387b570c49e79d8e28a08235a22728aee  E1M1.wad' | sha256sum -c -
echo '1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771  pkg/usr/share/games/doom/doom1.wad' | sha256sum -c -
cp E1M1.wad pkg/usr/share/games/doom/doom1.wad "$dest/"
