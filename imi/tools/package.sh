#!/bin/bash
# The Nunchi folder for a room's computer: the program, Seoul's two files, the model's license and what we changed in
# it, and the setup guide. Copy the folder (or its zip) to the computer and double-click the program.
#   imi/tools/package.sh PROGRAM SEOUL.gguf SEOUL-MM.gguf LFM_LICENSE OUT_DIR
# PROGRAM: nunchi.exe (Windows) or nunchi; the model files are copied in as seoul.gguf and seoul-mm.gguf
set -e
prog="$1"; brain="$2"; mm="$3"; lic="$4"; out="$5"
here="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$out"
cp "$prog" "$out/"
cp "$brain" "$out/seoul.gguf"
cp "$mm" "$out/seoul-mm.gguf"
cp "$lic" "$out/LICENSE-LFM.txt"
cat > "$out/MODEL-CHANGES.txt" <<'TXT'
seoul.gguf and seoul-mm.gguf are Liquid AI's LFM3-Omni-2B, under the LFM Open License v1.0 (LICENSE-LFM.txt),
changed for Nunchi (github.com/iimmyy/nunchi):

- seoul.gguf, the brain: 4-bit (Q4_0), with its 4-bit numbers re-picked so each layer's output on real inputs matches
  the original 16-bit model as closely as the 4-bit grid allows; its word table cut to the 97,496 entries that Korean
  and English text use (of 128,000), the text read and written exactly as before.
- seoul-mm.gguf, the ears, the voice generator and the sound codec: the voice generator's tables in 8-bit (Q8_0).
TXT
sed -e 's/^# //' "$here/docs/sermon-setup.md" > "$out/README.txt"
ls -la "$out"
