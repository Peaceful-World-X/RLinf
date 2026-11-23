#!/usr/bin/env bash
set -euo pipefail
export RELEASE_DIR
RELEASE_DIR="$(python3 -c 'import json,os;from pathlib import Path;print(Path(os.environ["CLOUD_WORK"]).resolve()/json.load(open(os.environ["RELEASE_MANIFEST"]))["release"])')"
export REPO_PATH
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PYTHONPATH="$REPO_PATH"
export OPENPI_DATA_HOME="$RELEASE_DIR/openpi-cache"
export PATH="$RELEASE_DIR/runtime/python/bin:$PATH"
export RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export TRAIN_PY="$RELEASE_DIR/runtime/python/bin/python3.11"
# Rebind wheel-generated console scripts to the restored standalone interpreter.
"$TRAIN_PY" - <<'PY'
import os,sys
from pathlib import Path
for name in ("ray","torchrun","accelerate","hf","huggingface-cli"):
 p=Path(sys.executable).parent/name
 if p.is_file():
  content=p.read_bytes()
  if content.startswith(b"#!"):
   new=b"#!"+os.fsencode(sys.executable)+b"\n"+content.partition(b"\n")[2]
   if content!=new:p.write_bytes(new)
PY
# Some AIStudio images have no matching system FFmpeg. The OpenCV wheel ships
# libav with hashed filenames; torchcodec expects the standard SONAMEs.
if ! "$TRAIN_PY" -c 'from torchcodec.decoders import VideoDecoder' 2>/dev/null; then
  codec_dir="$RELEASE_DIR/ffmpeg"
  mkdir -p "$codec_dir"
  lib_dir="$("$TRAIN_PY" - "$codec_dir" <<'PY'
import pathlib,re,sys,sysconfig
site=pathlib.Path(sysconfig.get_path("purelib"))
wheel=next(p for p in [site/"opencv_python_headless.libs",site/"opencv_python.libs"] if p.is_dir())
for lib in wheel.glob("lib*.so.*"):
    match=re.match(r"lib(avutil|avcodec|avformat|avfilter|swscale|swresample|postproc|avdevice)-.+\.so\.(\d+)",lib.name)
    if match:
        link=pathlib.Path(sys.argv[1])/("lib"+match[1]+".so."+match[2])
        if not link.exists(): link.symlink_to(lib)
print(wheel)
PY
)"
  export LD_LIBRARY_PATH="$codec_dir:$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$TRAIN_PY" -c 'from torchcodec.decoders import VideoDecoder'
fi
