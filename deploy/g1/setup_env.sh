#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
set -euo pipefail
G1_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
G1_ENV="$G1_ROOT/.venv-g1"
for command in uv git cmake g++; do command -v "$command" >/dev/null; done
if [[ ! -x "$G1_ENV/bin/python" ]]; then uv venv --python 3.11 "$G1_ENV"; fi
mkdir -p "$G1_ENV/vendor"

# 固定 DDS 版本；只安装到客户端环境，不改系统库或训练环境。
G1_DDS="$G1_ENV/vendor/cyclonedds"
if [[ ! -f "$G1_DDS/install/lib/libddsc.so" ]]; then
    if [[ ! -d "$G1_DDS/.git" ]]; then
        git clone --depth 1 --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git "$G1_DDS"
    fi
    cmake -S "$G1_DDS" -B "$G1_DDS/build" -DCMAKE_INSTALL_PREFIX="$G1_DDS/install" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
    cmake --build "$G1_DDS/build" --target install -j4
fi
export CYCLONEDDS_HOME="$G1_DDS/install"  # Unitree SDK 的 DDS 编译与加载目录。

# 旧压缩包内 SDK 缺 G1/Dex1 消息；固定官方版本，editable 保留 CRC 动态库。
G1_SDK="$G1_ENV/vendor/unitree_sdk2_python_current"
G1_SDK_REV=814556d15970dd2ecf1c9984e845ca02ab07e206
if [[ ! -d "$G1_SDK/.git" ]]; then
    git init "$G1_SDK"
    git -C "$G1_SDK" remote add origin https://github.com/unitreerobotics/unitree_sdk2_python.git
    git -C "$G1_SDK" fetch --depth 1 origin "$G1_SDK_REV"
    git -C "$G1_SDK" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$G1_SDK" rev-parse HEAD) == "$G1_SDK_REV" ]] || { echo 'G1 SDK revision differs' >&2; exit 1; }
uv pip install --python "$G1_ENV/bin/python" -r "$G1_ROOT/deploy/g1/requirements.in" -e "$G1_SDK"
"$G1_ENV/bin/python" -c 'import pinocchio, pyrealsense2; from unitree_sdk2py.utils.crc import CRC; from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_; CRC()'
