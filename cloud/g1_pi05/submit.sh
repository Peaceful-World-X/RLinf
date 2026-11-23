#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python="${SUBMIT_PYTHON:-python3}"
for arg in "$@"; do
  if [[ "$arg" == --dry-run || "$arg" == --help ]]; then exec "$python" submit.py "$@"; fi
done
sdk_ok() { "$1" -c 'from pypai.conf import ExecConf, KMConf, NasStoreConf; from pypai.job import PythonJobBuilder' >/dev/null 2>&1; }
if ! sdk_ok "$python"; then
  venv="${SUBMIT_VENV:-$HOME/.venvs/rlinf-aistudio}"
  if [[ ! -x "$venv/bin/python" ]]; then "$python" -m venv "$venv"; fi
  python="$venv/bin/python"
  if ! sdk_ok "$python"; then
    index="${SUBMIT_INDEX:-https://pypi.antfin-inc.com/simple/}"
    "$python" -m pip install --index-url "$index" aii-pypai
    "$python" -m pip uninstall -y couler
    "$python" -m pip install --index-url "$index" --force-reinstall couler-core ant-couler
    sdk_ok "$python"
  fi
fi
exec "$python" submit.py "$@"
