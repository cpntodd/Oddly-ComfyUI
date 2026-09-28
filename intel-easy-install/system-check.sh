#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python"
[[ -x "$PYTHON" ]] || { printf 'Run install.sh before checking the environment.\n' >&2; exit 1; }
exec "$PYTHON" "$SCRIPT_DIR/system_check.py"
