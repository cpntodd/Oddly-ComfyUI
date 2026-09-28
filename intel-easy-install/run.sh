#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_DIR="$INSTALL_ROOT"
PYTHON="$INSTALL_ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || { printf 'Missing environment: %s\nRun install.sh first.\n' "$PYTHON" >&2; exit 1; }
[[ -f "$COMFYUI_DIR/main.py" ]] || { printf 'Missing ComfyUI checkout: %s\n' "$COMFYUI_DIR" >&2; exit 1; }

cd "$COMFYUI_DIR"
exec "$PYTHON" main.py "$@"
