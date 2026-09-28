#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=""
INSTALL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INTERACTIVE="auto"
ARG_COUNT=$#

usage() {
    cat <<'EOF'
Usage: install.sh [--interactive | --non-interactive] [--dir PATH] [--python PATH]

Installs the Intel XPU runtime into an existing ComfyUI workspace .venv.
Host drivers and system packages are left untouched.

With no arguments in a terminal, starts a setup wizard. Use --non-interactive
for unattended installs; existing options also imply non-interactive mode.
EOF
}

while (($#)); do
    case "$1" in
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        --dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || { usage >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$INTERACTIVE" == auto ]]; then
    if ((ARG_COUNT == 0)) && [[ -t 0 && -t 1 ]]; then
        INTERACTIVE=true
    else
        INTERACTIVE=false
    fi
fi

if [[ "$INTERACTIVE" == true ]]; then
    printf '\nComfyUI Intel XPU setup\n'
    printf 'Press Enter to accept each suggested value.\n\n'

    answer=""
    if ! read -r -e -p "ComfyUI workspace [$INSTALL_ROOT]: " answer; then
        printf '\nInstallation cancelled.\n'
        exit 0
    fi
    INSTALL_ROOT="${answer:-$INSTALL_ROOT}"

    if [[ -z "$PYTHON_BIN" ]]; then
        for candidate in python3.13 python3.12 python3.11 python3.10; do
            if command -v "$candidate" >/dev/null 2>&1; then
                PYTHON_BIN="$(command -v "$candidate")"
                break
            fi
        done
    fi
    answer=""
    if ! read -r -e -p "Python executable [${PYTHON_BIN:-python3}]: " answer; then
        printf '\nInstallation cancelled.\n'
        exit 0
    fi
    PYTHON_BIN="${answer:-${PYTHON_BIN:-python3}}"
    printf '\nInstall summary\n'
    printf '  Location: %s\n' "$INSTALL_ROOT"
    printf '  Python:   %s\n' "$PYTHON_BIN"
    printf '  Required: PyTorch 2.11.0 XPU, torchvision 0.26.0 XPU, torchaudio 2.11.0 XPU, ComfyUI requirements.txt, and ComfyUI Manager\n'
    printf '  Add-ons:  none (no custom nodes will be installed)\n\n'
    answer=""
    if ! read -r -p 'Continue with installation? [y/N] ' answer; then
        printf '\nInstallation cancelled.\n'
        exit 0
    fi
    case "$answer" in
        y|Y|yes|YES|Yes) ;;
        *) printf 'Installation cancelled; no install files were changed.\n'; exit 0 ;;
    esac
fi

fail() {
    printf 'Intel ComfyUI setup: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == Linux ]] || fail "this installer currently supports Linux only."
[[ "$(uname -m)" == x86_64 ]] || fail "the XPU wheel set currently targets Linux x86_64."
[[ -f "$INSTALL_ROOT/main.py" && -f "$INSTALL_ROOT/requirements.txt" ]] || fail "--dir must point to an existing ComfyUI workspace."

intel_gpu_found=false
for vendor_file in /sys/class/drm/card*/device/vendor; do
    [[ -r "$vendor_file" ]] || continue
    if [[ "$(<"$vendor_file")" == 0x8086 ]]; then
        intel_gpu_found=true
        break
    fi
done
[[ "$intel_gpu_found" == true ]] || fail "no Intel GPU was found under /sys/class/drm. Check host driver/device access first."

if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.13 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || fail "Python 3.10–3.13 is required. No system Python will be installed automatically."
python_version="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$python_version" in
    3.10|3.11|3.12|3.13) ;;
    *) fail "Python $python_version is outside the supported 3.10–3.13 range." ;;
esac

INSTALL_ROOT="$(cd -- "$INSTALL_ROOT" && pwd)"
COMFYUI_DIR="$INSTALL_ROOT"
VENV_DIR="$INSTALL_ROOT/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip

# This is the locally exercised baseline for the Arc B580 workspace. ComfyUI
# requirements are constrained below so they cannot replace the XPU wheels.
"$VENV_PYTHON" -m pip install \
    --index-url https://download.pytorch.org/whl/xpu \
    'torch==2.11.0+xpu' \
    'torchvision==0.26.0+xpu' \
    'torchaudio==2.11.0+xpu'
"$VENV_PYTHON" -m pip install \
    --constraint "$SCRIPT_DIR/xpu-constraints.txt" \
    --requirement "$COMFYUI_DIR/requirements.txt" \
    comfyui-manager

printf 'comfyui-intel-xpu\n' > "$COMFYUI_DIR/.comfy_environment"
"$VENV_PYTHON" "$SCRIPT_DIR/system_check.py"
printf '\nInstalled in %s\nStart from that directory with:\n.venv/bin/python main.py --enable-manager --disable-api-nodes --port 8188\n' "$INSTALL_ROOT"
