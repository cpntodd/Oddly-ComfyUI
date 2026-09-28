#!/usr/bin/env python3

import glob
from importlib.metadata import PackageNotFoundError, version
import sys
from pathlib import Path


def intel_gpus():
    devices = []
    for vendor_path in glob.glob("/sys/class/drm/card[0-9]*/device/vendor"):
        try:
            if Path(vendor_path).read_text(encoding="ascii").strip() != "0x8086":
                continue
            device_path = vendor_path.rsplit("/", 1)[0]
            device_id = Path(device_path, "device").read_text(encoding="ascii").strip()
            devices.append(f"{device_path.rsplit('/', 2)[-2]} ({device_id})")
        except OSError:
            continue
    return devices


def main():
    print(f"Python: {sys.version.split()[0]}")
    devices = intel_gpus()
    print("Intel display devices: " + (", ".join(devices) if devices else "none detected"))

    try:
        import torch
    except ImportError as error:
        print(f"PyTorch import failed: {error}", file=sys.stderr)
        return 1

    print(f"PyTorch: {torch.__version__}")
    expected = {
        "torch": "2.11.0+xpu",
        "torchvision": "0.26.0+xpu",
        "torchaudio": "2.11.0+xpu",
    }
    try:
        installed = {package: version(package) for package in expected}
    except PackageNotFoundError as error:
        print(f"Missing XPU package: {error}", file=sys.stderr)
        return 1
    for package, expected_version in expected.items():
        print(f"{package}: {installed[package]}")
        if installed[package] != expected_version:
            print(f"Expected {package} {expected_version}.", file=sys.stderr)
            return 1

    try:
        xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception as error:
        print(f"PyTorch XPU initialization failed: {error}", file=sys.stderr)
        return 1
    if not xpu_available:
        print("PyTorch XPU is unavailable. Check Intel GPU drivers and device access.", file=sys.stderr)
        return 1

    try:
        value = (torch.ones((128, 128), device="xpu") * 2).sum().cpu().item()
    except Exception as error:
        print(f"XPU arithmetic failed: {error}", file=sys.stderr)
        return 1
    if value != 32768:
        print(f"XPU arithmetic returned an unexpected result: {value}", file=sys.stderr)
        return 1

    print(f"XPU: {torch.xpu.get_device_name(0)}")
    print("XPU arithmetic: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
