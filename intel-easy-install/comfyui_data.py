#!/usr/bin/env python3

import argparse
import datetime
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile


DATA_DIRS = ("user", "input", "output")
MODEL_DIRS = (
    "checkpoints", "configs", "loras", "vae", "text_encoders", "clip",
    "clip_vision", "diffusion_models", "unet", "controlnet", "t2i_adapter",
    "style_models", "embeddings", "diffusers", "vae_approx", "gligen",
    "upscale_models", "latent_upscale_models", "hypernetworks", "photomaker",
    "classifiers", "model_patches", "audio_encoders", "background_removal",
    "frame_interpolation", "geometry_estimation", "optical_flow", "detection",
)
ALLOWED_TOP_LEVEL = set(DATA_DIRS) | {"extra_model_paths.yaml", "intel-model-paths.yaml", "models"}


def comfy_dir_from(args):
    path = Path(args.comfyui).expanduser() if args.comfyui else Path(__file__).resolve().parent.parent
    path = path.resolve()
    if not (path / "main.py").is_file():
        raise SystemExit(f"Not a ComfyUI directory: {path}")
    return path


def backup(args):
    comfy = comfy_dir_from(args)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.destination:
        destination = Path(args.destination).expanduser().resolve()
    else:
        destination = comfy / "backups"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"comfyui-data-{stamp}.tar.gz"
    members = [comfy / name for name in DATA_DIRS]
    members.extend(comfy / name for name in ("extra_model_paths.yaml", "intel-model-paths.yaml") if (comfy / name).is_file())
    if args.include_models:
        members.append(comfy / "models")
    with tarfile.open(archive, "w:gz") as tar:
        for member in members:
            if member.exists():
                tar.add(member, arcname=member.relative_to(comfy))
    print(archive)


def validate_archive_member(member):
    name = PurePosixPath(member.name)
    if name.is_absolute() or not name.parts or ".." in name.parts:
        raise ValueError(f"Unsafe archive path: {member.name}")
    if name.parts[0] not in ALLOWED_TOP_LEVEL:
        raise ValueError(f"Archive contains unsupported path: {member.name}")
    if not (member.isdir() or member.isfile()):
        raise ValueError(f"Archive contains unsupported file type: {member.name}")
    if name.parts[0] == "models" and not member.isdir() and not member.isfile():
        raise ValueError(f"Unsupported model archive entry: {member.name}")


def copy_restore_tree(source, target):
    for path in source.iterdir():
        destination = target / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def restore(args):
    comfy = comfy_dir_from(args)
    archive = Path(args.archive).expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"Backup not found: {archive}")

    with tempfile.TemporaryDirectory(prefix="comfyui-restore-") as staging:
        staging_dir = Path(staging)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                validate_archive_member(member)
                if member.name.split("/", 1)[0] == "models" and not args.include_models:
                    continue
                target = staging_dir.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError(f"Could not read backup file: {member.name}")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        if not any(staging_dir.iterdir()):
            raise SystemExit("Backup contains no restorable files.")

        safety_dir = comfy / "backups"
        safety_path = safety_dir / ("before-restore-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".tar.gz")
        safety_dir.mkdir(parents=True, exist_ok=True)
        selected = [comfy / name for name in DATA_DIRS]
        selected.extend(comfy / name for name in ("extra_model_paths.yaml", "intel-model-paths.yaml") if (comfy / name).is_file())
        if args.include_models:
            selected.append(comfy / "models")
        with tarfile.open(safety_path, "w:gz") as tar:
            for member in selected:
                if member.exists():
                    tar.add(member, arcname=member.relative_to(comfy))
        copy_restore_tree(staging_dir, comfy)
    print(f"Restored from {archive}")
    print(f"Current data backup: {safety_path}")


def link_models(args):
    comfy = comfy_dir_from(args)
    source = Path(args.models).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Model directory not found: {source}")
    config_path = comfy / "extra_model_paths.yaml"
    if config_path.exists():
        raise SystemExit(f"Refusing to overwrite existing config: {config_path}")
    paths = {name: f"{name}/" for name in MODEL_DIRS}
    config = {"intel_shared_models": {"base_path": str(source), **paths}}
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path}; ComfyUI will load it on startup.")


def parser():
    result = argparse.ArgumentParser(description="Manage portable ComfyUI data and shared model paths.")
    subparsers = result.add_subparsers(required=True)

    backup_parser = subparsers.add_parser("backup", help="Back up user, input, and output data.")
    backup_parser.add_argument("--comfyui")
    backup_parser.add_argument("--destination")
    backup_parser.add_argument("--include-models", action="store_true")
    backup_parser.set_defaults(func=backup)

    restore_parser = subparsers.add_parser("restore", help="Restore a backup and save current data first.")
    restore_parser.add_argument("archive")
    restore_parser.add_argument("--comfyui")
    restore_parser.add_argument("--include-models", action="store_true")
    restore_parser.set_defaults(func=restore)

    models_parser = subparsers.add_parser("link-models", help="Create config for an existing shared models directory.")
    models_parser.add_argument("models")
    models_parser.add_argument("--comfyui")
    models_parser.set_defaults(func=link_models)
    return result


if __name__ == "__main__":
    try:
        options = parser().parse_args()
        options.func(options)
    except (OSError, tarfile.TarError, ValueError) as error:
        print(f"ComfyUI data operation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
