# ComfyUI Intel setup

These tools manage the ComfyUI workspace containing this folder. ComfyUI, custom nodes, models, workflows, and the Intel runtime use one checkout and its root `.venv`; setup does not clone another installation.

## Run

From `/mnt/Data/Projects/ComfyUI-master`:

```bash
.venv/bin/python main.py --enable-manager --disable-api-nodes --port 8188
```

Open http://127.0.0.1:8188. The installed PyTorch XPU runtime selects the Intel GPU automatically. `intel-easy-install/run.sh` also launches this same workspace and passes arguments to `main.py`.

## Set up or check the runtime

```bash
./intel-easy-install/install.sh
./intel-easy-install/system-check.sh
```

Interactive setup shows the destination and dependencies before confirmation. It defaults to this workspace; `--dir PATH` selects another existing ComfyUI workspace. `--python PATH` selects Python for creating a missing environment. `--non-interactive` skips prompts. Existing source files and custom nodes are preserved.

Setup installs PyTorch 2.11.0+xpu, torchvision 0.26.0+xpu, torchaudio 2.11.0+xpu, ComfyUI requirements under those constraints, and ComfyUI Manager. No custom nodes are installed. It checks for an Intel GPU first and verifies XPU arithmetic afterward. Host drivers and system packages are not modified. Linux x86_64 and Python 3.10–3.13 are supported by this installer.

## Shared models and backups

```bash
.venv/bin/python intel-easy-install/comfyui_data.py link-models /mnt/Data/Models
.venv/bin/python intel-easy-install/comfyui_data.py backup
.venv/bin/python intel-easy-install/comfyui_data.py restore backups/comfyui-data-TIMESTAMP.tar.gz
```

The models directory contains category folders such as `checkpoints/` and `loras/`. Linking writes the standard `extra_model_paths.yaml`, loaded by the direct launch command too; existing configuration is never overwritten.

Backups live in the workspace `backups/` directory and include user, input, output, and model-path configuration. Models require `--include-models` for backup and restore. Restore makes a safety backup before merging and rejects unsafe archive paths and symlinks.

## Credits

Portable setup and data-management ideas were inspired by Tavris1/ComfyUI-Easy-Install. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
