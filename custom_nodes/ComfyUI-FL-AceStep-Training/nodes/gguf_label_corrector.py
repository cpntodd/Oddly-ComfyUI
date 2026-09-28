"""GGUF text-model support for post-processing ACE-Step labels."""

import json
import logging
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

logger = logging.getLogger("FL_AceStep_Training")

DEFAULT_MODEL = "/mnt/Data/Models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_DIRECTORY = str(Path(DEFAULT_MODEL).parent)
DEFAULT_RUNTIME = (
    "/mnt/Data/Projects/PrismML-Arc/llama.cpp/build-intel-all/bin/llama-server"
)


def _discover_gguf_models(directory):
    model_directory = Path(directory).expanduser()
    if not model_directory.is_dir():
        return []
    return sorted(
        path.name
        for path in model_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".gguf"
    )


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _release_comfy_models():
    if not model_management:
        return
    try:
        model_management.unload_all_models()
        model_management.cleanup_models_gc()
    except Exception as error:
        logger.warning("Could not fully release ComfyUI models: %s", error)


class GGUFTextModel:
    def __init__(self, model_path, runtime_path, context_size, gpu_layers, reasoning="auto"):
        model = Path(model_path).expanduser().resolve()
        runtime = Path(runtime_path).expanduser().resolve()
        if not model.is_file() or model.suffix.lower() != ".gguf":
            raise ValueError(f"GGUF model not found: {model}")
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise ValueError(f"llama-server executable not found: {runtime}")

        self.model_path = model
        self.runtime_path = runtime
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        if int(gpu_layers) > 0:
            _release_comfy_models()
        runtime_env = os.environ.copy()

        command = [
            str(runtime),
            "--model", str(model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(max(1024, int(context_size))),
            "--parallel", "1",
            "--gpu-layers", str(max(0, int(gpu_layers))),
            "--flash-attn", "off",
            "--batch-size", "128",
            "--ubatch-size", "128",
            "--jinja",
            "--reasoning", reasoning.lower(),
            "--no-webui",
        ]
        if int(gpu_layers) == 0:
            # The Intel build otherwise selects OpenVINO automatically. Its
            # dynamic KV-cache path is unstable for this correction workload.
            device_args = ["--device", "none"]
        else:
            # Keep GPU selection on the tested Intel SYCL backend. Other
            # backends can override this without changing the node UI.
            device_args = [
                "--device",
                os.environ.get("FL_ACESTEP_GGUF_GPU_DEVICE", "SYCL0"),
            ]
        command[command.index("--gpu-layers") + 2:command.index("--gpu-layers") + 2] = device_args
        setvars = Path("/opt/intel/oneapi/setvars.sh")
        if setvars.is_file():
            command = [
                "bash",
                "-lc",
                f"source {setvars} >/dev/null 2>&1 && exec \"$@\"",
                "llama-server",
                *command,
            ]
        logger.info("Starting GGUF correction model: %s", model.name)
        self.log_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            env=runtime_env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_ready()

    def _wait_until_ready(self):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_file.flush()
                self.log_file.seek(0)
                details = self.log_file.read()[-4000:].strip()
                self.close()
                raise RuntimeError(
                    f"llama-server exited while loading {self.model_path.name} "
                    f"with code {self.process.returncode}.\n{details}"
                )
            try:
                with urlopen(f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        logger.info("GGUF correction model is ready")
                        return
            except (OSError, URLError):
                time.sleep(0.5)
        self.close()
        raise TimeoutError(f"Timed out loading GGUF model: {self.model_path}")

    def complete(self, messages, temperature, top_p, top_k, max_tokens):
        payload = {
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_tokens": int(max_tokens),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    def close(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log_file = getattr(self, "log_file", None)
        if log_file is not None:
            log_file.close()
            self.log_file = None

    def __del__(self):
        self.close()


class FL_AceStep_GGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _discover_gguf_models(DEFAULT_MODEL_DIRECTORY)
        if not models:
            models = [Path(DEFAULT_MODEL).name]
        return {
            "required": {},
            "optional": {
                "model_directory": ("STRING", {
                    "default": DEFAULT_MODEL_DIRECTORY,
                    "multiline": False,
                    "label": "GGUF model directory",
                }),
                "model_name": (models, {"label": "GGUF model"}),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
                "reasoning": (["Off", "On", "Auto"], {"default": "Off"}),
                "runtime_path": ("STRING", {
                    "default": DEFAULT_RUNTIME,
                    "multiline": False,
                    "label": "llama-server path",
                }),
                "context_size": ("INT", {
                    "default": 4096,
                    "min": 1024,
                    "max": 32768,
                    "step": 512,
                }),
            },
            "hidden": {
                "model_path": "STRING",
                "gpu_layers": "INT",
            },
        }

    RETURN_TYPES = ("GGUF_TEXT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Loaders"

    def load(
        self,
        model_directory=DEFAULT_MODEL_DIRECTORY,
        model_name=Path(DEFAULT_MODEL).name,
        device="CPU",
        reasoning="Off",
        runtime_path=DEFAULT_RUNTIME,
        context_size=4096,
        model_path=None,
        gpu_layers=None,
    ):
        if model_path:
            selected_model = Path(model_path).expanduser().resolve()
        else:
            directory = Path(model_directory).expanduser().resolve()
            if Path(model_name).name != model_name:
                raise ValueError(f"GGUF model name must be a file in {directory}")
            selected_model = directory / model_name
        if not selected_model.is_file() or selected_model.suffix.lower() != ".gguf":
            raise ValueError(f"GGUF model not found: {selected_model}")
        layers = int(gpu_layers) if gpu_layers is not None else (999 if device == "GPU" else 0)
        return (GGUFTextModel(selected_model, runtime_path, context_size, layers, reasoning),)


class FL_AceStep_GGUFModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        models = _discover_gguf_models(DEFAULT_MODEL_DIRECTORY)
        if not models:
            models = [Path(DEFAULT_MODEL).name]
        return {
            "required": {
                "model_directory": ("STRING", {
                    "default": DEFAULT_MODEL_DIRECTORY,
                    "multiline": False,
                    "label": "GGUF model directory",
                }),
                "model_name": (models, {"label": "GGUF model"}),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
                "reasoning": (["Off", "On", "Auto"], {"default": "Off"}),
                "runtime_path": ("STRING", {
                    "default": DEFAULT_RUNTIME,
                    "multiline": False,
                    "label": "llama-server path",
                }),
            },
            "optional": {
                "context_size": ("INT", {
                    "default": 4096,
                    "min": 1024,
                    "max": 32768,
                    "step": 512,
                }),
            },
        }

    RETURN_TYPES = ("GGUF_TEXT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Loaders"

    def load(self, model_directory, model_name, device, reasoning, runtime_path, context_size=4096):
        directory = Path(model_directory).expanduser().resolve()
        if Path(model_name).name != model_name:
            raise ValueError(f"GGUF model name must be a file in {directory}")
        model_path = directory / model_name
        if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
            raise ValueError(
                f"GGUF model not found: {model_path}. "
                "Restart ComfyUI after changing the model directory so the dropdown refreshes."
            )
        gpu_layers = 999 if device == "GPU" else 0
        return (GGUFTextModel(model_path, runtime_path, context_size, gpu_layers, reasoning),)


def _lyrics_context(sample):
    variants = getattr(sample, "lyrics_variants", {}) or {}
    sections = []
    for key, title in (("mk", "Macedonian Cyrillic"), ("mktl", "Macedonian transliteration"), ("en", "English translation")):
        text = variants.get(key, "")
        if text:
            sections.append(f"## {title}\n{text}")
    if not sections and sample.lyrics:
        sections.append(f"## Lyrics\n{sample.lyrics}")
    return "\n\n".join(sections) or "[Instrumental or lyrics unavailable]"


def _parse_json_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start, end = response.find("{"), response.rfind("}")
        if start >= 0 and end > start:
            return json.loads(response[start:end + 1])
        raise ValueError("GGUF model did not return a JSON object")


class FL_AceStep_GGUFLabelCorrector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("GGUF_TEXT_MODEL",),
            },
            "optional": {
                "language": ("STRING", {"default": "Macedonian (mk)"}),
                "label_guidance": ("STRING", {
                    "default": "Use Macedonian and Balkan musical terminology. Preserve facts supported by the draft caption. Do not invent instruments.",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 5}),
                "max_tokens": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64}),
                "max_samples": ("INT", {"default": 0, "min": 0, "max": 4428, "step": 1, "label": "0 = all samples"}),
                "review_path": ("STRING", {"default": "./output/acestep/label_review.json", "multiline": False}),
            },
        }

    RETURN_TYPES = ("ACESTEP_DATASET", "INT", "STRING")
    RETURN_NAMES = ("dataset", "corrected_count", "status")
    FUNCTION = "correct"
    CATEGORY = "FL AceStep/Dataset"

    def correct(
        self, dataset, model, language="Macedonian (mk)", label_guidance="",
        temperature=0.1, top_p=0.85, top_k=20, max_tokens=384,
        max_samples=0, review_path="./output/acestep/label_review.json",
    ):
        if model_management:
            try:
                # The GGUF server is a separate process. Release ComfyUI's
                # ACE-Step models before starting correction to avoid RAM/XPU
                # contention, then later nodes can load them again normally.
                model_management.unload_all_models()
                model_management.cleanup_models_gc()
            except Exception as error:
                logger.warning("Could not fully release ACE-Step models: %s", error)

        samples = [sample for sample in dataset.samples if sample.labeled or sample.caption]
        if max_samples:
            samples = samples[:int(max_samples)]
        if not samples:
            model.close()
            return dataset, 0, "No labeled samples to correct"

        pbar = ProgressBar(len(samples)) if ProgressBar else None
        corrected = 0
        errors = []
        for sample in samples:
            prompt = {
                "language": language,
                "guidance": label_guidance,
                "draft_caption": sample.caption,
                "draft_genre": sample.genre,
                "bpm": sample.bpm,
                "keyscale": sample.keyscale,
                "timesignature": sample.timesignature,
                "lyrics": _lyrics_context(sample),
            }
            messages = [
                {"role": "system", "content": "You are a Macedonian music metadata editor. Return only valid JSON with keys caption, genre, language, instruments, rhythm_form, region. Preserve supported facts and never invent audio details."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ]
            try:
                result = _parse_json_response(model.complete(messages, temperature, top_p, top_k, max_tokens))
                if result.get("caption"):
                    sample.caption = str(result["caption"])
                if result.get("genre"):
                    sample.genre = str(result["genre"])
                if result.get("language"):
                    sample.language = str(result["language"])
                sample.labeled = True
                corrected += 1
            except Exception as error:
                errors.append(f"{sample.id}: {error}")
                logger.warning("GGUF correction failed for %s: %s", sample.id, error)
            if pbar:
                pbar.update(1)

        try:
            self._write_review(review_path, dataset.samples)
        finally:
            model.close()
        status = f"Corrected {corrected}/{len(samples)} labels; review exported to {review_path}"
        if errors:
            status += f" ({len(errors)} errors)"
        return dataset, corrected, status

    @staticmethod
    def _write_review(path, samples):
        review_path = Path(path).expanduser()
        review_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for sample in samples:
            entries.append({
                "id": sample.id,
                "audio_path": sample.audio_path,
                "filename": sample.filename,
                "approved": False,
                "caption": sample.caption,
                "genre": sample.genre,
                "language": sample.language,
                "lyrics": sample.lyrics,
            })
        review_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
