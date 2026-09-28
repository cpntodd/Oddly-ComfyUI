"""
ACE-Step LLM Loader Node

Loads the 5Hz-lm model for audio understanding and auto-labeling.
"""

import os
import re
import logging
from contextlib import contextmanager
from pathlib import Path
import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.model_downloader import (
    get_acestep_models_dir,
    ensure_lm_model,
)

logger = logging.getLogger("FL_AceStep_Training")

# Available LLM model variants
LLM_MODELS = [
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-4B",
]

DEVICE_OPTIONS = ["auto", "xpu", "cuda", "cpu"]
BACKEND_OPTIONS = ["pt", "vllm"]

# System instructions matching the official ACE-Step 5Hz-lm training format
UNDERSTAND_INSTRUCTION = (
    "# Instruction\n"
    "Understand the given musical conditions and describe "
    "the audio semantics accordingly. Verified lyrics and translations "
    "are provided when available. Use them to identify language, subject, "
    "and culturally specific terms. Do not invent instruments, BPM, key, "
    "or genre; use unknown when the audio does not support a claim.\n\n"
)
FORMAT_INSTRUCTION = (
    "# Instruction\n"
    "Format the user's input into a more detailed and "
    "specific musical description:\n\n"
)


class ACEStepLLMHandler:
    """
    Handler for the 5Hz-lm model.

    Wraps the LLM for audio understanding and metadata generation.
    Uses ChatML prompt format matching the official ACE-Step training.
    """

    def __init__(self, model, tokenizer, device, dtype, model_name, target_device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_device = target_device or device
        self.dtype = dtype
        self.model_name = model_name

    def move_to_device(self, device):
        """Move a CPU-staged model to its generation device."""
        if self.device == device:
            return
        self.model = self.model.to(device)
        self.device = device

    def understand_audio_from_codes(
        self,
        audio_codes: str,
        lyrics_context: str = "",
        language_hint: str = "Auto",
        label_guidance: str = "",
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 768,
    ):
        """
        Generate metadata and lyrics from audio codes.

        Args:
            audio_codes: String of audio code tokens (e.g. "<|audio_code_123|>...")
            temperature: Sampling temperature (0.3 = deterministic for metadata)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with generated metadata (caption, bpm, keyscale, etc.)
        """
        messages = self._build_understanding_messages(
            audio_codes, lyrics_context, language_hint, label_guidance
        )
        response = self._generate(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        logger.info(f"LLM understanding response length: {len(response)}")
        return self._parse_response(response)

    def understand_audio_from_codes_batch(
        self,
        audio_codes: list[str],
        lyrics_contexts: list[str] = None,
        language_hints: list[str] = None,
        label_guidances: list[str] = None,
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 768,
    ) -> list[dict]:
        """Generate metadata for several audio-code prompts in one batch."""
        if lyrics_contexts is None:
            lyrics_contexts = [""] * len(audio_codes)
        if language_hints is None:
            language_hints = ["Auto"] * len(audio_codes)
        if label_guidances is None:
            label_guidances = [""] * len(audio_codes)
        messages = [
            self._build_understanding_messages(
                codes, context, language_hint, guidance
            )
            for codes, context, language_hint, guidance in zip(
                audio_codes, lyrics_contexts, language_hints, label_guidances
            )
        ]
        responses = self._generate_batch(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        return [self._parse_response(response) for response in responses]

    def format_sample(
        self,
        caption: str,
        lyrics: str,
        user_metadata: dict = None,
        instruction_context: str = "",
        language_hint: str = "Auto",
        label_guidance: str = "",
        temperature: float = 0.85,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 768,
    ):
        """
        Format user-provided caption and lyrics into structured metadata.

        Args:
            caption: User-provided caption
            lyrics: User-provided lyrics
            user_metadata: Optional metadata dict (unused, kept for API compat)
            temperature: Sampling temperature (0.85 = creative for formatting)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with formatted metadata
        """
        messages = self._build_formatting_messages(
            caption, lyrics, instruction_context, language_hint, label_guidance
        )
        response = self._generate(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        logger.info(f"LLM formatting response length: {len(response)}")
        return self._parse_response(response)

    def _build_understanding_messages(
        self,
        audio_codes: str,
        lyrics_context: str = "",
        language_hint: str = "Auto",
        label_guidance: str = "",
    ) -> list:
        """Build ChatML messages for audio understanding."""
        system_content = UNDERSTAND_INSTRUCTION
        if language_hint != "Auto":
            language_code, language_name = language_hint.split(" — ", 1)
            system_content += (
                "# Dataset language constraint\n"
                f"The dataset language is {language_name} (ISO 639-1: {language_code}). "
                "Treat this as authoritative. Describe language and cultural context "
                "as the selected language unless the supplied evidence "
                "clearly proves otherwise. Do not substitute a different regional "
                "category based only on instrumental associations.\n\n"
            )
        if label_guidance.strip():
            system_content += (
                "# Dataset labelling guidance\n"
                f"{label_guidance.strip()}\n\n"
            )
        user_content = audio_codes
        if lyrics_context:
            user_content = f"{audio_codes}\n\n{lyrics_context}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_formatting_messages(
        self,
        caption: str,
        lyrics: str,
        instruction_context: str = "",
        language_hint: str = "Auto",
        label_guidance: str = "",
    ) -> list:
        """Build ChatML messages for sample formatting."""
        caption = caption or "NO USER INPUT"
        lyrics = lyrics or "[Instrumental]"
        system_content = FORMAT_INSTRUCTION
        if language_hint != "Auto":
            language_code, language_name = language_hint.split(" — ", 1)
            system_content += (
                f"The dataset language is {language_name} (ISO 639-1: {language_code}) "
                "and must be retained in the formatted metadata.\n"
            )
        if label_guidance.strip():
            system_content += f"Dataset labelling guidance: {label_guidance.strip()}\n"
        user_content = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}"
        if instruction_context:
            user_content = f"{user_content}\n\n{instruction_context}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _generate(
        self,
        messages: list,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 768,
    ) -> str:
        """
        Generate response from LLM using ChatML prompt format.

        Args:
            messages: List of ChatML message dicts (role, content)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            max_new_tokens: Maximum tokens to generate

        Returns:
            Raw response string (with <think> tags preserved)
        """
        # Apply ChatML template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with self._cpu_generation_threads():
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

        # Decode only the generated tokens (not the prompt)
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Strip trailing end tokens
        for end_token in ["<|im_end|>", "<|endoftext|>"]:
            response = response.replace(end_token, "")

        return response.strip()

    def _generate_batch(
        self,
        messages_batch: list[list[dict]],
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 768,
    ) -> list[str]:
        """Generate responses for a batch while reusing the loaded model."""
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        old_padding_side = self.tokenizer.padding_side
        old_pad_token = self.tokenizer.pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            input_length = inputs.input_ids.shape[1]

            with self._cpu_generation_threads():
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )

            generated_ids = outputs[:, input_length:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        finally:
            self.tokenizer.padding_side = old_padding_side
            self.tokenizer.pad_token = old_pad_token

        for i, response in enumerate(responses):
            for end_token in ["<|im_end|>", "<|endoftext|>"]:
                response = response.replace(end_token, "")
            responses[i] = response.strip()

        return responses

    @contextmanager
    def _cpu_generation_threads(self):
        if self.device != "cpu":
            yield
            return

        previous_threads = torch.get_num_threads()
        target_threads = os.cpu_count() or previous_threads
        if target_threads > previous_threads:
            torch.set_num_threads(target_threads)
        try:
            yield
        finally:
            if target_threads > previous_threads:
                torch.set_num_threads(previous_threads)

    def _parse_response(self, response: str) -> dict:
        """
        Parse LLM response containing <think> tags with YAML metadata
        and optional lyrics after </think>.

        Expected format:
            <think>
            bpm: 120
            caption: A driving electronic track...
            duration: 180
            genres: Electronic, Synthwave
            keyscale: A minor
            language: en
            timesignature: 4
            </think>

            # Lyric
            [Verse 1]
            Walking through the city lights...
        """
        result = {
            "caption": "",
            "bpm": None,
            "keyscale": "",
            "timesignature": "4",
            "language": "instrumental",
            "genre": "",
            "lyrics": "[Instrumental]",
        }

        # Extract <think>...</think> block
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if not think_match:
            logger.warning("No <think> tags found in LLM response, attempting plain parse")
            logger.debug(f"Response: {response[:500]}")
            return self._parse_plain_response(response, result)

        think_content = think_match.group(1).strip()

        # Parse YAML-style key:value pairs inside think block
        current_key = None
        current_value_lines = []

        for line in think_content.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('<'):
                continue

            # Check if this is a new key:value pair (not indented, has colon)
            if ':' in stripped and not line.startswith((' ', '\t')):
                # Save previous field
                if current_key and current_value_lines:
                    self._set_metadata_field(
                        result, current_key,
                        ' '.join(v.strip() for v in current_value_lines if v.strip())
                    )

                key, value = stripped.split(':', 1)
                current_key = key.strip().lower()
                current_value_lines = [value] if value.strip() else []
            elif current_key:
                # Continuation line (multi-line YAML value)
                current_value_lines.append(stripped)

        # Save last field
        if current_key and current_value_lines:
            self._set_metadata_field(
                result, current_key,
                ' '.join(v.strip() for v in current_value_lines if v.strip())
            )

        # Extract lyrics after </think>
        after_think = response.split('</think>', 1)
        if len(after_think) > 1:
            lyrics_content = after_think[1].strip()

            # Remove # Lyric header
            lyrics_content = re.sub(
                r'^#\s*Lyric[s]?\s*\n?', '', lyrics_content, flags=re.IGNORECASE
            )
            lyrics_content = lyrics_content.strip()

            if lyrics_content and lyrics_content.lower() != "[instrumental]":
                result["lyrics"] = lyrics_content
                # If we got real lyrics and language is still default, set to unknown
                if result["language"] == "instrumental":
                    result["language"] = "unknown"

        return result

    def _set_metadata_field(self, result: dict, key: str, value: str):
        """Set a metadata field from a parsed YAML key:value pair."""
        if not value:
            return

        if key == "bpm":
            try:
                result["bpm"] = int(float(value))
            except (ValueError, TypeError):
                pass
        elif key == "caption":
            result["caption"] = value
        elif key in ("keyscale", "key"):
            result["keyscale"] = value
        elif key == "timesignature":
            result["timesignature"] = value
        elif key == "language":
            result["language"] = value.lower()
        elif key in ("genre", "genres"):
            result["genre"] = value
        # duration is ignored - we get it from the audio file

    def _parse_plain_response(self, response: str, result: dict) -> dict:
        """Fallback parser for responses without <think> tags."""
        for line in response.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            lower = stripped.lower()
            if lower.startswith("caption:"):
                result["caption"] = stripped.split(":", 1)[1].strip()
            elif lower.startswith("bpm:"):
                try:
                    result["bpm"] = int(stripped.split(":", 1)[1].strip())
                except (ValueError, TypeError):
                    pass
            elif lower.startswith(("key:", "keyscale:")):
                result["keyscale"] = stripped.split(":", 1)[1].strip()
            elif lower.startswith("timesignature:"):
                result["timesignature"] = stripped.split(":", 1)[1].strip()
            elif lower.startswith("language:"):
                result["language"] = stripped.split(":", 1)[1].strip().lower()
            elif lower.startswith(("genre:", "genres:")):
                result["genre"] = stripped.split(":", 1)[1].strip()

        return result


class FL_AceStep_LLMLoader:
    """
    Load ACE-Step LLM (5Hz Language Model)

    Loads the 5Hz-lm model for audio understanding and auto-labeling.
    This model is used to automatically generate captions, metadata, and lyrics
    from audio samples during dataset preparation.

    Available models:
    - acestep-5Hz-lm-1.7B (default, balanced)
    - acestep-5Hz-lm-0.6B (lightweight)
    - acestep-5Hz-lm-4B (high quality, requires more VRAM)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (LLM_MODELS, {"default": "acestep-5Hz-lm-1.7B"}),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "backend": (BACKEND_OPTIONS, {"default": "pt"}),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for auto-download"
                }),
            }
        }

    RETURN_TYPES = ("ACESTEP_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Loaders"

    def load(self, model_name, device, backend, checkpoint_path=""):
        """Load the LLM model."""
        logger.info(f"Loading ACE-Step LLM: {model_name}")

        # Progress bar
        pbar = ProgressBar(2) if ProgressBar else None

        # Determine models directory
        if checkpoint_path and checkpoint_path.strip():
            models_dir = Path(checkpoint_path.strip()).expanduser()
        else:
            models_dir = get_acestep_models_dir()

        # Step 1: Ensure LLM is downloaded
        if pbar:
            pbar.update(1)

        success, status = ensure_lm_model(model_name, models_dir)
        if not success:
            raise RuntimeError(f"Failed to ensure LLM: {status}")

        logger.info(status)

        # Determine device
        if device == "auto":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        requested_device = device
        load_device = "cpu" if device == "xpu" else device

        # Step 2: Load the LLM
        if pbar:
            pbar.update(1)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            lm_path = os.path.join(models_dir, model_name)

            logger.info(f"Loading LLM from {lm_path}")
            logger.info(f"Device: {device}, Backend: {backend}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lm_path)

            # Determine dtype based on device
            if device in ("cuda", "xpu"):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Load model
            if backend == "vllm":
                # vLLM backend (if available)
                try:
                    from vllm import LLM
                    model = LLM(model=lm_path)
                    logger.info("Using vLLM backend")
                except ImportError:
                    logger.warning("vLLM not available, falling back to PyTorch")
                    backend = "pt"

            if backend == "pt":
                # PyTorch backend
                model = AutoModelForCausalLM.from_pretrained(
                    lm_path,
                    torch_dtype=torch_dtype,
                    device_map=load_device if load_device == "cuda" else None,
                )
                if load_device == "cpu":
                    model = model.to("cpu")
                elif load_device != "cuda":
                    model = model.to(load_device)
                model.eval()

        except Exception as e:
            logger.exception("Failed to load LLM")
            raise RuntimeError(f"Failed to load LLM: {str(e)}")

        # Create handler
        handler = ACEStepLLMHandler(
            model=model,
            tokenizer=tokenizer,
            device=load_device,
            dtype=torch_dtype if backend == "pt" else None,
            model_name=model_name,
            target_device=requested_device,
        )

        logger.info(f"LLM '{model_name}' loaded successfully")

        return (handler,)
