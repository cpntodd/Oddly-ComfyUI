"""
ACE-Step Dataset Preprocess Node

Converts labeled samples to tensor files for training.
Uses native ComfyUI MODEL type for the ACE-Step model.

Performance-optimized to match the sdbds reference implementation:
- Models loaded once, kept on GPU for entire loop
- torch.inference_mode() wraps entire loop
- Cached refer_audio tensors and resampler objects
- non_blocking GPU transfers
- Periodic cache clearing
"""

import json
import logging
import math
import random
import re
from pathlib import Path

import torch

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_silence_latent,
    get_acestep_encoder,
)
from ..modules.audio_utils import load_audio, vae_encode_direct

logger = logging.getLogger("FL_AceStep_Training")

# SFT generation prompt template (from ACE-Step constants)
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# Cache for refer_audio placeholder tensors (avoid GPU alloc per sample)
_REFER_AUDIO_CACHE: dict = {}
_LYRIC_TIMESTAMP = re.compile(
    r"^\s*\[?(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]?\s*(.*)$"
)


def _get_refer_audio_tensors(device, dtype):
    """Get cached refer_audio placeholder tensors for text2music (no reference audio)."""
    cache_key = (device, dtype)
    if cache_key not in _REFER_AUDIO_CACHE:
        _REFER_AUDIO_CACHE[cache_key] = (
            torch.zeros(1, 1, 64, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=torch.long),
        )
    refer_audio_hidden, refer_audio_order_mask = _REFER_AUDIO_CACHE[cache_key]
    # Reset in-place (cheap) rather than allocating new tensors
    refer_audio_hidden.zero_()
    refer_audio_order_mask.zero_()
    return refer_audio_hidden, refer_audio_order_mask


def encode_text_and_lyrics(clip, text: str, lyrics: str, device, dtype):
    """
    Encode text and lyrics using ComfyUI's native CLIP pipeline.

    For ACE-Step 1.5, this uses the Qwen3 model:
    - Text: Full forward pass -> last_hidden_state
    - Lyrics: Layer 0 output only (shallow embedding)

    IMPORTANT: Must use return_dict=True to get lyrics embeddings.
    """
    tokens = clip.tokenize(text, lyrics=lyrics)
    result = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)

    text_hidden_states = result["cond"].to(device=device, dtype=dtype, non_blocking=True)
    text_attention_mask = torch.ones(
        text_hidden_states.shape[:2], device=device, dtype=dtype
    )

    lyric_hidden_states = result.get("conditioning_lyrics", None)
    if lyric_hidden_states is not None:
        lyric_hidden_states = lyric_hidden_states.to(device=device, dtype=dtype, non_blocking=True)
        if lyric_hidden_states.dim() == 2:
            lyric_hidden_states = lyric_hidden_states.unsqueeze(0)
        lyric_attention_mask = torch.ones(
            lyric_hidden_states.shape[:2], device=device, dtype=dtype
        )
    else:
        lyric_hidden_states = torch.zeros(1, 1, text_hidden_states.shape[-1],
                                          device=device, dtype=dtype)
        lyric_attention_mask = torch.zeros(1, 1, device=device, dtype=dtype)

    return text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask


def _lyrics_for_chunk(lyrics: str, chunk_start: float, chunk_duration: float) -> str:
    """Return timestamped lyric lines belonging to one audio chunk.

    Untimestamped lyrics are returned unchanged because there is no safe way
    to infer their alignment from plain Markdown text.
    """
    if not lyrics or lyrics == "[Instrumental]":
        return lyrics

    blocks = []
    current = None
    for line in lyrics.splitlines():
        match = _LYRIC_TIMESTAMP.match(line)
        if match:
            minutes, seconds, fraction, text = match.groups()
            offset = int(minutes) * 60 + int(seconds)
            if fraction:
                offset += int(fraction) / (10 ** len(fraction))
            current = [offset, text.strip()]
            blocks.append(current)
        elif current is not None:
            current[1] = f"{current[1]}\n{line}" if current[1] else line

    if not blocks:
        return lyrics

    chunk_end = chunk_start + chunk_duration
    selected = []
    for index, (offset, text) in enumerate(blocks):
        next_offset = blocks[index + 1][0] if index + 1 < len(blocks) else float("inf")
        if offset < chunk_end and next_offset > chunk_start:
            selected.append(text.strip())

    return "\n".join(text for text in selected if text) or "[Instrumental]"


def _apply_approved_labels(dataset, review_path):
    review_file = Path(review_path)
    if not review_file.exists():
        raise ValueError(f"Label review file not found: {review_file}")

    entries = json.loads(review_file.read_text(encoding="utf-8"))
    approved = {entry.get("id"): entry for entry in entries if entry.get("approved") is True}
    missing = [sample.id for sample in dataset.samples if sample.labeled and sample.id not in approved]
    if missing:
        raise ValueError(
            f"{len(missing)} labeled samples are not approved in {review_file}"
        )

    for sample in dataset.samples:
        entry = approved.get(sample.id)
        if entry is None:
            continue
        for field in ("caption", "genre", "language", "lyrics"):
            if field in entry:
                setattr(sample, field, entry[field])

    return len(approved)


def _write_review_snapshot(path, samples):
    """Create an editable review file before preprocessing begins."""
    review = []
    for sample in samples:
        review.append({
            "id": sample.id,
            "audio_path": sample.audio_path,
            "filename": sample.filename,
            "approved": False,
            "caption": sample.caption,
            "genre": sample.genre,
            "language": sample.language,
            "lyrics": sample.lyrics,
        })
    review_file = Path(path)
    review_file.parent.mkdir(parents=True, exist_ok=True)
    review_file.write_text(
        json.dumps(review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(review_file)


def _is_xpu_device_lost(error):
    message = str(error).upper()
    return "UR_RESULT_ERROR_DEVICE_LOST" in message or "DEVICE_LOST" in message


def _safe_empty_cache(device):
    """Best-effort cache cleanup; a lost accelerator cannot be repaired here."""
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()
    except Exception as error:
        logger.warning("Skipping %s cache cleanup: %s", device, error)
        return _is_xpu_device_lost(error)
    return False


class FL_AceStep_PreprocessDataset:
    """
    Preprocess Dataset

    Converts labeled audio samples to preprocessed tensor files for training.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "output_dir": ("STRING", {
                    "default": "./output/acestep/datasets",
                    "multiline": False,
                }),
            },
            "optional": {
                "max_duration": ("FLOAT", {
                    "default": 240.0,
                    "min": 10.0,
                    "max": 600.0,
                    "step": 10.0,
                }),
                "vae_chunk_seconds": ("FLOAT", {
                    "default": 30.0,
                    "min": 10.0,
                    "max": 120.0,
                    "step": 5.0,
                    "label": "VAE chunk length (seconds)",
                }),
                "genre_ratio": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                }),
                "require_label_approval": ("BOOLEAN", {
                    "default": False,
                    "label": "Require approved labels",
                }),
                "review_path": ("STRING", {
                    "default": "./output/acestep/label_review.json",
                    "multiline": False,
                    "label": "Label review file",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_path", "sample_count", "status")
    FUNCTION = "preprocess"
    CATEGORY = "FL AceStep/Dataset"
    OUTPUT_NODE = True

    def preprocess(
        self,
        dataset,
        model,
        vae,
        clip,
        output_dir,
        max_duration=240.0,
        vae_chunk_seconds=30.0,
        genre_ratio=0,
        require_label_approval=False,
        review_path="./output/acestep/label_review.json",
    ):
        """Preprocess the dataset to tensor files."""
        samples = dataset.samples
        if not samples:
            return (output_dir, 0, "No samples to preprocess")

        if not is_acestep_model(model):
            return (output_dir, 0, "Error: Model is not an ACE-Step model")

        labeled_samples = [s for s in samples if s.labeled or s.caption]
        if not labeled_samples:
            return (output_dir, 0, "No labeled samples to preprocess")

        if require_label_approval:
            review_file = Path(review_path)
            if not review_file.exists():
                try:
                    created_review = _write_review_snapshot(review_path, labeled_samples)
                except Exception as e:
                    return (output_dir, 0, f"Could not create label review file: {e}")
                return (
                    output_dir,
                    0,
                    f"Manual label review required. Edit {created_review} and set "
                    "approved=true for each sample, then run preprocessing again",
                )
            try:
                approved_count = _apply_approved_labels(dataset, review_path)
                logger.info("Applied %d approved labels from %s", approved_count, review_path)
            except Exception as e:
                return (output_dir, 0, f"Manual label review required: {e}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        device = model_management.get_torch_device() if model_management else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae_model = vae.first_stage_model
        vae_dtype = vae.vae_dtype
        condition_encoder = get_acestep_encoder(model)
        enc_dtype = next(condition_encoder.parameters()).dtype
        logger.info(f"Preprocess device: {device}")

        # Get silence latent
        silence_latent = get_silence_latent(model)
        if silence_latent is None:
            silence_latent = torch.zeros(1, 750, 64, device=device, dtype=enc_dtype)

        chunk_seconds = max(10.0, float(vae_chunk_seconds))
        total_chunks = sum(
            max(1, math.ceil(min(float(sample.duration), max_duration) / chunk_seconds))
            for sample in labeled_samples
        )

        # Progress bar tracks source samples; logs report chunk progress.
        pbar = ProgressBar(len(labeled_samples)) if ProgressBar else None

        processed_count = 0
        manifest = []
        errors = []

        logger.info(
            f"Preprocessing {len(labeled_samples)} samples as up to "
            f"{total_chunks} {chunk_seconds:.0f}s chunks to {output_dir}"
        )

        device_lost = False

        # --- Main loop: inference_mode for entire batch ---
        with torch.inference_mode():
            for i, sample in enumerate(labeled_samples):
                source_duration = min(float(sample.duration), max_duration)
                chunk_count = max(1, math.ceil(source_duration / chunk_seconds))
                encoded_chunks = []
                try:
                    # Keep the VAE resident while encoding all chunks from this
                    # source. Repeated VAE/LLM swaps were destabilising XPU.
                    if model_management:
                        model_management.load_models_gpu(
                            [vae.patcher],
                            force_full_load=getattr(vae, 'disable_offload', False),
                        )
                    else:
                        vae_model.to(device)

                    for chunk_index in range(chunk_count):
                        chunk_start = chunk_index * chunk_seconds
                        chunk_duration = min(chunk_seconds, source_duration - chunk_start)
                        if chunk_duration < 1.0:
                            continue

                        audio = load_audio(
                            sample.audio_path,
                            max_duration=chunk_duration,
                            start_seconds=chunk_start,
                        )[0].unsqueeze(0).to(device=device, dtype=vae_dtype)
                        target_latents = vae_encode_direct(
                            vae_model,
                            audio,
                            device=device,
                            dtype=vae_dtype,
                        )
                        del audio
                        encoded_chunks.append((chunk_index, chunk_start, chunk_duration, target_latents))

                    # Keep CLIP and the ACE-Step condition encoder resident while
                    # all encoded chunks from this source are being conditioned.
                    if model_management:
                        model_management.load_models_gpu([clip.patcher])
                        model_management.load_models_gpu([model])
                    else:
                        clip.cond_stage_model.to(device)
                        condition_encoder.to(device)

                    for chunk_index, chunk_start, chunk_duration, target_latents in encoded_chunks:
                        chunk_lyrics = _lyrics_for_chunk(
                            sample.lyrics,
                            chunk_start,
                            chunk_duration,
                        )
                        tensor_data = self._preprocess_sample(
                            sample=sample,
                            target_latents=target_latents,
                            clip=clip,
                            condition_encoder=condition_encoder,
                            silence_latent=silence_latent,
                            max_duration=max_duration,
                            genre_ratio=genre_ratio,
                            custom_tag=dataset.metadata.custom_tag,
                            tag_position=dataset.metadata.tag_position,
                            device=device,
                            vae_dtype=vae_dtype,
                            enc_dtype=enc_dtype,
                            chunk_duration=chunk_duration,
                            chunk_lyrics=chunk_lyrics,
                        )

                        if tensor_data is None:
                            continue

                        chunk_id = f"{sample.id}_chunk_{chunk_index:03d}"
                        tensor_filename = f"{chunk_id}.pt"
                        torch.save(tensor_data, output_path / tensor_filename)

                        manifest.append({
                            "id": chunk_id,
                            "filename": tensor_filename,
                            "audio_path": sample.audio_path,
                            "caption": sample.caption,
                            "duration": chunk_duration,
                            "chunk_index": chunk_index,
                            "chunk_start": chunk_start,
                            "bpm": sample.bpm,
                            "keyscale": sample.keyscale,
                            "is_instrumental": sample.is_instrumental,
                        })

                        processed_count += 1
                        logger.info(
                            f"[{processed_count}/{total_chunks}] {sample.filename} "
                            f"chunk {chunk_index + 1}/{chunk_count} "
                            f"({chunk_start:.0f}-{chunk_start + chunk_duration:.0f}s)"
                        )

                        del target_latents, tensor_data

                except Exception as e:
                    error_msg = f"Error processing sample {sample.id}: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    if _is_xpu_device_lost(e):
                        device_lost = True
                        logger.error(
                            "XPU device lost; stopping preprocessing. Restart "
                            "ComfyUI before retrying."
                        )

                if pbar:
                    pbar.update(1)

                del encoded_chunks
                if device_lost:
                    break

                # Periodic device cache clearing (every 8 samples)
                if (i + 1) % 8 == 0:
                    if _safe_empty_cache(device):
                        device_lost = True
                        logger.error(
                            "XPU device lost during cleanup; stopping preprocessing. "
                            "Restart ComfyUI before retrying."
                        )
                        break

        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "samples": manifest,
                "metadata": {
                    "total_samples": processed_count,
                    "max_duration": max_duration,
                    "vae_chunk_seconds": chunk_seconds,
                    "genre_ratio": genre_ratio,
                    "custom_tag": dataset.metadata.custom_tag,
                }
            }, f, indent=2, ensure_ascii=False)

        status = f"Preprocessed {processed_count} chunks from {len(labeled_samples)} samples"
        if errors:
            status += f" ({len(errors)} errors)"
        if device_lost:
            status += (
                "; stopped after XPU device loss - restart ComfyUI before retrying"
            )

        logger.info(status)
        return (str(output_path), processed_count, status)

    def _preprocess_sample(
        self,
        sample,
        target_latents,
        clip,
        condition_encoder,
        silence_latent,
        max_duration,
        genre_ratio,
        custom_tag,
        tag_position,
        device,
        vae_dtype,
        enc_dtype,
        chunk_duration=None,
        chunk_lyrics=None,
    ):
        """Preprocess a single sample to tensor data."""
        # The audio has already been encoded before loading the text encoder.

        latent_length = target_latents.shape[1]
        attention_mask = torch.ones(1, latent_length, device=device)

        # Step 3: Build caption with custom tag
        caption = sample.caption
        if custom_tag:
            if tag_position == "prepend":
                caption = f"{custom_tag}, {caption}"
            elif tag_position == "append":
                caption = f"{caption}, {custom_tag}"
            elif tag_position == "replace":
                caption = custom_tag

        use_genre = random.randint(0, 100) < genre_ratio and sample.genre
        text_content = sample.genre if use_genre else caption

        # Build metadata string (always include all fields, N/A for missing)
        metas_str = (
            f"- bpm: {sample.bpm if sample.bpm else 'N/A'}\n"
            f"- timesignature: {sample.timesignature if sample.timesignature else 'N/A'}\n"
            f"- keyscale: {sample.keyscale if sample.keyscale else 'N/A'}\n"
            f"- duration: {int(chunk_duration if chunk_duration is not None else sample.duration)} seconds\n"
        )

        text_prompt = SFT_GEN_PROMPT.format(
            DEFAULT_DIT_INSTRUCTION,
            text_content,
            metas_str
        )

        # Step 4: Encode text and lyrics via ComfyUI CLIP
        lyrics = chunk_lyrics if chunk_lyrics is not None else (
            sample.lyrics if sample.lyrics else "[Instrumental]"
        )
        text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = \
            encode_text_and_lyrics(clip, text_prompt, lyrics, device, enc_dtype)

        # Step 5: Run condition encoder to merge text+lyrics+timbre
        refer_audio_hidden, refer_audio_order_mask = _get_refer_audio_tensors(device, enc_dtype)

        encoder_hidden_states, encoder_attention_mask = condition_encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # Step 6: Build context latents [1, T, 128] = [silence(64), chunk_mask(64)]
        context_latents = torch.empty((1, latent_length, 128), device=device, dtype=enc_dtype)

        # Fill silence latent into first 64 channels
        src = silence_latent.to(dtype=enc_dtype)
        src_len = src.shape[1]
        take = min(latent_length, src_len)
        context_latents[:, :take, :64] = src[:, :take, :]
        if take < latent_length:
            # Tile silence to fill remaining length
            remaining = latent_length - take
            pos = take
            while remaining > 0:
                chunk = min(remaining, src_len)
                context_latents[:, pos:pos + chunk, :64] = src[:, :chunk, :]
                pos += chunk
                remaining -= chunk

        # Last 64 channels = 1 (chunk mask: generate all)
        context_latents[:, :, 64:] = 1

        # Step 7: Prepare output (squeeze batch dim, move to CPU for storage)
        tensor_data = {
            "target_latents": target_latents.squeeze(0).cpu(),
            "attention_mask": attention_mask.squeeze(0).cpu(),
            "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
            "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
            "context_latents": context_latents.squeeze(0).cpu(),
            "metadata": {
                "audio_path": sample.audio_path,
                "filename": sample.filename,
                "caption": caption,
                "lyrics": lyrics,
                "duration": chunk_duration if chunk_duration is not None else sample.duration,
                "bpm": sample.bpm,
                "keyscale": sample.keyscale,
                "timesignature": sample.timesignature,
                "language": sample.language,
                "is_instrumental": sample.is_instrumental,
            }
        }

        return tensor_data
