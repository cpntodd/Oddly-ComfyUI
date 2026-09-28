"""
ACE-Step Dataset Label Node

Auto-labels audio samples using the LLM for metadata generation.
Uses native ComfyUI MODEL type for the ACE-Step model.
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
import re

try:
    from comfy.utils import ProgressBar
    import comfy.model_management as model_management
except ImportError:
    ProgressBar = None
    model_management = None

from ..modules.acestep_model import (
    is_acestep_model,
    get_acestep_tokenizer,
)
from ..modules.audio_utils import audio_to_codes

logger = logging.getLogger("FL_AceStep_Training")

LANGUAGE_OPTIONS = [
    "Auto",
    "AA — Afar", "AB — Abkhazian", "AE — Avestan", "AK — Akan",
    "AN — Aragonese", "AV — Avaric", "BA — Bashkir", "BH — Bihari",
    "BI — Bislama", "BM — Bambara", "BO — Tibetan", "BR — Breton",
    "CE — Chechen", "CH — Chamorro", "CR — Cree", "CU — Church Slavic",
    "CV — Chuvash", "DV — Divehi", "DZ — Dzongkha", "FF — Fulah",
    "FY — Western Frisian", "GV — Manx", "HO — Hiri Motu", "HZ — Herero",
    "IA — Interlingua", "IE — Interlingue", "II — Sichuan Yi", "IK — Inupiaq",
    "IO — Ido", "IU — Inuktitut", "KG — Kongo", "KI — Kikuyu",
    "KJ — Kwanyama", "KL — Kalaallisut", "KR — Kanuri", "KS — Kashmiri",
    "KV — Komi", "KW — Cornish", "LG — Ganda", "LN — Lingala",
    "LU — Luba-Katanga", "MH — Marshallese", "NA — Nauru", "NB — Norwegian Bokmål",
    "ND — North Ndebele", "NG — Ndonga", "NN — Norwegian Nynorsk", "NR — South Ndebele",
    "NV — Navajo", "OC — Occitan", "OJ — Ojibwa", "OM — Oromo",
    "OS — Ossetian", "PI — Pali", "RM — Romansh", "RN — Rundi",
    "SC — Sardinian", "SE — Northern Sami", "SG — Sango", "SS — Swati",
    "TL — Tagalog", "TN — Tswana", "TO — Tonga", "TW — Twi",
    "TY — Tahitian", "VE — Venda", "VO — Volapük", "WA — Walloon",
    "WO — Wolof", "ZA — Zhuang",
    "AF — Afrikaans", "SQ — Albanian", "AM — Amharic", "AR — Arabic",
    "HY — Armenian", "AS — Assamese", "AY — Aymara", "AZ — Azerbaijani",
    "EU — Basque", "BE — Belarusian", "BN — Bengali", "BS — Bosnian",
    "BG — Bulgarian", "CA — Catalan", "ZH — Chinese", "CO — Corsican",
    "HR — Croatian", "CS — Czech", "DA — Danish", "NL — Dutch",
    "EN — English", "EO — Esperanto", "ET — Estonian", "EE — Ewe",
    "FO — Faroese", "FJ — Fijian", "FI — Finnish", "FR — French",
    "GL — Galician", "KA — Georgian", "DE — German", "EL — Greek",
    "GN — Guarani", "GU — Gujarati", "HT — Haitian Creole", "HA — Hausa",
    "HE — Hebrew", "HI — Hindi", "HU — Hungarian", "IS — Icelandic",
    "IG — Igbo", "ID — Indonesian", "GA — Irish", "IT — Italian",
    "JA — Japanese", "JV — Javanese", "KN — Kannada", "KK — Kazakh",
    "KM — Khmer", "RW — Kinyarwanda", "KO — Korean", "KU — Kurdish",
    "KY — Kyrgyz", "LO — Lao", "LA — Latin", "LV — Latvian",
    "LI — Limburgish", "LT — Lithuanian", "LB — Luxembourgish", "MK — Macedonian",
    "MG — Malagasy", "MS — Malay", "ML — Malayalam", "MT — Maltese",
    "MI — Maori", "MR — Marathi", "MN — Mongolian", "MY — Burmese",
    "NE — Nepali", "NO — Norwegian", "NY — Nyanja", "OR — Odia",
    "PS — Pashto", "FA — Persian", "PL — Polish", "PT — Portuguese",
    "PA — Punjabi", "QU — Quechua", "RO — Romanian", "RU — Russian",
    "SM — Samoan", "SA — Sanskrit", "GD — Scottish Gaelic", "SR — Serbian",
    "ST — Sesotho", "SN — Shona", "SD — Sindhi", "SI — Sinhala",
    "SK — Slovak", "SL — Slovenian", "SO — Somali", "ES — Spanish",
    "SU — Sundanese", "SW — Swahili", "SV — Swedish", "TG — Tajik",
    "TA — Tamil", "TT — Tatar", "TE — Telugu", "TH — Thai",
    "TI — Tigrinya", "TS — Tsonga", "TR — Turkish", "TK — Turkmen",
    "UK — Ukrainian", "UR — Urdu", "UG — Uyghur", "UZ — Uzbek",
    "VI — Vietnamese", "CY — Welsh", "XH — Xhosa", "YI — Yiddish",
    "YO — Yoruba", "ZU — Zulu",
]


@contextmanager
def _llm_generation_device(llm, model, vae):
    """Make room for a GPU-staged LLM, then restore ComfyUI's models."""
    target_device = getattr(llm, "target_device", getattr(llm, "device", "cpu"))
    if target_device != "xpu":
        yield
        return

    logger.info("Preparing XPU for ACE-Step LLM labelling")
    if model_management:
        model_management.unload_model_and_clones(model, all_devices=True)
        model_management.unload_model_and_clones(vae.patcher, all_devices=True)
        model_management.soft_empty_cache(force=True)

    llm.move_to_device("xpu")
    try:
        yield
    finally:
        llm.move_to_device("cpu")
        if model_management:
            model_management.soft_empty_cache(force=True)
        logger.info("Restored ACE-Step LLM to CPU staging")


def _build_lyrics_context(sample, language_hint="Auto", label_guidance="") -> str:
    variants = getattr(sample, "lyrics_variants", {})
    guidance = (label_guidance or "").strip()
    if not variants and not sample.raw_lyrics and language_hint == "Auto" and not guidance:
        return ""

    sections = [
        "# Verified song context",
        "Use these lyrics as reference for language, subject, names, and cultural terminology. "
        "They are metadata for this song, not words to invent in the caption.",
    ]
    if sample.custom_tag:
        sections.append(f"Dataset tag: {sample.custom_tag}")
    if language_hint != "Auto":
        language_code, language_name = language_hint.split(" — ", 1)
        sections.append(
            f"Declared song language: {language_name} (ISO 639-1: {language_code}). "
            "Treat this language as authoritative when interpreting the lyrics and metadata."
        )
    if guidance:
        sections.append(
            "Label guidance:\n"
            f"{guidance}"
        )
    if variants.get("mk"):
        sections.append(f"Macedonian Cyrillic lyrics:\n{variants['mk']}")
    if variants.get("mktl"):
        sections.append(f"Macedonian transliteration:\n{variants['mktl']}")
    if variants.get("en"):
        sections.append(f"English translation:\n{variants['en']}")
    if not variants and sample.raw_lyrics:
        sections.append(f"Provided lyrics:\n{sample.raw_lyrics}")
    return "\n\n".join(sections)


def _enforce_language_metadata(metadata, language_hint):
    """Apply deterministic language constraints after LLM generation."""
    if language_hint == "Auto":
        return metadata

    language_code, language_name = language_hint.split(" — ", 1)
    metadata["language"] = language_name

    # These are common 0.6B failure modes for Macedonian/Balkan audio. Replace
    # only explicit regional claims; preserve useful neutral audio descriptions.
    if language_code == "MK":
        replacements = (
            (r"\btraditional\s+middle\s+eastern\s+music\b", "traditional folk music"),
            (r"\bmiddle\s+eastern\b", "folk"),
            (r"\barabic\b", "folk"),
            (r"\bpersian\b", "folk"),
            (r"\bduduk\b", "woodwind"),
            (r"\boud\b", "lute-like string instrument"),
            (r"\bdarbuka\b", "hand percussion"),
        )
        caption = metadata.get("caption", "")
        genre = metadata.get("genre", "")
        for pattern, replacement in replacements:
            caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
            genre = re.sub(pattern, replacement, genre, flags=re.IGNORECASE)
        metadata["caption"] = caption
        metadata["genre"] = genre

    return metadata


def _write_label_review(path, samples):
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
    review_path = Path(path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps(review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(review_path)


class FL_AceStep_LabelSamples:
    """
    Auto-Label Samples

    Uses the 5Hz-lm model to automatically generate metadata for audio samples.
    This includes:
    - Caption/description
    - Genre tags
    - BPM (tempo)
    - Key/scale
    - Time signature
    - Language
    - Lyrics (transcription or formatting)

    Requires:
    - dataset: Dataset from Scan Directory node
    - model: ACE-Step MODEL (purple connection) for audio tokenization
    - vae: ACE-Step VAE (red connection) for audio encoding
    - llm: LLM model for metadata generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "model": ("MODEL",),  # Native ComfyUI MODEL type (purple connection)
                "vae": ("VAE",),  # Native ComfyUI VAE type (red connection)
                "llm": ("ACESTEP_LLM",),
            },
            "optional": {
                "skip_metas": ("BOOLEAN", {
                    "default": False,
                    "label": "Skip BPM/Key/TimeSig (generate caption only)"
                }),
                "only_unlabeled": ("BOOLEAN", {
                    "default": False,
                    "label": "Only process samples without captions"
                }),
                "format_lyrics": ("BOOLEAN", {
                    "default": False,
                    "label": "Format user-provided lyrics with LLM"
                }),
                "transcribe_lyrics": ("BOOLEAN", {
                    "default": False,
                    "label": "Transcribe lyrics from audio"
                }),
                "llm_batch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "label": "LLM batch size"
                }),
                "language": (LANGUAGE_OPTIONS, {
                    "default": "Auto",
                    "label": "Language"
                }),
                "label_guidance": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "label": "Label guidance",
                    "placeholder": "Optional cultural or dataset-specific labelling instructions",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                    "label": "LLM temperature",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "label": "LLM top-p",
                }),
                "top_k": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 200,
                    "step": 5,
                    "label": "LLM top-k",
                }),
                "max_new_tokens": ("INT", {
                    "default": 768,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "label": "LLM max output tokens",
                }),
                "review_path": ("STRING", {
                    "default": "./output/acestep/label_review.json",
                    "multiline": False,
                    "label": "Label review file",
                }),
            }
        }

    RETURN_TYPES = ("ACESTEP_DATASET", "INT", "STRING")
    RETURN_NAMES = ("dataset", "labeled_count", "status")
    FUNCTION = "label"
    CATEGORY = "FL AceStep/Dataset"

    def label(
        self,
        dataset,
        model,  # ComfyUI MODEL (ModelPatcher)
        vae,  # ComfyUI VAE
        llm,
        skip_metas=False,
        only_unlabeled=False,
        format_lyrics=False,
        transcribe_lyrics=False,
        llm_batch_size=2,
        language="Auto",
        label_guidance="",
        temperature=0.2,
        top_p=0.9,
        top_k=30,
        max_new_tokens=768,
        review_path="./output/acestep/label_review.json",
    ):
        """Label all samples in the dataset."""
        logger.info("Starting auto-labeling...")
        logger.info(
            "Label constraints: language=%s, guidance=%s",
            language,
            "provided" if label_guidance.strip() else "none",
        )

        # Verify this is an ACE-Step model
        if not is_acestep_model(model):
            return (dataset, 0, "Error: Model is not an ACE-Step model")

        # Get the tokenizer from the MODEL for audio-to-codes conversion
        tokenizer = get_acestep_tokenizer(model)

        # Use ComfyUI's managed device and load model components as needed.
        import torch
        device = model_management.get_torch_device() if model_management else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        samples = dataset.samples
        if not samples:
            return (dataset, 0, "No samples to label")

        # Filter samples if only_unlabeled
        samples_to_label = []
        for i, sample in enumerate(samples):
            if only_unlabeled and (sample.labeled or sample.caption):
                continue
            samples_to_label.append((i, sample))

        if not samples_to_label:
            return (dataset, 0, "All samples already labeled")

        # Progress bar
        pbar = ProgressBar(len(samples_to_label)) if ProgressBar else None

        labeled_count = 0
        errors = []
        batch_size = max(1, int(llm_batch_size))
        understand_batch = getattr(llm, "understand_audio_from_codes_batch", None)
        language_name = (
            language.split(" — ", 1)[1] if language != "Auto" else None
        )
        metadata_by_index = {}
        code_items = []
        format_items = []

        # Stage 1: keep the diffusion model and VAE on XPU while producing all
        # audio codes. The LLM is not needed for this phase.
        for idx, sample in samples_to_label:
            if format_lyrics and sample.raw_lyrics:
                format_items.append((idx, sample))
                continue

            logger.info(f"Encoding audio to codes for sample {idx}: {sample.filename}")
            try:
                if model_management:
                    model_management.load_models_gpu([vae.patcher])
                    model_management.load_models_gpu([model])
                dtype = next(tokenizer.parameters()).dtype
                codes = audio_to_codes(
                    vae=vae,
                    tokenizer=tokenizer,
                    audio_path=sample.audio_path,
                    device=device,
                    dtype=dtype,
                    max_duration=30.0,
                )
            except Exception as e:
                logger.warning(f"Audio encoding failed for sample {idx}: {e}")
                codes = ""

            if codes:
                logger.info(
                    f"Generated {len(codes)} chars of audio codes for sample {idx}, "
                    "queuing LLM..."
                )
                code_items.append((
                    idx,
                    codes,
                    _build_lyrics_context(sample, language, label_guidance),
                    language,
                    label_guidance,
                ))
            else:
                logger.warning(f"No audio codes for sample {idx}, skipping LLM labeling")

        # Stages 2-4: release the heavy audio models, run the LLM on XPU, then
        # return it to CPU staging so later training can reclaim XPU memory.
        with _llm_generation_device(llm, model, vae):
            total_batches = (len(code_items) + batch_size - 1) // batch_size
            for batch_start in range(0, len(code_items), batch_size):
                batch = code_items[batch_start:batch_start + batch_size]
                try:
                    codes = [item[1] for item in batch]
                    lyrics_contexts = [item[2] for item in batch]
                    language_hints = [item[3] for item in batch]
                    label_guidances = [item[4] for item in batch]
                    if understand_batch is not None:
                        metadata = understand_batch(
                            codes,
                            lyrics_contexts,
                            language_hints,
                            label_guidances,
                            temperature=float(temperature),
                            top_k=int(top_k),
                            top_p=float(top_p),
                            max_new_tokens=int(max_new_tokens),
                        )
                    else:
                        metadata = [
                            llm.understand_audio_from_codes(
                                code,
                                lyrics_context=context,
                                language_hint=language_hint,
                                label_guidance=guidance,
                                temperature=float(temperature),
                                top_k=int(top_k),
                                top_p=float(top_p),
                                max_new_tokens=int(max_new_tokens),
                            )
                            for code, context, language_hint, guidance in zip(
                                codes, lyrics_contexts, language_hints, label_guidances
                            )
                        ]
                    for (idx, _, _, _, _), item_metadata in zip(batch, metadata):
                        metadata_by_index[idx] = item_metadata
                    logger.info(
                        "LLM labelling batch %d/%d complete",
                        batch_start // batch_size + 1,
                        total_batches,
                    )
                except Exception as e:
                    logger.warning(
                        "LLM labelling batch %d/%d failed: %s",
                        batch_start // batch_size + 1,
                        total_batches,
                        e,
                    )

            for idx, sample in format_items:
                try:
                    logger.info(f"Formatting lyrics for sample {idx}: {sample.filename}")
                    metadata_by_index[idx] = llm.format_sample(
                        caption=sample.caption,
                        lyrics=sample.raw_lyrics,
                        instruction_context=_build_lyrics_context(
                            sample, language, label_guidance
                        ),
                        language_hint=language,
                        label_guidance=label_guidance,
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        max_new_tokens=int(max_new_tokens),
                    )
                except Exception as e:
                    logger.warning(f"Error formatting sample {idx}: {e}")

        # Apply generated metadata in the original dataset order.
        for idx, sample in samples_to_label:
            metadata = metadata_by_index.get(idx)
            if metadata is None:
                errors.append(f"Sample {idx}: no metadata generated")
                if pbar:
                    pbar.update(1)
                continue

            try:
                metadata = _enforce_language_metadata(metadata, language)
                if metadata.get("caption"):
                    sample.caption = metadata["caption"]
                if metadata.get("genre"):
                    sample.genre = metadata["genre"]

                if not skip_metas:
                    if metadata.get("bpm") and sample.bpm is None:
                        sample.bpm = metadata["bpm"]
                    if metadata.get("keyscale") and not sample.keyscale:
                        sample.keyscale = metadata["keyscale"]
                    if metadata.get("timesignature"):
                        sample.timesignature = metadata["timesignature"]

                if metadata.get("language"):
                    sample.language = language_name or metadata["language"]
                    sample.is_instrumental = metadata["language"].lower() == "instrumental"
                elif language_name:
                    sample.language = language_name

                if metadata.get("lyrics") and metadata["lyrics"] != "[Instrumental]":
                    if transcribe_lyrics or format_lyrics:
                        sample.lyrics = metadata["lyrics"]
                        sample.formatted_lyrics = metadata["lyrics"]
                        sample.is_instrumental = False

                sample.labeled = True
                labeled_count += 1

                logger.info(
                    f"Sample {idx} labeled: caption='{sample.caption[:60]}...', "
                    f"bpm={sample.bpm}, key={sample.keyscale}"
                )
            except Exception as e:
                error_msg = f"Error labeling sample {idx}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

            if pbar:
                pbar.update(1)

        # Build status message
        status = f"Labeled {labeled_count}/{len(samples_to_label)} samples"
        if errors:
            status += f" ({len(errors)} errors)"

        logger.info(status)

        try:
            review_file = _write_label_review(review_path, samples)
            status += f"; review file: {review_file}"
            logger.info("Label review exported to %s", review_file)
        except Exception as e:
            logger.warning("Could not export label review file: %s", e)

        return (dataset, labeled_count, status)
