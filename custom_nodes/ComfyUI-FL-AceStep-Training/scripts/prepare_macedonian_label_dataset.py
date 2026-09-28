#!/usr/bin/env python3
"""Prepare song-grouped JSONL data for Macedonian label-model training."""

import argparse
import hashlib
import json
from pathlib import Path


LYRIC_FILES = {
    "mk": "mk_lyrics.md",
    "mktl": "mktl_lyrics.md",
    "en": "en_lyrics.md",
}
SUPPORTED_AUDIO = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def split_for_song(song_id, seed):
    digest = hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def load_review(path):
    if not path:
        return {}
    review_path = Path(path).expanduser()
    if not review_path.exists():
        raise ValueError(f"Review file not found: {review_path}")
    entries = json.loads(review_path.read_text(encoding="utf-8"))
    return {
        str(Path(entry["audio_path"]).expanduser().resolve()): entry
        for entry in entries
        if entry.get("audio_path")
    }


def prepare(dataset_root, output_path, review_path=None, seed=20260923):
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")

    review = load_review(review_path)
    records = []
    stats = {
        "song_folders": 0,
        "audio_files": 0,
        "songs_with_mk_lyrics": 0,
        "songs_with_transliteration": 0,
        "songs_with_english": 0,
        "review_matches": 0,
    }

    for song_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        audio_files = sorted(
            path for path in song_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO
        )
        if not audio_files:
            continue

        lyrics = {
            language: read_text(song_dir / filename)
            for language, filename in LYRIC_FILES.items()
        }
        split = split_for_song(song_dir.name, seed)
        stats["song_folders"] += 1
        stats["audio_files"] += len(audio_files)
        stats["songs_with_mk_lyrics"] += bool(lyrics["mk"])
        stats["songs_with_transliteration"] += bool(lyrics["mktl"])
        stats["songs_with_english"] += bool(lyrics["en"])

        for audio_path in audio_files:
            resolved_audio = str(audio_path.resolve())
            reviewed = review.get(resolved_audio)
            if reviewed:
                stats["review_matches"] += 1

            record = {
                "id": f"{song_dir.name}/{audio_path.stem}",
                "song_id": song_dir.name,
                "artist_rendition": audio_path.stem,
                "audio_path": resolved_audio,
                "split": split,
                "language": "mk",
                "lyrics": lyrics,
                "label_status": "approved" if reviewed and reviewed.get("approved") else "unreviewed",
                "labels": {
                    "caption": reviewed.get("caption", "") if reviewed else "",
                    "genre": reviewed.get("genre", "") if reviewed else "",
                    "bpm": reviewed.get("bpm") if reviewed else None,
                    "keyscale": reviewed.get("keyscale", "") if reviewed else "",
                    "timesignature": reviewed.get("timesignature", "") if reviewed else "",
                    "instruments": reviewed.get("instruments", []) if reviewed else [],
                    "rhythm_form": reviewed.get("rhythm_form", "") if reviewed else "",
                    "region": reviewed.get("region", "") if reviewed else "",
                },
            }
            records.append(record)

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["records"] = len(records)
    stats["seed"] = seed
    stats["dataset_root"] = str(root)
    stats_path = output.with_suffix(".summary.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats, output, stats_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("output_path")
    parser.add_argument("--review", default=None)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    stats, output, stats_path = prepare(
        args.dataset_root,
        args.output_path,
        review_path=args.review,
        seed=args.seed,
    )
    print(json.dumps({"output": str(output), "summary": str(stats_path), **stats}, indent=2))


if __name__ == "__main__":
    main()
