"""Download official PulseDB v2 subject files from the public Google Drive.

Examples
--------
Download five MIMIC subjects for a smoke test:
python scripts/download_pulsedb_gdrive.py --output D:/Datasets/PulseDB --source mimic --limit 5

Download all official subject files:
python scripts/download_pulsedb_gdrive.py --output D:/Datasets/PulseDB --source all --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown


PULSEDB_FOLDER = "https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV?usp=sharing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("mimic", "vital", "all"), default="mimic")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Download all matching files")
    args = parser.parse_args()

    entries = gdown.download_folder(
        url=PULSEDB_FOLDER,
        output=str(args.output),
        quiet=True,
        skip_download=True,
    )
    if entries is None:
        raise RuntimeError("Could not list the official PulseDB Google Drive folder")

    source_tokens = {
        "mimic": ("Segment_Files\\PulseDB_MIMIC", "Segment_Files/PulseDB_MIMIC"),
        "vital": ("Segment_Files\\PulseDB_Vital", "Segment_Files/PulseDB_Vital"),
        "all": ("Segment_Files",),
    }
    tokens = source_tokens[args.source]
    files = [entry for entry in entries if str(entry.path).endswith(".mat") and any(t in str(entry.path) for t in tokens)]
    if not args.all:
        files = files[: max(args.limit, 0)]

    print(f"Selected {len(files)} files from the official PulseDB folder")
    for position, entry in enumerate(files, start=1):
        relative = Path(str(entry.path))
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[{position}/{len(files)}] exists: {destination.name}")
            continue
        print(f"[{position}/{len(files)}] downloading: {destination.name}")
        gdown.download(id=entry.id, output=str(destination), quiet=False, resume=True)


if __name__ == "__main__":
    main()

