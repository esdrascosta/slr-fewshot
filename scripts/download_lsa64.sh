#!/usr/bin/env bash
# Download and unpack the LSA64 dataset ("raw" cut version, ~1.9 GB).
# Usage: bash scripts/download_lsa64.sh [target_dir]   (default: data/lsa64)
set -euo pipefail

TARGET="${1:-data/lsa64}"
mkdir -p "${TARGET}/videos"

# Known mirrors for the LSA64 raw archive. URLs occasionally move; the python
# downloader tries them in order and verifies the zip.
python -m src.data.download --out "${TARGET}"

echo "Videos available in ${TARGET}/videos:"
ls "${TARGET}/videos" | head
echo "Total: $(ls "${TARGET}/videos" | wc -l) files (expected 3200)"
