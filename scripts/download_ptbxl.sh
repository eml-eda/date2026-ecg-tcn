#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST_DIR="${1:-${PROJECT_ROOT}/dataset}"
BASE_URL="${PTBXL_URL:-https://physionet.org/files/ptb-xl/1.0.3/}"

if ! command -v wget >/dev/null 2>&1; then
    echo "Error: wget not found. Please install wget and try again." >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

wget \
    --recursive \
    --no-parent \
    --no-host-directories \
    --cut-dirs=3 \
    --timestamping \
    --continue \
    --reject "index.html*" \
    --directory-prefix "${DEST_DIR}" \
    "${BASE_URL}"

echo
echo "Download completed in: ${DEST_DIR}"
echo "For training you can use: python main.py --root ${DEST_DIR} ..."
