#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DEMO_DIR}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DEMO_DIR}/.matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}"

python tools/train_pixelcnn.py --profile h100 --resume auto "$@"
