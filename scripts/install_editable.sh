#!/usr/bin/env bash
set -euo pipefail

EXTRA="${1:-}"

if [[ -z "${EXTRA}" ]]; then
    echo "Usage: bash scripts/install_editable.sh <base|pseudo|gen|dpo|all>" >&2
    exit 1
fi

case "${EXTRA}" in
    base)
        pip install flash-attn==2.8.2 --no-build-isolation
        pip install -e ".[base]"
        ;;
    pseudo|gen|dpo|all)
        pip install -e ".[${EXTRA}]"
        ;;
    *)
        echo "Unknown extra: ${EXTRA}" >&2
        exit 1
        ;;
esac
