#!/bin/sh
set -eu

PACKAGE=${IVONAR_PACKAGE:-https://github.com/LuisCode28/ivonar-inference/archive/refs/heads/main.tar.gz}
TORCH_BACKEND=${IVONAR_TORCH_BACKEND:-auto}

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv, which provides Python for Ivonar"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi
    PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Ivonar"
if ! uv tool install --force --python 3.12 --torch-backend "$TORCH_BACKEND" "ivonar-inference @ $PACKAGE"; then
    echo "Installing Ivonar failed. If uv is more than a year old, run 'uv self update' and try again." >&2
    exit 1
fi

BIN="$(uv tool dir --bin)"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) uv tool update-shell >/dev/null 2>&1 || true ;;
esac

echo "Starting Ivonar; next time run: ivonar serve"
exec "$BIN/ivonar" serve "$@"
