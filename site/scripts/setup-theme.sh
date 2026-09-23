#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THEME_DIR="$ROOT/themes/kraiklyn"

if [ -d "$THEME_DIR/.git" ]; then
  git -C "$THEME_DIR" pull --ff-only
elif [ -e "$THEME_DIR" ]; then
  echo "Error: $THEME_DIR exists but is not a git checkout." >&2
  exit 1
else
  git clone --depth 1 https://github.com/jsnjack/kraiklyn.git "$THEME_DIR"
fi

echo "Kraiklyn is installed at $THEME_DIR"
