#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/power-ba"
BIN_DIR="$HOME/.local/bin"

rm -f "$BIN_DIR/power-bi" "$BIN_DIR/power-ba"
rm -rf "$INSTALL_ROOT"

echo "Power BA removed for current user."
