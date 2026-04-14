#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/power-ba"
VENV_DIR="$INSTALL_ROOT/venv"
BIN_DIR="$HOME/.local/bin"

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install --upgrade "$PROJECT_DIR"

ln -sfn "$VENV_DIR/bin/power-bi" "$BIN_DIR/power-bi"
ln -sfn "$VENV_DIR/bin/power-ba" "$BIN_DIR/power-ba"

cat <<EOF
Installed Power BA globally for this user.
Commands:
  $BIN_DIR/power-bi
  $BIN_DIR/power-ba

If command is not found, add this to your shell profile:
  export PATH="$BIN_DIR:\$PATH"
EOF
