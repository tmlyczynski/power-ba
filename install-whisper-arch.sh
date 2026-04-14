#!/usr/bin/env bash
set -euo pipefail

MODE="auto"
MODEL="small"

usage() {
  cat <<'EOF'
Usage: ./install-whisper-arch.sh [options]

Options:
  --model <name>     GGML model name (default: small)
  --aur-only         Install only via AUR (yay)
  --source-only      Build whisper.cpp from source (skip AUR)
  -h, --help         Show this help

What it does:
  1) In auto mode: tries AUR package install first.
  2) If AUR fails (for example mirror 404/openblas), falls back to source build.
  3) Ensures whisper-cli is available in ~/.local/bin.
  4) Ensures model file exists under ~/.local/share/power-ba/whisper.cpp/models.
EOF
}

log() {
  echo "[install-whisper] $*"
}

warn() {
  echo "[install-whisper][warn] $*" >&2
}

die() {
  echo "[install-whisper][error] $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        [[ $# -ge 2 ]] || die "--model requires a value"
        MODEL="$2"
        shift 2
        ;;
      --aur-only)
        MODE="aur"
        shift
        ;;
      --source-only)
        MODE="source"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

init_paths() {
  INSTALL_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/power-ba"
  SRC_DIR="$INSTALL_ROOT/whisper.cpp"
  BUILD_DIR="$SRC_DIR/build"
  MODEL_DIR="$SRC_DIR/models"
  MODEL_FILE="$MODEL_DIR/ggml-${MODEL}.bin"
  BIN_DIR="$HOME/.local/bin"
  LOCAL_WHISPER_BIN="$BIN_DIR/whisper-cli"
  TOOLS_VENV="$INSTALL_ROOT/tools-venv"

  mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
}

try_aur_install() {
  if ! command -v yay >/dev/null 2>&1; then
    warn "yay is not installed, skipping AUR install path."
    return 1
  fi

  local model_pkg="whisper.cpp-model-${MODEL}"
  log "Trying AUR install: whisper.cpp + ${model_pkg}"

  if yay -S --needed whisper.cpp "$model_pkg"; then
    log "AUR install completed."
    return 0
  fi

  warn "AUR install failed."
  return 1
}

ensure_build_toolchain() {
  require_cmd git
  require_cmd make
  require_cmd gcc
  require_cmd pkg-config
  require_cmd curl
  require_cmd python3
}

ensure_cmake() {
  if command -v cmake >/dev/null 2>&1; then
    CMAKE_BIN="$(command -v cmake)"
    NINJA_BIN="$(command -v ninja || true)"
    return
  fi

  log "cmake not found system-wide. Installing local cmake via Python venv."
  python3 -m venv "$TOOLS_VENV"
  "$TOOLS_VENV/bin/python" -m pip install --upgrade pip cmake ninja
  CMAKE_BIN="$TOOLS_VENV/bin/cmake"
  NINJA_BIN="$TOOLS_VENV/bin/ninja"
}

build_from_source() {
  ensure_build_toolchain
  ensure_cmake

  if [[ -d "$SRC_DIR/.git" ]]; then
    log "Updating existing whisper.cpp checkout."
    git -C "$SRC_DIR" pull --ff-only
  else
    log "Cloning whisper.cpp sources."
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$SRC_DIR"
  fi

  log "Configuring whisper.cpp build."
  local generator_args=()
  if [[ -n "${NINJA_BIN:-}" ]]; then
    generator_args=("-G" "Ninja")
  fi

  "$CMAKE_BIN" \
    -S "$SRC_DIR" \
    -B "$BUILD_DIR" \
    "${generator_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_TESTS=OFF

  log "Building whisper.cpp."
  "$CMAKE_BIN" --build "$BUILD_DIR" --config Release -j"$(nproc)"

  local built_bin=""
  for candidate in \
    "$BUILD_DIR/bin/whisper-cli" \
    "$BUILD_DIR/bin/main" \
    "$BUILD_DIR/src/whisper-cli"; do
    if [[ -x "$candidate" ]]; then
      built_bin="$candidate"
      break
    fi
  done

  [[ -n "$built_bin" ]] || die "Could not find built whisper executable in build output."

  cp -f "$built_bin" "$LOCAL_WHISPER_BIN"
  chmod +x "$LOCAL_WHISPER_BIN"
  log "Installed whisper-cli to $LOCAL_WHISPER_BIN"
}

ensure_model() {
  if [[ -f "$MODEL_FILE" ]]; then
    log "Model already exists: $MODEL_FILE"
    return
  fi

  local script_path="$SRC_DIR/models/download-ggml-model.sh"
  [[ -x "$script_path" || -f "$script_path" ]] || die "Model download script not found: $script_path"

  log "Downloading GGML model: $MODEL"
  (
    cd "$SRC_DIR/models"
    bash "$script_path" "$MODEL"
  )

  [[ -f "$MODEL_FILE" ]] || die "Model download finished but model file not found: $MODEL_FILE"
}

resolve_whisper_bin() {
  if command -v whisper-cli >/dev/null 2>&1; then
    command -v whisper-cli
    return
  fi
  if [[ -x "$LOCAL_WHISPER_BIN" ]]; then
    echo "$LOCAL_WHISPER_BIN"
    return
  fi
  echo ""
}

print_summary() {
  local whisper_bin
  whisper_bin="$(resolve_whisper_bin)"
  local model_path=""
  if [[ -f "$MODEL_FILE" ]]; then
    model_path="$MODEL_FILE"
  elif [[ -f "/usr/share/whisper.cpp/models/ggml-${MODEL}.bin" ]]; then
    model_path="/usr/share/whisper.cpp/models/ggml-${MODEL}.bin"
  fi

  echo
  log "Installation finished."
  echo "whisper-cli: ${whisper_bin:-NOT FOUND}"
  echo "model:      ${model_path:-NOT FOUND}"
  echo
  echo "If command is not found in new shell, add PATH:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
  echo
  echo "Suggested app start command:"
  if [[ -n "$model_path" ]]; then
    echo "  power-bi start --stt-backend whisper_cpp --whisper-binary whisper-cli --whisper-model-path $model_path"
  else
    echo "  power-bi settings"
    echo "  # then set whisper model path manually"
  fi
}

main() {
  parse_args "$@"
  init_paths

  case "$MODE" in
    aur)
      try_aur_install || die "AUR install failed in --aur-only mode."
      ;;
    source)
      build_from_source
      ;;
    auto)
      if ! try_aur_install; then
        warn "Falling back to source build mode."
        build_from_source
      fi
      ;;
    *)
      die "Unsupported mode: $MODE"
      ;;
  esac

  # Ensure local source tree exists for model download when AUR path did not provide it.
  if [[ ! -d "$SRC_DIR/.git" ]]; then
    log "Preparing local whisper.cpp checkout for model management."
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$SRC_DIR"
  fi

  ensure_model
  print_summary
}

main "$@"
