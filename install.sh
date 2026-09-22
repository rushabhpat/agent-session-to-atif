#!/usr/bin/env bash
# Install the `atif` CLI. Idempotent; safe to re-run.
#
#   tools/atif-rs/install.sh
#
# The parsing core is Rust, so installation compiles it once. `uv` handles the
# build and the Python side; the only other requirement is a Rust toolchain.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v uv >/dev/null 2>&1 || {
  echo "atif: needs uv. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

# rustup installs to ~/.cargo/bin, which is not on PATH in a non-login shell.
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null 2>&1 || {
  echo "atif: needs a Rust toolchain. Install it with:" >&2
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" >&2
  exit 1
}

# The interpreter maturin builds against decides the Rust target, and uv only
# offers interpreters matching its own architecture. On Apple silicon an Intel
# uv or python3 earlier on PATH (Homebrew under Rosetta) therefore makes an
# arm64 toolchain cross-compile to x86_64 and fail on a missing std. Prefer a uv
# whose architecture matches cargo's, rather than whichever comes first on PATH.
HOST_ARCH="$(cargo -vV | sed -n 's/^host: //p' | cut -d- -f1)"
case "$HOST_ARCH" in
  aarch64) WANT="arm64" ;;
  *)       WANT="$HOST_ARCH" ;;
esac
UV="$(command -v uv)"
for cand in "$UV" "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
  [ -x "$cand" ] || continue
  if file -b "$cand" 2>/dev/null | grep -q "$WANT"; then UV="$cand"; break; fi
done
file -b "$UV" 2>/dev/null | grep -q "$WANT" || {
  echo "atif: the uv on PATH is $(file -b "$UV" | grep -o 'x86_64\|arm64') but cargo targets" >&2
  echo "  $WANT. Install a matching uv:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

echo "building the Rust core (first run takes a minute)..."
"$UV" tool install --force --quiet "$SRC"

BIN="$("$UV" tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
echo "installed: $BIN/atif  ($("$BIN/atif" --version))"
case ":$PATH:" in
  *":$BIN:"*) echo "ready: run 'atif' to export a session" ;;
  *) echo "note: $BIN is not on PATH. Add it with:"
     echo "  echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.zshrc && exec zsh" ;;
esac
