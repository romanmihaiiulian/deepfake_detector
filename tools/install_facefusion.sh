#!/usr/bin/env bash
# Install FaceFusion (pinned tag) into its own directory and virtual environment, for the
# attack lab only. It is an external tool: never imported by this project, never
# distributed. License: OpenRAIL-AS. Most swapper models are non-commercial (see
# dependencies/licenses.yml). Use it only on videos of people who gave written consent
# (docs/lab/data-collection.md).
#
#   tools/install_facefusion.sh [TARGET_DIR]      (default: ~/.local/share/deepfake-lab/facefusion)
#   export FACEFUSION_HOME=TARGET_DIR
set -euo pipefail

TAG="3.9.1"
TARGET="${1:-$HOME/.local/share/deepfake-lab/facefusion}"

if [ ! -d "$TARGET/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/facefusion/facefusion "$TARGET"
fi
git -C "$TARGET" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
git -C "$TARGET" checkout --quiet "$TAG"

uv venv --python 3.12 "$TARGET/.venv"
# FaceFusion's installer picks the onnxruntime flavour; "default" includes CoreML on macOS.
(cd "$TARGET" && VIRTUAL_ENV="$TARGET/.venv" PATH="$TARGET/.venv/bin:$PATH" \
  uv pip install --python "$TARGET/.venv/bin/python" pip && \
  "$TARGET/.venv/bin/python" install.py --onnxruntime default --skip-conda)

"$TARGET/.venv/bin/python" "$TARGET/facefusion.py" --version
echo
echo "FaceFusion $TAG installed in $TARGET"
echo "  export FACEFUSION_HOME=$TARGET"
