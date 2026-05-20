#!/usr/bin/env zsh
set -e

SNAP_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/bb5d59e06d9551d752d08b292a50eb208b07ab1f"
BASE_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main"

mkdir -p "$SNAP_DIR"

FILES=(
    "qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf"
    "qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf"
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
)

echo "=== Downloading GGUF shards ==="
for f in $FILES; do
    DEST="$SNAP_DIR/$f"
    if [ -f "$DEST" ]; then
        echo "  $f already exists, skipping"
        continue
    fi
    echo "  Downloading $f..."
    curl -L -o "$DEST.tmp" "$BASE_URL/$f"
    mv "$DEST.tmp" "$DEST"
    echo "  Done: $f"
done

echo ""
echo "=== All shards downloaded ==="
ls -lh "$SNAP_DIR"

echo ""
echo "=== Running llamacpp-q8 backend ==="
cd /Users/mediratta/code/hiring/llm-backend-divergence
uv run divergence run --backends llamacpp-q8 --datasets gsm8k,mmlu,canary --output results/run.db --resume

echo ""
echo "=== Running llamacpp-q4km backend ==="
uv run divergence run --backends llamacpp-q4km --datasets gsm8k,mmlu,canary --output results/run.db --resume

echo ""
echo "=== Running torch-mps backend ==="
uv run divergence run --backends torch-mps --datasets gsm8k,mmlu,canary --output results/run.db --resume || echo "torch-mps failed (expected on 24GB)"

echo ""
echo "=== All done ==="
uv run divergence summarize results/run.db
