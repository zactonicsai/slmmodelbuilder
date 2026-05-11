#!/usr/bin/env bash
# One-command pipeline: train -> merge -> GGUF -> quantize.
set -euo pipefail

echo "==> Step 1/2: LoRA fine-tune"
python train.py

echo "==> Step 2/2: Merge + convert to GGUF + quantize"
bash convert.sh

echo ""
echo "All done. Next steps (run on the host):"
echo "  docker compose up -d ollama"
echo "  docker compose exec ollama ollama create my-slm -f /models/Modelfile"
echo "  docker compose exec ollama ollama run my-slm"
