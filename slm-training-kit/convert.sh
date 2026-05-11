#!/usr/bin/env bash
# Merge LoRA adapter into base, convert to GGUF, and quantize.
set -euo pipefail

BASE="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER_DIR:-/workspace/output/adapter}"
MERGED="${MERGED_DIR:-/workspace/output/merged}"
GGUF_F16="${GGUF_F16:-/workspace/output/model-f16.gguf}"
GGUF_OUT="${GGUF_OUT:-/workspace/output/model.gguf}"
QUANT="${QUANT:-q4_k_m}"

echo "[convert] merging adapter into base model..."
python - <<PY
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "$BASE", torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "$ADAPTER")
model = model.merge_and_unload()
model.save_pretrained("$MERGED", safe_serialization=True)
AutoTokenizer.from_pretrained("$BASE", trust_remote_code=True).save_pretrained("$MERGED")
print("[convert] merged -> $MERGED")
PY

echo "[convert] converting HF -> GGUF (f16)..."
python /opt/llama.cpp/convert_hf_to_gguf.py "$MERGED" \
  --outfile "$GGUF_F16" --outtype f16

if [ "$QUANT" = "f16" ]; then
  cp "$GGUF_F16" "$GGUF_OUT"
  echo "[convert] kept f16 (no quantization)"
else
  echo "[convert] quantizing -> $QUANT ..."
  /opt/llama.cpp/build/bin/llama-quantize "$GGUF_F16" "$GGUF_OUT" "$QUANT"
fi

# Copy the Modelfile alongside the GGUF so ollama can find it
cp /workspace/Modelfile /workspace/output/Modelfile

echo "[convert] done:"
ls -lah /workspace/output/*.gguf /workspace/output/Modelfile
