
# SLM Data Generator

A single-file, browser-based tool for building JSONL training datasets to fine-tune Small Language Models (SLMs) like Qwen2.5, SmolLM2, TinyLlama, or Llama-3.2 — then turning that dataset into a complete Docker-based training pipeline that produces a GGUF model ready to serve with Ollama.

No installation, no build step, no internet required after first load. Open `slm-data-generator.html` in any modern browser and start working.


**Pros**
- Low cost & fast inference (laptop/GPU).
- Strong at repetitive tasks (Terraform, Java DTOs/Lombok/Controllers/Services, basic security).
- JSONL + LoRA specializes efficiently.
- ChromaDB RAG sharply limits hallucinations.

**Cons + Resolutions**
- Weak complex reasoning → Add CoT/few-shot in JSONL; hybrid larger LLM fallback.
- Residual hallucinations → Reranking, guardrails, code validation.
- Dataset effort → Synthetic JSONL generation.
- RAG latency → Fast embedder + caching.
- Setup complexity → LangChain templates; automated DB updates.

## Quick start

1. Double-click `slm-data-generator.html` (or drag it into any browser tab).
2. The dataset comes pre-seeded with **47 example chat pairs** covering Python classes/functions, Terraform AWS/Azure, Kubernetes, CI/CD pipelines, and network/security configurations.
3. Add your own examples in the **Add entry** tab, or edit/delete the defaults.
4. Go to the **Deploy** tab, configure your training run, and click **Download training kit (.zip)**.
5. Unzip and run `docker compose run --rm trainer bash run_all.sh`.
6. After training finishes, register the resulting GGUF with Ollama and chat with your model.

NOTE: a sample export is included in the `slm-training-kit` subdirectory of this repo.

---

## What it does

Three things, in one HTML file:

1. **Authoring** — a clean form-based editor for OpenAI/HuggingFace chat-format training data.
2. **Validation** — a built-in retrieval simulator so you can sanity-check what a model would learn from your examples before committing to a training run.
3. **Deployment** — one click generates a complete, runnable Docker training kit: Dockerfile, `docker-compose.yml`, LoRA fine-tuning script, GGUF converter, Modelfile for Ollama, and a README — all configured to your chosen base model and hyperparameters.

Everything happens in the browser. No data leaves your machine.

---

## The tabs

### 1. Add entry

- **User message** — what a user would type to your model
- **Assistant message** — the response you want the model to learn to produce
- **System prompt (optional)** — biases this specific example's behavior; usually leave blank and set the system prompt once in the Modelfile instead
- Live character and approximate-token counts under each field
- **Save entry** appends to the dataset and clears the form
- **Load examples** appends the default 47 examples (deduped, so safe to click twice)

### 2. Dataset

- Searchable table of every entry
- Per-row `edit` link loads the entry back into the form for modification
- Per-row `del` link deletes one entry (with confirmation)
- **Delete all** wipes the dataset
- **Reset to defaults** replaces whatever you have with the 47 default examples

### 3. Simulate

A miniature retrieval-based "model" that responds using the closest matching entry in your dataset (Jaccard similarity over tokens). It's not a real LLM — but it's useful for:

- **Finding gaps**: if the simulator can't answer a reasonable question, the trained model probably won't either
- **Stress-testing**: turn up temperature to see what alternative matches exist for ambiguous queries
- **Quick smoke tests** before kicking off a training run

Controls:
- **Similarity threshold** — how close a match must be to "answer"
- **Temperature** — 0 picks the single best match; >0 picks from top-K randomly

### 4. Import / Export

- Export as **JSONL** (one chat per line — what training scripts expect)
- Export as **JSON array** (single document, useful for human inspection)
- **Copy to clipboard** for quick paste
- **Import** auto-detects format (JSONL, JSON array, or single object) and dedupes on insert
- Live preview of the first 5 entries

### 5. Deploy

Configure everything about your training run, then download a complete kit:

| Setting | What it does |
|---------|--------------|
| **Base model** | The model you're fine-tuning. SmolLM2-360M is the smallest (CPU-trainable). Qwen2.5-0.5B is the recommended starter. Llama-3.2 requires a Hugging Face token. |
| **Ollama model name** | What you'll type after `ollama run` |
| **System prompt** | Baked into the generated Modelfile |
| **Quantization** | `q4_k_m` (recommended balance), `q5_k_m` (higher quality), `q8_0` (near-lossless), `f16` (no quantization) |
| **LoRA r / alpha** | Defaults of 8 / 16 work for most starter datasets |
| **Epochs / batch / LR** | Defaults of 3 / 2 / 2e-4 are sensible starting points |
| **Hardware target** | CUDA (NVIDIA GPU) or CPU. CPU works anywhere but is slow. |
| **Bundle current dataset** | When checked, your data ships inside the ZIP as `data/training.jsonl` |
| **Include Ollama service** | Adds an Ollama container to the compose file so you can chat with the model immediately |

A **live preview** at the bottom lets you read each generated file before downloading.

---

## End-to-end workflow

Going from "I have some example Q&A pairs" to "I'm chatting with a fine-tuned model in Ollama":

1. **Open the tool** — defaults are pre-loaded
2. **Curate your dataset** — keep defaults that fit your domain, delete what doesn't, and add your own examples; aim for at least 30–50 high-quality, consistent examples for a noticeable behavior shift, 200+ for strong specialization
3. **Test in Simulate** — verify the retrieval baseline answers common questions reasonably; if it doesn't, you have coverage gaps
4. **Configure Deploy tab** — small base (Qwen2.5-0.5B-Instruct is a great default), CPU or CUDA, `q4_k_m` quantization
5. **Download the ZIP** — unzip somewhere convenient
6. **Build and train**:
   ```bash
   docker compose run --rm trainer bash run_all.sh
   ```
   First build takes ~5–10 minutes (image layers + llama.cpp compile). The actual training takes 10–60 minutes depending on hardware and base model size.
7. **Register the GGUF with Ollama**:
   ```bash
   docker compose up -d ollama
   docker compose exec ollama ollama create my-slm -f /models/Modelfile
   ```
8. **Chat with your model**:
   ```bash
   docker compose exec ollama ollama run my-slm
   ```
   Or via HTTP API:
   ```bash
   curl http://localhost:11434/api/generate -d '{"model":"my-slm","prompt":"hello"}'
   ```

---

## Output format

Each JSONL line is a complete chat conversation:

```json
{"messages":[{"role":"user","content":"What does this do?"},{"role":"assistant","content":"It prints hello."}]}
```

If you set a system prompt on an entry, it becomes the first message:

```json
{"messages":[{"role":"system","content":"You are a Python tutor."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

This format is compatible with:
- `trl.SFTTrainer` (used by the generated `train.py`)
- OpenAI's fine-tuning API
- Most HuggingFace training pipelines that accept chat-format datasets

---

## What makes good training data

A few principles that matter much more than total volume:

- **Consistency** — if the same question gets different answers across examples, the model learns nothing useful
- **Coverage** — span the styles and topics you actually want the model to handle, not just the easy ones
- **Realism** — the user side should look like real prompts (typos, terseness, ambiguity, all OK); over-formatted "perfect" prompts make a brittle model
- **Length variety** — if every example is long, the model will always answer at length; mix short and long responses
- **Explicit refusals** — if you want the model to decline certain things, include examples of declining; otherwise it'll never learn the boundary

For a task-specific fine-tune with LoRA, **100–1000 high-quality examples** is usually enough.

---

## Storage

All data is stored in browser `localStorage` under key `slm_training_data_v1`. It survives reloads but is per-browser-per-origin — opening the file from a different path or in a different browser starts fresh.

**Back up your work** with **Export → JSONL** before clearing browser data or switching machines.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Don't see the example dataset on first load | The tool seeds defaults once per browser; click **Reset to defaults** in the Dataset tab to force-load them |
| `pytorch/pytorch:X.X.X-cpu` image not found | Regenerate the kit from a current build of the tool — older kits referenced a tag that doesn't exist on Docker Hub |
| `nvidia-container-cli: WSL environment detected but no adapters were found` | You selected CUDA but don't have a working GPU. Switch **Hardware target** to **CPU only** and regenerate the kit |
| Build takes forever on CPU | Pick the smallest base (`SmolLM2-360M-Instruct`) and reduce epochs to 1–2 |
| HF 401 / "unauthorized" errors | The base model is gated (Llama-3.2). Add `HF_TOKEN=hf_xxxxx` to the kit's `.env` file |
| Theme toggle doesn't persist | The tool stores theme in `localStorage` key `slm_theme`; check that your browser allows localStorage for `file://` URLs |
| File didn't import | Check that each line is valid JSON. The importer accepts JSONL (one chat per line), a JSON array of chats, or a single chat object — but malformed lines are skipped silently |

The generated training kit ships with its own troubleshooting section in its README — that one's regenerated based on your config and includes the exact commands you need for your setup.

---

## Tech notes

- **Tailwind CSS** via CDN — no build step, just styling
- **JSZip** via CDN — for packaging the training kit
- **No external API calls** — everything runs in your browser
- **No telemetry**, no analytics, no tracking
- Theme (light/dark) persists across sessions
- Single self-contained HTML file (~125 KB)

Works in any modern browser (Chrome, Firefox, Safari, Edge). Tested with the file opened both via `file://` and served over HTTP.

---

## Files

- `slm-data-generator.html` — the tool itself; the only file you need
- `README.md` — this file



# SLM Training Kit

Fine-tune **Qwen/Qwen2.5-0.5B-Instruct** with LoRA on your custom dataset, convert to GGUF, and serve through Ollama — all inside Docker.

Generated from the SLM Data Generator with 52 training examples.

---

## Prerequisites

- Docker + Docker Compose v2
- CPU-only training is **slow** — expect 10–30+ minutes for a 500MB model on a small dataset
- ~10–20 GB free disk (model cache + outputs)
- For gated base models (Llama, etc.), a Hugging Face access token

## Quick start

```bash
# 1. Configure (optional — defaults are fine)
cp .env.example .env
# edit .env to add HF_TOKEN if you're using a gated model

# 2. Build image and run the full pipeline (train + convert)
docker compose run --rm trainer bash run_all.sh

# 3. Start Ollama and register the fine-tuned GGUF
docker compose up -d ollama
docker compose exec ollama ollama create my-slm -f /models/Modelfile

# 4. Chat with your model
docker compose exec ollama ollama run my-slm

# Or via HTTP API
curl http://localhost:11434/api/generate -d '{
  "model": "my-slm",
  "prompt": "Hello!",
  "stream": false
}'
```

## What each file does

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Defines the `trainer` and `ollama` services with shared volumes |
| `Dockerfile` | PyTorch + transformers + PEFT + TRL + llama.cpp toolchain |
| `train.py` | Loads JSONL, applies the chat template, LoRA-fine-tunes the base |
| `convert.sh` | Merges adapter into base, converts to GGUF, quantizes |
| `run_all.sh` | Calls train.py then convert.sh |
| `Modelfile` | Ollama model definition (FROM, TEMPLATE, SYSTEM, PARAMETER) |
| `data/training.jsonl` | Your training data (one chat per line) |
| `output/` | Receives adapter, merged model, and final GGUF |

## Configuration

Current settings (edit `.env` or docker-compose.yml to change):

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Hardware | CPU |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-4 |
| Quantization | `q4_k_m` |
| Ollama name | `my-slm` |

## Run individual steps

Train only:
```bash
docker compose run --rm trainer python train.py
```

Convert only (after training):
```bash
docker compose run --rm trainer bash convert.sh
```

## Iterating on your dataset

1. Edit `data/training.jsonl` (or regenerate it from the data generator UI)
2. Re-run the pipeline: `docker compose run --rm trainer bash run_all.sh`
3. Update the Ollama model: `docker compose exec ollama ollama create my-slm -f /models/Modelfile`

## Troubleshooting

- **OOM during training**: lower `BATCH` to 1 in `.env`, or pick a smaller base model
- **"401 unauthorized" downloading the base**: set `HF_TOKEN` in `.env`
- **Slow on CPU**: switch to the smallest base (`SmolLM2-360M-Instruct`) and ≤2 epochs
- **`nvidia-smi` not found in container**: install the NVIDIA Container Toolkit on the host

## What gets produced

After a successful run, `output/` contains:
- `adapter/` — the trained LoRA weights (small, ~10–100 MB)
- `merged/` — full model with LoRA merged in (HF format)
- `model-f16.gguf` — unquantized GGUF
- `model.gguf` — final quantized GGUF (this is what Ollama loads)
- `Modelfile` — copied alongside for ollama create

