"""
LoRA fine-tune a small language model on chat JSONL data.

Reads:   /workspace/data/training.jsonl
Writes:  /workspace/output/adapter/

Each JSONL line:
  {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
Optional "system" role at the start of messages is supported.
"""
import os, json, argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            records.append(json.loads(ln))
    return records

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",       default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--data",       default="/workspace/data/training.jsonl")
    p.add_argument("--output_dir", default="/workspace/output/adapter")
    p.add_argument("--epochs",     type=int,   default=int(os.environ.get("EPOCHS", 3)))
    p.add_argument("--batch",      type=int,   default=int(os.environ.get("BATCH", 2)))
    p.add_argument("--lr",         type=float, default=float(os.environ.get("LR", 2e-4)))
    p.add_argument("--lora_r",     type=int,   default=int(os.environ.get("LORA_R", 8)))
    p.add_argument("--lora_alpha", type=int,   default=int(os.environ.get("LORA_ALPHA", 16)))
    args = p.parse_args()

    print(f"[train] base={args.base}")
    print(f"[train] data={args.data}")
    records = load_jsonl(args.data)
    print(f"[train] {len(records)} training examples")
    if len(records) < 5:
        print("[train] WARNING: very small dataset; results will be poor.")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_row(ex):
        msgs = ex["messages"]
        # If no system, prepend one so the chat template is well-formed for inference too.
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": "You are a helpful, concise assistant."}] + msgs
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list(records).map(format_row, remove_columns=["messages"])

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        report_to="none",
        max_seq_length=1024,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train] adapter saved -> {args.output_dir}")

if __name__ == "__main__":
    main()
