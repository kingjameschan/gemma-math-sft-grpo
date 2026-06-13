"""v3 Stage 1: SFT on Gemma2-2B-IT with GSM8K (DS-CoT format).

Locked configuration (see v3/CLAUDE.md):
  - LoRA r=64, alpha=32, dropout=0, all linear modules
  - constant LR, no warmup (Schulman protocol)
  - bf16, sdpa attention, gradient checkpointing
  - prompt+completion data format → TRL auto-masks prompt tokens via
    native Gemma2 chat template (no template patch needed)
  - eff_batch=16 (per_device=2 * accum=8)
  - max_length=1024
  - save_steps=50

Three runs (lr sweep, see Phase 1 plan):
  python3 v3/train/01_sft.py --lr 1e-4
  python3 v3/train/01_sft.py --lr 5e-4
  python3 v3/train/01_sft.py --lr 1e-3
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
TRAIN_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "train.jsonl"
DEFAULT_MODEL = ROOT / "models" / "gemma-2-2b-it"
CKPT_DIR = ROOT / "v3" / "E2_sft" / "checkpoints"
TRAIN_LOG = ROOT / "v3" / "E2_sft" / "outputs" / "train_log.jsonl"


def lr_str(lr: float) -> str:
    """Format LR for filenames: 1e-4 / 5e-4 / 1e-3 (no leading zero in exp)."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def load_pc_dataset(path: Path) -> Dataset:
    """Load prompt+completion dataset (TRL conversational format)."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def load_base_model(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )


def make_lora_config(r: int = 64, alpha: int = 32) -> LoraConfig:
    """Default: r=64, alpha=32, dropout=0, all linear (Schulman aligned).

    Fastgrid sweep can override r via --lora_r."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def make_sft_config(args, output_dir: Path) -> SFTConfig:
    return SFTConfig(
        output_dir=str(output_dir),
        # Optimization
        num_train_epochs=args.ep,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        # Schulman uses pure constant; we add 3% warmup as a small safety ramp
        # (mainly for the lr=1e-3 high-end run where LoRA's implicit warmup may
        # not be enough). 97% of training is still at constant target LR.
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused",
        # Loss masking is automatic for prompt+completion format
        max_length=args.max_length,
        packing=True,
        # Memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # I/O
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,           # keep all ckpts (we want full trajectory)
        save_only_model=True,            # skip optimizer.pt (664MB/ckpt) — saves disk
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
    )


def _rel(p: Path) -> str:
    """Try to express p relative to ROOT; fall back to absolute."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def print_config_table(args, output_dir: Path, n_train: int, total_steps: int):
    print("\n" + "=" * 64)
    print("v3 SFT Phase 1 — Training Config")
    print("=" * 64)
    print(f"  base model       : {Path(args.model).name}")
    print(f"  train data       : {_rel(Path(args.train_file))}")
    print(f"  n_train          : {n_train}")
    print(f"  output_dir       : {_rel(output_dir)}")
    print(f"  ─ LoRA")
    print(f"    r              : {args.lora_r}")
    print(f"    alpha          : {args.lora_alpha}")
    print(f"    dropout        : 0.0")
    print(f"    target         : q,k,v,o,gate,up,down (all linear)")
    print(f"  ─ Optimization")
    print(f"    lr             : {args.lr:.0e}")
    if args.warmup_steps > 0:
        eff_warmup = args.warmup_steps
        warmup_src = f"warmup_steps={args.warmup_steps} (overrides ratio)"
    else:
        import math
        eff_warmup = math.ceil(total_steps * args.warmup_ratio)
        warmup_src = f"warmup_ratio={args.warmup_ratio} → ceil={eff_warmup}"
    print(f"    schedule       : constant_with_warmup ({warmup_src})")
    print(f"    warmup_steps   : {eff_warmup} of {total_steps}")
    print(f"    optimizer      : AdamW torch fused")
    print(f"    epochs         : {args.ep}")
    print(f"    per_device_bs  : {args.batch_size}")
    print(f"    accum          : {args.accum}")
    print(f"    eff_batch      : {args.batch_size * args.accum}")
    print(f"    total_steps    : ~{total_steps} (unpacked est; packing reduces ~5x — see tqdm)")
    print(f"  ─ Loss / Memory")
    print(f"    loss_mask      : auto (prompt+completion format)")
    print(f"    max_length     : {args.max_length}")
    print(f"    packing        : True (concat samples to max_length)")
    print(f"    bf16           : True")
    print(f"    grad_ckpt      : True")
    print(f"    attn_impl      : flash_attention_2")
    print(f"  ─ I/O")
    print(f"    save_steps     : {args.save_steps}")
    print(f"    n_ckpts        : depends on packed step count (typically ~18 for ep=2)")
    print(f"    logging_steps  : {args.logging_steps}")
    print(f"    seed           : {args.seed}")
    print("=" * 64)


def append_train_log(output_dir: Path, args, train_loss: float, duration_s: float, n_train: int):
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage": "v3_sft",
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "ep": args.ep,
        "eff_batch": args.batch_size * args.accum,
        "n_train": n_train,
        "save_steps": args.save_steps,
        "final_train_loss": round(train_loss, 4),
        "duration_s": round(duration_s, 1),
        "output_dir": _rel(output_dir),
    }
    with open(TRAIN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--train_file", default=str(TRAIN_FILE))
    ap.add_argument("--lr", type=float, required=True,
                    help="learning rate (1e-4, 5e-4, 1e-3 for v3 sweep)")
    ap.add_argument("--ep", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.03,
                    help="warmup as fraction of total steps (3%% default)")
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="absolute warmup steps; if > 0, overrides warmup_ratio")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--save_steps", type=int, default=10,
                    help="default 10 because packing collapses 4.8× → ~180 total steps")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=64,
                    help="LoRA rank (default 64; fastgrid sweeps 8/16/32/64)")
    ap.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha (default 32; keep at 32 for Schulman alignment)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="for sanity test: limit training samples")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="for sanity test: hard cap on steps (overrides epochs)")
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="path to checkpoint-N dir to resume")
    args = ap.parse_args()

    # Auto output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = CKPT_DIR / f"sft_lr{lr_str(args.lr)}_r{args.lora_r}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume check (per CLAUDE.md feedback_resume_check)
    existing_ckpts = sorted(output_dir.glob("checkpoint-*"),
                            key=lambda p: int(p.name.split("-")[1]))
    if existing_ckpts and args.resume_from_checkpoint is None:
        latest = existing_ckpts[-1]
        latest_step = int(latest.name.split("-")[1])
        print(f"\n[resume check] found existing ckpts in {output_dir}:")
        print(f"  latest: {latest.name} (step {latest_step})")
        print(f"  to resume, rerun with: --resume_from_checkpoint {latest}")
        print(f"  to start fresh, delete the directory first.")
        raise SystemExit(1)

    # Data
    ds = load_pc_dataset(Path(args.train_file))
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    # Compute total_steps for config table
    eff_batch = args.batch_size * args.accum
    steps_per_ep = (len(ds) + eff_batch - 1) // eff_batch
    total_steps = args.max_steps if args.max_steps else steps_per_ep * args.ep

    print_config_table(args, output_dir, len(ds), total_steps)

    # Tokenizer + model. Use Gemma2 native chat_template (no patch).
    # TRL detects prompt+completion format and auto-masks prompt tokens.
    print("\n[load] tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_base_model(Path(args.model))
    peft_model = get_peft_model(model, make_lora_config(r=args.lora_r, alpha=args.lora_alpha))
    peft_model.print_trainable_parameters()

    # SFT config
    cfg = make_sft_config(args, output_dir)
    if args.max_steps:
        cfg.max_steps = args.max_steps

    trainer = SFTTrainer(
        model=peft_model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n[train] starting...")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    duration = time.time() - t0
    trainer.save_model(str(output_dir))

    append_train_log(output_dir, args, result.training_loss, duration, len(ds))

    print(f"\ntraining done in {duration / 60:.1f} min")
    print(f"final train loss: {result.training_loss:.4f}")
    print(f"adapter saved to: {_rel(output_dir)}")


if __name__ == "__main__":
    main()
