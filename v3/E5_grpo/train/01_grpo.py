"""v3 E5 Stage 1: GRPO fastgrid on Gemma2-2B-IT with GSM8K (DS-CoT format).

Stage 1 sweeps (LR x beta) crash triage:
  LR ∈ {1e-6, 5e-6, 1e-5, 5e-5}
  beta ∈ {0.01, 0.04, 0.1}
  G=8 (fixed), max_steps=20, save_steps=5

Locked configuration:
  - LoRA r=64, alpha=32, dropout=0, all linear (Schulman aligned)
  - constant_with_warmup (matches v3 SFT, not DSMath which is unspecified)
  - bf16, flash_attention_2, gradient checkpointing
  - reward = rule-based binary boxed_accuracy (E1/E2 same protocol)
  - loss_type = "grpo" (DSMath original; TRL default is "dapo")
  - scale_rewards = "group" (DSMath default)
  - num_iterations = 1 (DSMath spec, single update per generation set)
  - temperature 0.7, top_p 1.0

Single-config usage:
  python3 v3/E5_grpo/train/01_grpo.py --lr 5e-6 --beta 0.04
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[3]
TRAIN_FILE = ROOT / "v3" / "shared" / "data" / "sft" / "train.jsonl"
DEFAULT_MODEL = ROOT / "models" / "gemma-2-2b-it"
CKPT_BASE = ROOT / "v3" / "E5_grpo" / "checkpoints" / "fastgrid" / "stage1"
TRAIN_LOG = ROOT / "v3" / "E5_grpo" / "outputs" / "fastgrid" / "train_log.jsonl"

# Reuse shared answer-extraction helpers (no vllm dep, train-env safe)
sys.path.insert(0, str(ROOT / "v3" / "shared"))
from answer_extraction import extract_answer, math_equal_numerical, gold_from_completion


def lr_str(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def beta_str(beta: float) -> str:
    return f"{beta:g}"


def load_grpo_dataset(path: Path) -> Dataset:
    """Load v3 train.jsonl and convert to GRPO format (prompt + answer)."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt": r["prompt"],
                "answer": gold_from_completion(r["completion"]),
            })
    return Dataset.from_list(rows)


def boxed_accuracy_reward(completions, answer, **kwargs):
    """Rule-based binary reward.

    1.0 if extract_answer(completion) numerically equals gold answer; else 0.0.
    Uses the exact same DS 5-layer extraction + math_equal_numerical as E1/E2 eval.
    """
    rewards = []
    for c, gold in zip(completions, answer):
        text = c if isinstance(c, str) else c[0]["content"]
        pred = extract_answer(text)
        rewards.append(1.0 if math_equal_numerical(pred, gold) else 0.0)
    return rewards


def load_base_model(model_path: Path):
    # Try flash_attention_2 first; fall back to sdpa if unavailable (e.g. inside
    # vllm-openai docker image which doesn't bundle flash-attn).
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    print(f"[model] attn_implementation = {attn}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=attn,
    )


def make_lora_config(r: int = 64, alpha: int = 32) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def make_grpo_config(args, output_dir: Path) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(output_dir),
        # ---- GRPO algorithm ----
        learning_rate=args.lr,
        beta=args.beta,                                # KL coefficient
        loss_type="grpo",                              # DSMath original
        scale_rewards="group",                         # DSMath default
        num_iterations=1,                              # single update per gen set
        epsilon=0.2,                                   # PPO clip (TRL default)
        # ---- generation ----
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        # max_prompt_length added in TRL 0.29; on 0.28 prompts are truncated
        # by the tokenizer's default. Our DS-CoT prompts are ~p99 200 tokens
        # so default truncation is fine.
        temperature=args.temperature,
        top_p=args.top_p,
        # ---- schedule (matches v3 SFT, Schulman protocol) ----
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        # ---- training ----
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        # ---- memory ----
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # ---- I/O ----
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        save_only_model=True,
        logging_steps=args.logging_steps,
        log_completions=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def print_config_table(args, output_dir: Path, n_train: int):
    print("\n" + "=" * 64)
    print("v3 E5 GRPO Stage 1 — Training Config")
    print("=" * 64)
    print(f"  base model       : {Path(args.model).name}")
    print(f"  train data       : {_rel(Path(args.train_file))} ({n_train} prompts)")
    print(f"  output_dir       : {_rel(output_dir)}")
    print(f"  ─ LoRA")
    print(f"    r / alpha      : {args.lora_r} / {args.lora_alpha}")
    print(f"    target         : all linear, dropout=0")
    print(f"  ─ GRPO")
    print(f"    learning_rate  : {args.lr:g}")
    print(f"    beta (kl_coef) : {args.beta:g}")
    print(f"    loss_type      : grpo (DSMath original)")
    print(f"    scale_rewards  : group (DSMath default)")
    print(f"    num_iterations : 1 (single update per gen set)")
    print(f"    epsilon (clip) : 0.2")
    print(f"    group_size G   : {args.group_size}")
    print(f"    temperature    : {args.temperature}")
    print(f"    top_p          : {args.top_p}")
    print(f"    max_completion : {args.max_new_tokens}")
    print(f"    max_prompt     : {args.max_prompt_length}")
    print(f"  ─ Schedule (Schulman, matches v3 SFT)")
    print(f"    scheduler      : constant_with_warmup")
    print(f"    warmup_steps   : {args.warmup_steps}")
    print(f"    optimizer      : AdamW torch fused (wd=0.01)")
    print(f"    max_steps      : {args.max_steps}")
    print(f"  ─ Batch")
    print(f"    per_device_bs  : {args.batch_size} prompts/device")
    print(f"    accum          : {args.accum}")
    print(f"    eff_batch      : {args.batch_size * args.accum} prompts/step")
    print(f"    rollouts/step  : {args.batch_size * args.accum * args.group_size} generations")
    print(f"  ─ Memory")
    print(f"    bf16 + grad_ckpt + flash_attn_2")
    print(f"  ─ I/O")
    print(f"    save_steps     : {args.save_steps} (→ {args.max_steps // args.save_steps} ckpts)")
    print(f"    logging_steps  : {args.logging_steps}")
    print(f"    seed           : {args.seed}")
    print("=" * 64)


def append_train_log(output_dir: Path, args, duration_s: float, n_train: int):
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage": "v3_grpo_stage1",
        "lr": args.lr,
        "beta": args.beta,
        "group_size": args.group_size,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "eff_batch_prompts": args.batch_size * args.accum,
        "rollouts_per_step": args.batch_size * args.accum * args.group_size,
        "n_train_prompts": n_train,
        "duration_s": round(duration_s, 1),
        "output_dir": _rel(output_dir),
    }
    with open(TRAIN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--train_file", default=str(TRAIN_FILE))
    # GRPO core
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True, help="KL coefficient")
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    # Schedule
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=5)
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=3)
    # Batch
    ap.add_argument("--batch_size", type=int, default=2,
                    help="per-device batch (prompts); batch * accum must be divisible by group_size")
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation steps")
    # LoRA
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=32)
    # I/O
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Validate batch * accum divisible by group_size
    eff_batch = args.batch_size * args.accum
    if eff_batch % args.group_size != 0:
        raise SystemExit(
            f"per_device_batch ({args.batch_size}) * accum ({args.accum}) = {eff_batch} "
            f"must be divisible by group_size ({args.group_size})"
        )

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = CKPT_BASE / f"lr{lr_str(args.lr)}_b{beta_str(args.beta)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if final ckpt exists
    final_ckpt = output_dir / f"checkpoint-{args.max_steps}"
    if final_ckpt.exists():
        print(f"[done] {final_ckpt} already exists; skipping")
        return

    # Resume check
    existing = sorted(output_dir.glob("checkpoint-*"),
                      key=lambda p: int(p.name.split("-")[1]))
    if existing:
        latest = existing[-1]
        latest_step = int(latest.name.split("-")[1])
        print(f"[resume check] found existing ckpt: {latest.name} (step {latest_step})")
        print(f"  to resume, use --resume_from_checkpoint {latest}")
        print(f"  to start fresh, delete the directory first")
        # For Stage 1 fastgrid, default behavior = exit (avoid accidental overwrite)
        raise SystemExit(1)

    # Data
    print("\n[load] dataset...")
    ds = load_grpo_dataset(Path(args.train_file))

    print_config_table(args, output_dir, len(ds))

    # Model
    print("\n[load] tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_base_model(Path(args.model))
    peft_model = get_peft_model(model, make_lora_config(r=args.lora_r, alpha=args.lora_alpha))
    peft_model.print_trainable_parameters()

    # Trainer
    cfg = make_grpo_config(args, output_dir)
    trainer = GRPOTrainer(
        model=peft_model,
        reward_funcs=[boxed_accuracy_reward],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("\n[train] starting GRPO...")
    t0 = time.time()
    trainer.train()
    duration = time.time() - t0
    trainer.save_model(str(output_dir))

    append_train_log(output_dir, args, duration, len(ds))
    print(f"\ntraining done in {duration / 60:.1f} min")
    print(f"adapter saved to: {_rel(output_dir)}")


if __name__ == "__main__":
    main()
