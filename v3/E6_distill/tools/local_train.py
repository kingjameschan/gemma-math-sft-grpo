"""Local SFT on RTX 5080 16G: distill OpenMathInstruct-2 → Gemma2-2B-IT.
   Accel: flash-attn-2 + packing + bf16 + grad-checkpoint + 8bit adam (bnb).
   --max_steps for speed probe; omit for full run.
"""
import argparse, json, time
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import os
# siren is Windows python.exe → needs D:\ paths, not /mnt/d
WIN = os.environ.get('USE_WIN_PATHS', '1') == '1'
if WIN:
    ROOT = Path(r'D:\fine-tuning')
else:
    ROOT = Path('/mnt/d/fine-tuning')
TRAIN_FILE = ROOT / 'v3/E6_distill/data/distill_train_100k.jsonl'
MODEL = ROOT / 'models/gemma-2-2b-it'
CKPT_DIR = ROOT / 'v3/E6_distill/checkpoints'


def load_pc(path, limit=None):
    rows = []
    for i, l in enumerate(open(path)):
        if limit and i >= limit:
            break
        r = json.loads(l)
        rows.append({'prompt': r['prompt'], 'completion': r['completion']})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--ep', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--max_length', type=int, default=768)
    ap.add_argument('--train_file', default=str(TRAIN_FILE))
    ap.add_argument('--save_steps', type=int, default=1000)
    ap.add_argument('--logging_steps', type=int, default=5)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_steps', type=int, default=-1)
    ap.add_argument('--limit', type=int, default=None, help='cap dataset rows (probe)')
    ap.add_argument('--opt', default='adamw_8bit', help='adamw_torch_fused | adamw_8bit')
    ap.add_argument('--model', default=str(MODEL), help='base model path')
    ap.add_argument('--output_dir', default=str(CKPT_DIR), help='ckpt output dir')
    ap.add_argument('--resume', action='store_true',
                    help='resume from latest checkpoint-* in output_dir if any exist')
    args = ap.parse_args()

    ckpt_dir = Path(args.output_dir)
    print(f'[load] dataset {args.train_file} (limit={args.limit})', flush=True)
    print(f'[load] base {args.model} -> ckpt {ckpt_dir}', flush=True)
    ds = load_pc(args.train_file, limit=args.limit)
    print(f'[load] {len(ds)} examples', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='flash_attention_2')

    lora = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.0,
                      target_modules=['q_proj','k_proj','v_proj','o_proj',
                                      'gate_proj','up_proj','down_proj'],
                      bias='none', task_type='CAUSAL_LM')

    cfg = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.ep,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=args.warmup_ratio,
        optim=args.opt,
        max_length=args.max_length,
        packing=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=4,
        logging_steps=args.logging_steps, seed=args.seed,
        report_to=[], dataloader_num_workers=4,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
    eff = args.batch_size * args.accum
    print(f'[train] eff_batch={eff} opt={args.opt} packing=True flash_attn=2', flush=True)
    # auto-resume: only if --resume AND a checkpoint-* actually exists (passing
    # resume_from_checkpoint=True with no checkpoint would raise).
    has_ckpt = any(ckpt_dir.glob('checkpoint-*'))
    resume = args.resume and has_ckpt
    print(f'[train] resume={resume} (flag={args.resume} has_ckpt={has_ckpt})', flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    dt = time.time() - t0
    print(f'[train] done {dt:.0f}s', flush=True)
    if args.max_steps > 0:
        print(f'[probe] {args.max_steps} steps in {dt:.1f}s = {dt/args.max_steps:.2f}s/step', flush=True)
    else:
        trainer.save_model(str(ckpt_dir / 'final'))
        print('[train] saved final', flush=True)


if __name__ == '__main__':
    main()
