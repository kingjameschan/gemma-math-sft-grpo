"""Remote SFT on A10: distill OpenMathInstruct-2 → Gemma2-2B-IT.
   Mirrors v3/E2_sft/train/01_sft.py config (r=64 a=32 lr=5e-4, 2ep) — only data source differs.
   Runs on the aliyun instance. Model pulled from ModelScope (gemma-2-2b-it)."""
import argparse, json, time
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path('/root/distill')
TRAIN_FILE = ROOT / 'distill_train.jsonl'
CKPT_DIR = ROOT / 'checkpoints'


def load_pc(path):
    rows = [json.loads(l) for l in open(path)]
    # keep only prompt+completion for TRL
    return Dataset.from_list([{'prompt': r['prompt'], 'completion': r['completion']} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/gemma-2-2b-it')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--ep', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--max_length', type=int, default=1024)
    ap.add_argument('--save_steps', type=int, default=500)
    ap.add_argument('--logging_steps', type=int, default=20)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('[load] dataset', flush=True)
    ds = load_pc(TRAIN_FILE)
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
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.ep,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=args.warmup_ratio,
        optim='adamw_torch_fused',
        max_length=args.max_length,
        packing=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=None,
        logging_steps=args.logging_steps, seed=args.seed,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
    eff = args.batch_size * args.accum
    steps_per_ep = len(ds) // eff
    print(f'[train] eff_batch={eff} steps/ep~{steps_per_ep} total~{steps_per_ep*args.ep}', flush=True)
    t0 = time.time()
    trainer.train()
    print(f'[train] done {time.time()-t0:.0f}s', flush=True)
    trainer.save_model(str(CKPT_DIR / 'final'))
    print('[train] saved final', flush=True)


if __name__ == '__main__':
    main()
