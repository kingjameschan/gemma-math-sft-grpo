# E7 Recovery (paused 2026-06-03, laptop went on battery)

## What happened
Training (pretrain gemma-2-2b SFT on 100K distill) was at **step ~5363/6274 (~85%)**
when the laptop **lost AC power** and dropped to a SW-power-capped GPU state (P4, 60W,
1687MHz) → 22× slowdown (5.9 → 131 s/it). Battery hit 13% / ~8 min, so training was
**cleanly killed** to conserve battery and protect checkpoints. NOT a training bug.

## State (all safe)
- Latest valid checkpoint: `v3/E6_distill/checkpoints_pt/checkpoint-5200/`
  (adapter + optimizer + scheduler + rng + trainer_state — full resume point)
- Cron `8fa2cbf2` DELETED (so it can't auto-resume training on battery).
- Auto-resume is wired: `local_train.py --resume`, `launch_train_pt.sh --resume`,
  and `e7_advance.sh` crash-branch (resume ≤3 if no progress).
- Power plan switched to 节能/power-saver while idle.

## To recover (AFTER restoring AC power)
1. Restore High Performance plan:
   `cmd.exe /c "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"`
   (verify GPU un-throttles: `nvidia-smi -q -d PERFORMANCE | grep -i "power cap"`)
2. Resume training from checkpoint-5200 (identical hyperparams):
   `bash /mnt/d/fine-tuning/v3/E6_distill/tools/launch_train_pt.sh --resume`
3. Re-arm the chain driver cron (every 15 min) pointing at:
   `bash /mnt/d/fine-tuning/v3/E6_distill/tools/e7_advance.sh`
   (the driver then auto-runs: finish train → eval 8 runs K=64 → merge K=128 →
   `outputs/distill_pretrain_combined.png` → push notify)

## Remaining work after resume
~911 steps train (~1.5h on AC) → eval (~2.5-3h) → merge+plot. Then E7 (task #76) done.
