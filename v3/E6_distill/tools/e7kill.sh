#!/bin/bash
# Kill the E7 eval process tree cleanly. Run as `bash .../e7kill.sh` so the
# patterns are in this FILE, not the caller's command line (avoids pkill self-match).
pkill -9 -f run_eval_ptbase_only.sh
pkill -9 -f run_eval_pt_chain.sh
pkill -9 -f 03_eval_pass_at_k
pkill -9 -f vllm
pkill -9 -f EngineCore
sleep 3
exit 0
