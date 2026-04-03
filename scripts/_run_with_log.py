"""
运行指定脚本并实时写日志。用法：
  python scripts/_run_with_log.py logs/xxx.log scripts/train_sft_v2.py [args...]
"""
import sys, subprocess, os

log_path = sys.argv[1]
cmd = [sys.executable] + sys.argv[2:]

os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

with open(log_path, "w", buffering=1, encoding="utf-8", errors="replace") as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
    proc.wait()
    print(f"\n[Exit code: {proc.returncode}]", flush=True)
    log.write(f"\n[Exit code: {proc.returncode}]\n")

sys.exit(proc.returncode)
