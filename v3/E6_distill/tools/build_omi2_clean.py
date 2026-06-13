"""Download full OpenMathInstruct-2 (32 shards) → filter numeric → decontaminate
   against GSM8K test + MATH numeric test → save cleaned full set.

Output: /mnt/d/fine-tuning/data/openmathinstruct2/
  - clean_all.parquet        (numeric + decontaminated, all sources)
  - stats.json
Decontamination: normalize(problem) exact match against test problems.
"""
import json, re, time
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd

ROOT = Path('/mnt/d/fine-tuning')
OUT_DIR = ROOT / 'data' / 'openmathinstruct2'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'v3/E6_distill/hf_cache'
N_SHARDS = 32

# ---- normalize for decontamination + numeric check ----
_punct = re.compile(r'[^a-z0-9]+')
def norm_text(s):
    """lowercase, strip all non-alnum (whitespace/punct/latex braces collapse)."""
    return _punct.sub('', str(s).lower())

def is_numeric(s):
    s = str(s).replace(',', '').replace('$', '').strip()
    try:
        float(s); return True
    except Exception:
        return False

# ---- build test decontamination set ----
print('[decon] building test problem set', flush=True)
test_norm = set()
for line in open(ROOT / 'data/gsm8k/test.jsonl'):
    test_norm.add(norm_text(json.loads(line)['question']))
for line in open(ROOT / 'v3/shared/data/math/test_numeric.jsonl'):
    test_norm.add(norm_text(json.loads(line)['problem']))
print(f'[decon] {len(test_norm)} test problems (GSM8K 1319 + MATH 2927)', flush=True)

# ---- process shard by shard ----
kept = []
stats = {'total': 0, 'non_numeric_dropped': 0, 'contaminated_dropped': 0, 'kept': 0,
         'by_source': {}}
t0 = time.time()
for i in range(N_SHARDS):
    fn = f'data/train-{i:05d}-of-{N_SHARDS:05d}.parquet'
    fp = hf_hub_download('nvidia/OpenMathInstruct-2', fn, repo_type='dataset', cache_dir=str(CACHE))
    df = pd.read_parquet(fp)
    n0 = len(df)
    stats['total'] += n0
    # filter numeric
    num_mask = df['expected_answer'].apply(is_numeric)
    stats['non_numeric_dropped'] += int((~num_mask).sum())
    df = df[num_mask]
    # decontaminate
    pn = df['problem'].apply(norm_text)
    contam = pn.isin(test_norm)
    stats['contaminated_dropped'] += int(contam.sum())
    df = df[~contam]
    kept.append(df)
    for src, c in df['problem_source'].value_counts().items():
        stats['by_source'][src] = stats['by_source'].get(src, 0) + int(c)
    print(f'[shard {i:02d}/{N_SHARDS}] rows={n0} kept={len(df)} '
          f'(cum {sum(len(k) for k in kept)}) {time.time()-t0:.0f}s', flush=True)

alldf = pd.concat(kept, ignore_index=True)
stats['kept'] = len(alldf)
out_pq = OUT_DIR / 'clean_all.parquet'
alldf.to_parquet(out_pq)
json.dump(stats, open(OUT_DIR / 'stats.json', 'w'), indent=2)
print(f'\n[done] wrote {len(alldf)} rows → {out_pq}', flush=True)
print(json.dumps(stats, indent=2), flush=True)
