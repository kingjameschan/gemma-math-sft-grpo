"""Stream-sample ~500K distill subset from clean_all.parquet (memory-safe).
   Global keep-prob preserves source proportions (stratified by construction).
   Convert to TRL prompt+completion format. Hold out dev set.

Output: v3/E6_distill/data/
  - distill_train.jsonl  (~500K, TRL conversational)
  - distill_dev.jsonl    (500 held out)
"""
import json, random, collections
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path('/mnt/d/fine-tuning')
SRC = ROOT / 'data/openmathinstruct2/clean_all.parquet'
OUT = ROOT / 'v3/E6_distill/data'
OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
N_TOTAL = 10_277_201
N_TARGET = 500_000
N_DEV = 500
SEED = 42
KEEP_P = N_TARGET / N_TOTAL

rng = random.Random(SEED)
pf = pq.ParquetFile(SRC)
cols = ['problem', 'generated_solution', 'expected_answer', 'problem_source']
print(f'[stream] keep_p={KEEP_P:.5f} over {N_TOTAL} rows', flush=True)

kept = []
seen = 0
src_ct = collections.Counter()
for bi, batch in enumerate(pf.iter_batches(batch_size=20000, columns=cols)):
    d = batch.to_pydict()
    n = len(d['problem'])
    for i in range(n):
        seen += 1
        if rng.random() < KEEP_P:
            rec = {
                'prompt': [{'role': 'user', 'content': d['problem'][i] + SUFFIX}],
                'completion': [{'role': 'assistant', 'content': d['generated_solution'][i]}],
                'source': d['problem_source'][i],
                'gold': str(d['expected_answer'][i]),
            }
            kept.append(rec)
            src_ct[d['problem_source'][i]] += 1
    if bi % 20 == 0:
        print(f'[stream] batch {bi} seen={seen} kept={len(kept)}', flush=True)

print(f'[stream] DONE seen={seen} kept={len(kept)}', flush=True)
rng.shuffle(kept)

dev = kept[:N_DEV]
train = kept[N_DEV:]
with open(OUT / 'distill_train.jsonl', 'w') as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(OUT / 'distill_dev.jsonl', 'w') as f:
    for r in dev:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'[done] train={len(train)} dev={len(dev)} → {OUT}', flush=True)
print('[source dist]', dict(src_ct), flush=True)
