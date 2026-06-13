import glob, pandas as pd
fp = glob.glob('/mnt/d/fine-tuning/v3/E6_distill/hf_cache/**/train-00000-of-00032.parquet', recursive=True)[0]
df = pd.read_parquet(fp)
vc = df['problem_source'].value_counts(dropna=False)
print("SOURCE_COUNTS"); print(vc.to_string())

def is_num(s):
    s = str(s).replace(',', '').replace('$', '').strip()
    try:
        float(s); return True
    except Exception:
        return False

print("\nNUMERIC_FRAC (source / frac / n)")
for src in vc.index:
    sub = df[df['problem_source'] == src]
    print(f"{src}\t{sub['expected_answer'].apply(is_num).mean():.3f}\t{len(sub)}")

print("\nLEN_PCT (chars)")
print(df['generated_solution'].str.len().quantile([.5, .9, .95, .99]).to_string())

# MATH example
r = df[df['problem_source'] == 'augmented_math'].iloc[0]
print("\nMATH_EXAMPLE")
print("Q:", r['problem'][:180])
print("A:", repr(r['expected_answer']))
print("SOL:", r['generated_solution'][:400])
