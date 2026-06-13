"""Compare v4 vs v5 judge runs on the same 200 stratified rows.

Inputs:
  /tmp/judge_validation_200.jsonl       (v4 baseline)
  /tmp/judge_validation_200_v5.jsonl    (v5 from Gemini 3.1 Flash-Lite Preview)

Outputs to stdout:
  1. processed/failed counts (v4 vs v5)
  2. cap-rule violations (is_correct=False but score>=5; is_correct=True but score<=4)
  3. confusion matrix vs extractor_correct
  4. score histograms side-by-side
  5. avg latency (v5 only — v4 had no latency field)
  6. 30 cases with |v4_score - v5_score| >= 3
  7. recommendation block
"""
import json
import statistics
from collections import Counter
from pathlib import Path

V4 = Path("/tmp/judge_validation_200.jsonl")
V5 = Path("/tmp/judge_validation_200_v5.jsonl")


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    v4 = load(V4)
    v5 = load(V5)
    v4_by = {r["idx"]: r for r in v4}
    v5_by = {r["idx"]: r for r in v5}
    common = [i for i in v4_by if i in v5_by]
    common.sort()

    print("=" * 78)
    print(" SECTION 1 — Processed / Failed")
    print("=" * 78)
    v4_failed = [r for r in v4 if r.get("judge_error") or r.get("judge_score") is None]
    v5_failed = [r for r in v5 if r.get("err_tag", "ok") != "ok" or r.get("judge_score") is None]
    print(f"v4 total={len(v4)}  failed/None={len(v4_failed)}  ok={len(v4)-len(v4_failed)}")
    print(f"v5 total={len(v5)}  failed/None={len(v5_failed)}  ok={len(v5)-len(v5_failed)}")
    if v5_failed:
        print("v5 failure tags:")
        tags = Counter(r.get("err_tag") for r in v5_failed)
        for t, c in tags.most_common():
            print(f"   {t}: {c}")
        for r in v5_failed[:5]:
            print(f"   idx={r['idx']} tag={r.get('err_tag')} msg={r.get('err_msg','')[:120]}")
    print()

    print("=" * 78)
    print(" SECTION 2 — Cap-rule Enforcement")
    print("=" * 78)
    def cap_violations(rows, score_key="judge_score", icc_key="judge_is_correct"):
        v_lo = []  # is_correct=True but score<=4
        v_hi = []  # is_correct=False but score>=5
        n_scoreable = 0
        for r in rows:
            s = r.get(score_key)
            ic = r.get(icc_key)
            if s is None or ic is None:
                continue
            n_scoreable += 1
            if ic is True and s <= 4:
                v_lo.append(r)
            if ic is False and s >= 5:
                v_hi.append(r)
        return v_lo, v_hi, n_scoreable

    v4_lo, v4_hi, v4_n = cap_violations(v4)
    v5_lo, v5_hi, v5_n = cap_violations(v5)

    print(f"v4 (n_scoreable={v4_n}):")
    print(f"   is_correct=False but score>=5: {len(v4_hi)}  (target=0)")
    print(f"   is_correct=True  but score<=4: {len(v4_lo)}  (target=0)")
    print(f"v5 (n_scoreable={v5_n}):")
    print(f"   is_correct=False but score>=5: {len(v5_hi)}  (target=0)")
    print(f"   is_correct=True  but score<=4: {len(v5_lo)}  (target=0)")
    if v5_hi:
        print("\n   Examples of v5 hi-cap violations (False but score>=5):")
        for r in v5_hi[:5]:
            print(f"     idx={r['idx']} score={r['judge_score']} reason={r.get('judge_reason','')[:120]}")
    if v5_lo:
        print("\n   Examples of v5 lo-cap violations (True but score<=4):")
        for r in v5_lo[:5]:
            print(f"     idx={r['idx']} score={r['judge_score']} reason={r.get('judge_reason','')[:120]}")
    print()

    print("=" * 78)
    print(" SECTION 3 — Confusion matrix: judge_is_correct vs extractor_correct")
    print("=" * 78)
    def confusion(rows):
        tt = tf = ft = ff = none = 0
        false_pos = []   # judge True, extractor False
        for r in rows:
            jic = r.get("judge_is_correct")
            ec = r.get("extractor_correct")
            if jic is None:
                none += 1; continue
            if jic and ec: tt += 1
            elif jic and not ec:
                tf += 1
                false_pos.append(r)
            elif not jic and ec: ft += 1
            else: ff += 1
        return tt, tf, ft, ff, none, false_pos

    v4_tt, v4_tf, v4_ft, v4_ff, v4_none, v4_fp = confusion(v4)
    v5_tt, v5_tf, v5_ft, v5_ff, v5_none, v5_fp = confusion(v5)
    print(f"            judge=T,extr=T  judge=T,extr=F  judge=F,extr=T  judge=F,extr=F  none")
    print(f"     v4:    {v4_tt:>14}  {v4_tf:>14}  {v4_ft:>14}  {v4_ff:>14}  {v4_none:>4}")
    print(f"     v5:    {v5_tt:>14}  {v5_tf:>14}  {v5_ft:>14}  {v5_ff:>14}  {v5_none:>4}")
    print()
    print(f"   FALSE POSITIVES — judge says CORRECT but extractor says WRONG (rare/critical):")
    print(f"     v4 count={len(v4_fp)},  v5 count={len(v5_fp)}")
    if v5_fp:
        print("     v5 details:")
        for r in v5_fp[:8]:
            tail = r.get("response", "")[-150:].replace("\n", " ")
            print(f"       idx={r['idx']} gold={r['gold']} extr_pred={r.get('extractor_pred')} v5_score={r.get('judge_score')}")
            print(f"         reason: {r.get('judge_reason','')[:120]}")
    print()

    print("=" * 78)
    print(" SECTION 4 — Score histogram")
    print("=" * 78)
    def hist(rows, key="judge_score"):
        c = Counter()
        for r in rows:
            s = r.get(key)
            if s is not None:
                c[s] += 1
        return c
    h4 = hist(v4)
    h5 = hist(v5)
    print(f"   score | v4_count | v5_count | bar(v4) / bar(v5)")
    for s in range(11):
        c4 = h4.get(s, 0)
        c5 = h5.get(s, 0)
        b4 = "#" * c4
        b5 = "#" * c5
        print(f"     {s:>2}  | {c4:>8} | {c5:>8} | {b4:<60}  /  {b5}")
    # bimodality check: % in mid-band 4-7
    mid_v4 = sum(h4.get(s, 0) for s in range(4, 8))
    mid_v5 = sum(h5.get(s, 0) for s in range(4, 8))
    n4 = sum(h4.values()); n5 = sum(h5.values())
    print(f"   mid-band (4-7): v4={mid_v4}/{n4} ({mid_v4/max(n4,1)*100:.1f}%)  "
          f"v5={mid_v5}/{n5} ({mid_v5/max(n5,1)*100:.1f}%)")
    print(f"   extreme (0-1 or 9-10): v4={(h4.get(0,0)+h4.get(1,0)+h4.get(9,0)+h4.get(10,0))}/{n4}  "
          f"v5={(h5.get(0,0)+h5.get(1,0)+h5.get(9,0)+h5.get(10,0))}/{n5}")
    print()

    print("=" * 78)
    print(" SECTION 5 — Latency (v5 only — v4 has no latency field)")
    print("=" * 78)
    lats = [r["latency_s"] for r in v5 if r.get("latency_s") is not None]
    if lats:
        print(f"   v5 n={len(lats)}  avg={statistics.mean(lats):.2f}s  "
              f"median={statistics.median(lats):.2f}s  p95={statistics.quantiles(lats, n=20)[-1]:.2f}s  "
              f"max={max(lats):.2f}s")
    print()

    print("=" * 78)
    print(" SECTION 6 — 30 large-divergence cases (|v4-v5|>=3)")
    print("=" * 78)
    diffs = []
    for i in common:
        s4 = v4_by[i].get("judge_score")
        s5 = v5_by[i].get("judge_score")
        if s4 is None or s5 is None:
            continue
        d = abs(s4 - s5)
        if d >= 3:
            diffs.append((d, i))
    diffs.sort(reverse=True)
    print(f"  Total cases with |Δ|>=3: {len(diffs)}")
    for d, i in diffs[:30]:
        v4r = v4_by[i]
        v5r = v5_by[i]
        tail = (v5r.get("response") or "")[-200:].replace("\n", " ")
        print(f"\n  --- idx={i}  Δ={d}  gold={v5r['gold']}  extr_correct={v5r.get('extractor_correct')} ---")
        print(f"  RESPONSE TAIL: ...{tail}")
        print(f"  v4: score={v4r.get('judge_score'):>3} ic={v4r.get('judge_is_correct')} "
              f"reason={(v4r.get('judge_reason') or '')[:140]}")
        print(f"  v5: score={v5r.get('judge_score'):>3} ic={v5r.get('judge_is_correct')} "
              f"reason={(v5r.get('judge_reason') or '')[:140]}")
    print()

    print("=" * 78)
    print(" SECTION 7 — Summary stats for recommendation")
    print("=" * 78)
    # alignment with extractor_correct (rough proxy for judge accuracy)
    def align(rows):
        n = m = 0
        for r in rows:
            jic = r.get("judge_is_correct")
            ec = r.get("extractor_correct")
            if jic is None or ec is None: continue
            n += 1
            if jic == ec: m += 1
        return m, n
    m4, n4r = align(v4)
    m5, n5r = align(v5)
    print(f"   judge_is_correct ↔ extractor_correct agreement:")
    print(f"     v4: {m4}/{n4r} = {m4/max(n4r,1)*100:.1f}%")
    print(f"     v5: {m5}/{n5r} = {m5/max(n5r,1)*100:.1f}%")

    # cap violations as % of scoreable
    print(f"   cap-rule violation rate (target=0%):")
    print(f"     v4: {(len(v4_lo)+len(v4_hi))}/{v4_n} = {(len(v4_lo)+len(v4_hi))/max(v4_n,1)*100:.1f}%")
    print(f"     v5: {(len(v5_lo)+len(v5_hi))}/{v5_n} = {(len(v5_lo)+len(v5_hi))/max(v5_n,1)*100:.1f}%")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
