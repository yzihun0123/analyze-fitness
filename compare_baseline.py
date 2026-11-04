#!/usr/bin/env python3
"""analyze_fitness.py 재현 결과(results/baseline_repro_<ds>.json) vs baseline_results_all.csv 비교.

사용:  python compare_baseline.py [--tol 0.02]
출력: results/baseline_repro_compare.csv, 콘솔 요약
"""
import os, json, argparse
import pandas as pd

DS = {"기구": "기구", "바벨,덤벨": "바벨_덤벨", "맨몸운동": "맨몸"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tol", type=float, default=0.02); a = ap.parse_args()
    ref = pd.read_csv("baseline_results_all.csv", encoding="utf-8")
    rows = []
    for f, ds in DS.items():
        raw = open(f"results/baseline_repro_{f}.json", "rb").read()   # analyze_fitness 는 인코딩 미지정 저장(Windows: cp949)
        try: recs = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError: recs = json.loads(raw.decode("cp949"))
        for r in recs:
            rows.append(dict(dataset=ds, exercise=r["exercise"], condition=r["condition"].strip(), clips_repro=r["clips"],
                             performers_repro=r["performers"], majority_repro=r["majority"], acc_repro=r["acc_LogReg"]))
    new = pd.DataFrame(rows)
    ref["condition"] = ref["condition"].str.strip()
    m = ref.merge(new, on=["dataset", "exercise", "condition"], how="outer", indicator=True)
    m["diff"] = m["acc_repro"] - m["acc_logreg"]
    m.to_csv("results/baseline_repro_compare.csv", index=False, encoding="utf-8-sig")
    both = m[m["_merge"] == "both"]
    print(f"rows: ref={len(ref)} repro={len(new)} matched={len(both)} only_ref={(m['_merge'] == 'left_only').sum()} only_repro={(m['_merge'] == 'right_only').sum()}")
    print(f"clips equal: {(both['clips'] == both['clips_repro']).mean():.3f} | performers equal: {(both['performers'] == both['performers_repro']).mean():.3f}")
    print(f"|diff| max={both['diff'].abs().max():.4f} mean={both['diff'].abs().mean():.4f} | within ±{a.tol}: {(both['diff'].abs() <= a.tol).sum()}/{len(both)}")
    bad = both[both["diff"].abs() > a.tol]
    if len(bad): print(bad[["dataset", "exercise", "condition", "clips", "clips_repro", "acc_logreg", "acc_repro", "diff"]].to_string())

if __name__ == "__main__":
    main()
