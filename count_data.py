#!/usr/bin/env python3
"""풀린 데이터셋별 2D/3D JSON 개수, 짝 일치, 사용 가능 클립, 수행자(Z) 수 집계 (핸드오프 §2.7 검증용).

사용:  python count_data.py DATA_DIR [--out logs/counts.json]
DATA_DIR 아래 각 하위 폴더(docs 제외)를 데이터셋 하나로 본다.
사용 가능 클립 = 2D 16프레임 + 라벨 + Z코드 + 3D 짝 존재 + 3D 16프레임 + 3D 이상값 아님 (analyze_fitness.py 와 같은 규칙).
"""
import os, sys, json, argparse, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_fitness import PAT, scan_2d, feat_3d, NFRAMES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--out", default="logs/counts.json")
    a = ap.parse_args()
    summary = {}
    for ds in sorted(os.listdir(a.data)):
        root = os.path.join(a.data, ds)
        if ds == "docs" or not os.path.isdir(root): continue
        f2 = [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if PAT.search(f)]
        f3 = [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if f.endswith("-3d.json")]
        s3 = {p[:-8] for p in f3}; s2 = {p[:-5] for p in f2}
        with Pool() as pl:
            rows = [r for r in pl.map(scan_2d, f2, chunksize=50) if r]
        reasons = collections.Counter()
        ok = []
        for r in rows:
            if r["nframes"] != NFRAMES: reasons["2D 프레임 16 아님"] += 1
            elif not r["vals"]: reasons["라벨 없음"] += 1
            elif not r["z"]: reasons["Z코드 없음"] += 1
            else: ok.append(r)
        reasons["2D 읽기 실패"] = len(f2) - len(rows)
        with Pool() as pl:
            F = pl.map(feat_3d, [r["path"] for r in ok], chunksize=100)
        usable = []
        for r, f in zip(ok, F):
            if f is None:
                reasons["3D 없음" if r["path"][:-5] not in s3 else "3D 프레임 16 아님/읽기 실패"] += 1
            elif isinstance(f, str): reasons["3D 이상값"] += 1
            else: usable.append(r)
        summary[ds] = dict(json_2d=len(f2), json_3d=len(f3), only_2d=len(s2 - s3), only_3d=len(s3 - s2),
                           usable=len(usable), performers=len({r["z"] for r in usable}),
                           performers_2d_ok=len({r["z"] for r in ok}), excluded=dict(reasons),
                           exercises=len({r["exercise"] for r in usable}))
        print(ds, json.dumps(summary[ds], ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(summary, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
