#!/usr/bin/env python3
"""AI Hub '피트니스 자세 이미지' 라벨 분석 + 수행자 분리 기준 모델.

사용:  python analyze_fitness.py ROOT_DIR [--out results.json] [--folds 8]
ROOT_DIR 아래를 재귀 탐색해 D{day}-{slot}-{code}.json (2D, 라벨 포함)과
같은 이름의 -3d.json (3D 좌표)을 짝지어 읽는다.
핵심 규칙: 수행자 = img_key 의 -(Z\\d+)_ 코드. 파일명 두 번째 숫자는 '촬영 순서'일 뿐 수행자가 아님.
"""
import os, re, sys, json, argparse, collections, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")

PAT = re.compile(r"D(\d+)-(\d+)-(\d+)\.json$")
ZPAT = re.compile(r"-(Z\d+)_")
NFRAMES = 16
RAW_LIMIT = 1e4     # 정상 3D 좌표는 대략 -200~300 범위
NORM_LIMIT = 50.0   # 어깨너비로 정규화한 특징의 상한

def scan_2d(p):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return None
    ti = d.get("type_info") or {}
    fr = d.get("frames") or []
    key = fr[0]["view1"]["img_key"] if fr and "view1" in fr[0] else ""
    m = ZPAT.search(key)
    conds = ti.get("conditions", [])
    return dict(path=p, category=ti.get("type"), exercise=(ti.get("exercise") or "").strip(),
                names=[c["condition"] for c in conds], vals=[int(bool(c["value"])) for c in conds],
                nframes=len(fr), z=m.group(1) if m else None,
                day=PAT.search(os.path.basename(p)).group(1))

def feat_3d(p):
    """3D 파일 -> (16,24,3) 정규화 후 평균/표준편차/가동범위 특징(216)."""
    try:
        fr = json.load(open(p[:-5] + "-3d.json", encoding="utf-8"))["frames"]
        if len(fr) != NFRAMES: return None
        J = list(fr[0]["pts"].keys())
        a = np.array([[[f["pts"][k]["x"], f["pts"][k]["y"], f["pts"][k]["z"]] for k in J] for f in fr], float)
        neck = a[:, J.index("Neck"), :][:, None, :]
        sc = np.linalg.norm(a[:, J.index("Left Shoulder")] - a[:, J.index("Right Shoulder")], axis=1).mean() + 1e-6
        if not np.isfinite(a).all() or np.abs(a).max() > RAW_LIMIT:   # 3D 복원 오류(10^12 등) 제외
            return "OUTLIER"
        n = (a - neck) / sc
        f = np.concatenate([n.mean(0).ravel(), n.std(0).ravel(), (n.max(0) - n.min(0)).ravel()]).astype(np.float32)
        return "OUTLIER" if np.abs(f).max() > NORM_LIMIT else f
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--out", default="results.json"); ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-rf", action="store_true", help="RF 생략(속도)")
    a = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    files = [os.path.join(r, f) for r, _, fs in os.walk(a.root) for f in fs if PAT.search(f)]
    print("2D files:", len(files), flush=True)
    with Pool(a.workers) as pl:
        rows = [r for r in pl.map(scan_2d, files, chunksize=50) if r]
    bad = [r for r in rows if r["nframes"] != NFRAMES or not r["vals"] or not r["z"]]
    rows = [r for r in rows if r not in bad]
    print(f"usable clips: {len(rows)} (excluded {len(bad)}: not 16 frames / no label / no Z code)", flush=True)
    with Pool(a.workers) as pl:
        F = pl.map(feat_3d, [r["path"] for r in rows], chunksize=100)
    n_out = sum(1 for f in F if isinstance(f, str))
    n_missing = sum(1 for f in F if f is None)
    print(f"3D outliers excluded: {n_out} | missing/short 3D: {n_missing}", flush=True)
    keep = [i for i, f in enumerate(F) if f is not None and not isinstance(f, str)]
    X = np.stack([F[i] for i in keep]); rows = [rows[i] for i in keep]
    print("feature matrix:", X.shape, "| performers (Z):", len({r["z"] for r in rows}), flush=True)

    byex = collections.defaultdict(list)
    for i, r in enumerate(rows): byex[(r["category"], r["exercise"])].append(i)
    out = []
    for (cat, ex), ids in sorted(byex.items(), key=lambda x: -len(x[1])):
        ids = np.array(ids); Xe = X[ids]; Y = np.array([rows[i]["vals"] for i in ids])
        G = np.array([rows[i]["z"] for i in ids]); names = rows[ids[0]]["names"]
        k = min(a.folds, len(set(G))); gkf = GroupKFold(n_splits=k)
        print(f"\n=== [{cat}] {ex}: clips={len(ids)}, performers={len(set(G))}, conditions={len(names)}", flush=True)
        for j, cn in enumerate(names):
            pos = float(Y[:, j].mean()); res = {}
            models = [("LogReg", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))]
            if not a.no_rf: models.append(("RF", lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1)))
            for nm, mk in models:
                acc = []
                for tr, te in gkf.split(Xe, Y[:, j], G):
                    if len(set(Y[tr, j])) < 2: continue
                    acc.append((mk().fit(Xe[tr], Y[tr, j]).predict(Xe[te]) == Y[te, j]).mean())
                res[nm] = float(np.mean(acc)) if acc else None
            maj = max(pos, 1 - pos)
            print(f"  {cn:<26} majority={maj:.2f} | " + " | ".join(f"{k_} {v:.2f}" for k_, v in res.items() if v is not None), flush=True)
            out.append(dict(category=cat, exercise=ex, condition=cn, clips=len(ids), performers=len(set(G)),
                            pos_rate=pos, majority=maj, **{f"acc_{k_}": v for k_, v in res.items()}))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", a.out)

if __name__ == "__main__":
    main()
