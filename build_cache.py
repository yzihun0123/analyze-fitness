#!/usr/bin/env python3
"""풀린 JSON -> 데이터셋별 캐시(cache/<이름>.npz + .json). 이후 실험은 캐시만 읽는다.

사용:  python build_cache.py DATA_DIR [--out cache]
클립마다 3D (16,24,3), 2D 5시점 (16,5,24,2), 시점 active (16,5), 라벨, Z코드, 촬영일, 필터 플래그를 저장한다.
필터 규칙은 analyze_fitness.py 와 같다: 2D 16프레임·라벨·Z코드, 3D 짝·16프레임, 3D 이상값(|좌표|>1e4 또는 정규화 특징 |f|>50).
"""
import os, sys, json, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_fitness import PAT, ZPAT, NFRAMES, RAW_LIMIT, NORM_LIMIT

JOINTS = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
          "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
          "Right Knee", "Left Ankle", "Right Ankle", "Neck", "Left Palm", "Right Palm", "Back", "Waist",
          "Left Foot", "Right Foot"]
VIEWS = ["view1", "view2", "view3", "view4", "view5"]

def summary_feat(a):
    """(16,24,3) -> analyze_fitness.feat_3d 와 같은 216 특징 (이상값 판정용)."""
    n = (a - a[:, None, JOINTS.index("Neck")]) / (np.linalg.norm(a[:, JOINTS.index("Left Shoulder")] - a[:, JOINTS.index("Right Shoulder")], axis=1).mean() + 1e-6)
    return np.concatenate([n.mean(0).ravel(), n.std(0).ravel(), (n.max(0) - n.min(0)).ravel()])

def load(p):
    r = dict(path=p, ok2d=False, has3d=False, ok3d=False, outlier3d=False)
    X2 = np.full((NFRAMES, 5, 24, 2), np.nan, np.float32); act = np.zeros((NFRAMES, 5), bool)
    X3 = np.full((NFRAMES, 24, 3), np.nan, np.float32)
    try:
        d = json.load(open(p, encoding="utf-8"))
        ti = d.get("type_info") or {}; fr = d.get("frames") or []
        key = fr[0]["view1"]["img_key"] if fr and "view1" in fr[0] else ""
        m = ZPAT.search(key)
        r.update(category=ti.get("type"), exercise=(ti.get("exercise") or "").strip(),
                 names=[c["condition"].strip() for c in ti.get("conditions", [])],
                 vals=[int(bool(c["value"])) for c in ti.get("conditions", [])],
                 z=m.group(1) if m else None, nframes2d=len(fr))
        r["day"], r["slot"] = PAT.search(os.path.basename(p)).group(1, 2)
        if len(fr) == NFRAMES and r["vals"] and r["z"]:
            for t, f in enumerate(fr):
                for v, vn in enumerate(VIEWS):
                    X2[t, v] = [[f[vn]["pts"][j]["x"], f[vn]["pts"][j]["y"]] for j in JOINTS]
                    act[t, v] = str(f[vn].get("active", "")).lower() in ("yes", "true", "1")
            r["ok2d"] = True
    except Exception as e:
        r["err2d"] = repr(e)[:200]
    p3 = p[:-5] + "-3d.json"
    if os.path.exists(p3):
        r["has3d"] = True
        try:
            fr = json.load(open(p3, encoding="utf-8"))["frames"]
            if len(fr) == NFRAMES:
                X3[:] = [[[f["pts"][j]["x"], f["pts"][j]["y"], f["pts"][j]["z"]] for j in JOINTS] for f in fr]
                r["ok3d"] = True
                a = X3.astype(float)
                r["outlier3d"] = bool(not np.isfinite(a).all() or np.abs(a).max() > RAW_LIMIT or np.abs(summary_feat(a)).max() > NORM_LIMIT)
        except Exception as e:
            r["err3d"] = repr(e)[:200]
    return r, X2, act, X3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--out", default="cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ds in sorted(os.listdir(a.data)):
        root = os.path.join(a.data, ds)
        if ds == "docs" or not os.path.isdir(root): continue
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if PAT.search(f))
        with Pool() as pl:
            res = pl.map(load, files, chunksize=50)
        meta = [r for r, *_ in res]
        for m in meta: m["path"] = os.path.relpath(m["path"], a.data)
        np.savez_compressed(os.path.join(a.out, ds + ".npz"), X2=np.stack([x for _, x, _, _ in res]),
                            act=np.stack([x for _, _, x, _ in res]), X3=np.stack([x for *_, x in res]))
        json.dump(meta, open(os.path.join(a.out, ds + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{ds}: clips={len(meta)} ok2d={sum(m['ok2d'] for m in meta)} has3d={sum(m['has3d'] for m in meta)} "
              f"ok3d={sum(m['ok3d'] for m in meta)} outlier3d={sum(m['outlier3d'] for m in meta)}", flush=True)

if __name__ == "__main__":
    main()
