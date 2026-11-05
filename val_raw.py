#!/usr/bin/env python3
"""AI Hub 원시 이미지(babel_15.tar, Day16 = 라벨 바벨·덤벨 zip 15)로 영상 판별 경로의 실제 정확도 측정.

사용:  python val_raw.py
- 라벨 클립(2D JSON 의 img_key)과 짝 맞는 프레임만 tar 에서 스트리밍 추출 → MediaPipe(이미지 모드) → to_aihub → align_yaw
- 모델: 운동마다 '이 묶음 수행자를 뺀' 영상용 LogReg(observable 좌표 특징, export_models 와 같은 방식)를 새로 학습 → 공정한 평가
- 비교: (MP) MediaPipe 좌표, 시점 A~E 각각  vs  (3D) 같은 클립의 AI Hub 3D(observable) 좌표 = 상한
출력: rf_cache/raw15_pose.npz (캐시), results/tables/raw15_validation.csv
"""
import os, re, sys, json, tarfile, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_pose as vp
from exp_seq import load_cache
from export_models import features
from build_cache import VIEWS

TAR, CACHE = "dataset/babel_15.tar", "rf_cache/raw15_pose.npz"

class Chain:
    """여러 조각(part)을 이어 붙인 읽기 전용 스트림"""
    def __init__(self, fs): self.fs = fs; self.i = 0
    def read(self, n=-1):
        out = b""
        while self.i < len(self.fs) and (n < 0 or len(out) < n):
            b = self.fs[self.i].read(-1 if n < 0 else n - len(out))
            if not b: self.i += 1
            else: out += b
        return out

def extract(need):
    """need: {img_key: (clip i, view v, frame t)} → W (클립,5,16,33,3)"""
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    outer = tarfile.open(TAR); parts = sorted([m for m in outer if ".part" in m.name], key=lambda m: int(m.name.rsplit("part", 1)[1]))
    inner = tarfile.open(fileobj=Chain([outer.extractfile(m) for m in parts]), mode="r|")
    n = 1 + max(c for c, _, _ in need.values()); W = np.full((n, 5, 16, 33, 3), np.nan, np.float32); done = 0
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=vp.MODEL), running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for m in inner:
            k = m.name[m.name.find("Day"):] if "Day" in m.name else m.name
            if not m.isfile() or k not in need: continue
            img = cv2.imdecode(np.frombuffer(inner.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            h, w = img.shape[:2]; s = 960 / max(h, w)
            if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)))
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            c, v, t = need[k]
            if r.pose_world_landmarks: W[c, v, t] = [[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]]
            done += 1
            if done % 2000 == 0: print("frames", done, "/", len(need), flush=True)
    return W

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    metas, _, _, X3 = load_cache("cache", "paired")
    ids = [i for i, r in enumerate(metas) if re.split(r"[\\/]", r["path"])[:2] == ["바벨,덤벨", "15"]]
    zs = sorted({metas[i]["z"] for i in ids}); print("클립", len(ids), "수행자", zs, flush=True)
    if not os.path.exists(CACHE):
        need = {}
        for c, i in enumerate(ids):
            fr = json.load(open(os.path.join("data", metas[i]["path"]), encoding="utf-8"))["frames"]
            for t in range(16):
                for v, vn in enumerate(VIEWS): need[fr[t][vn]["img_key"]] = (c, v, t)
        W = extract(need); os.makedirs("rf_cache", exist_ok=True); np.savez_compressed(CACHE, W=W, ids=np.array(ids))
    W = np.load(CACHE)["W"].astype(np.float64)
    models = json.load(open("app_data/models.json", encoding="utf-8"))["models"]
    ex = np.array([m["exercise"] for m in metas]); Z = np.array([m["z"] for m in metas]); rows = []
    for e in sorted({metas[i]["exercise"] for i in ids}):
        tr = np.where((ex == e) & ~np.isin(Z, zs))[0]; te = [c for c, i in enumerate(ids) if metas[i]["exercise"] == e]
        Ftr = features(vp.observable(X3[tr].astype(np.float64))); const = Ftr.std(0) < 1e-6; Ftr[:, const] = 0
        sc = StandardScaler().fit(Ftr); Ytr = np.array([metas[i]["vals"] for i in tr]); names = metas[tr[0]]["names"]
        clfs = [LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Ftr), Ytr[:, j]) for j in range(len(names))]
        def pred(F):
            F = F.copy(); F[:, const] = 0; Zs = np.clip(sc.transform(F), -3, 3) if True else sc.transform(F)
            return np.stack([c.predict_proba(Zs)[:, 1] for c in clfs], 1)
        Yte = np.array([metas[ids[c]]["vals"] for c in te])
        P3 = pred(features(vp.observable(X3[[ids[c] for c in te]].astype(np.float64))))
        res = {"3D(AI Hub 좌표, 상한)": P3}
        for v in range(5):
            ok = [k for k, c in enumerate(te) if np.isfinite(W[c, v]).all()]
            if len(ok) < len(te) * 0.8: continue
            Xmp = np.stack([vp.align_yaw(vp.to_aihub(W[te[k], v]), models[e]["yaw_mean"]) for k in ok])
            P = np.full(P3.shape, np.nan); P[ok] = pred(features(Xmp)); res[f"MP 시점{'ABCDE'[v]}"] = P
        for name, P in res.items():
            ok = np.isfinite(P[:, 0])
            for j, cn in enumerate(names):
                rows.append(dict(exercise=e, input=name, condition=cn, clips=int(ok.sum()), bal_acc=balanced_accuracy_score(Yte[ok, j], P[ok, j] >= .5)))
    R = pd.DataFrame(rows); R.to_csv("results/tables/raw15_validation.csv", index=False, encoding="utf-8-sig")
    print(R.groupby(["exercise", "input"]).bal_acc.mean().unstack().round(3).to_string())
    print("\n전체 평균:", R.groupby("input").bal_acc.mean().round(3).to_dict())

if __name__ == "__main__":
    main()
