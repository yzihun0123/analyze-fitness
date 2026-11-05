#!/usr/bin/env python3
"""MediaPipe → AI Hub 특징 보정 효과 (val_raw.py 캐시 사용). 운동 하나를 빼고 나머지 4개로 보정 학습 → 뺀 운동에서 시험.

보정: none(그대로) / shift(특징별 평균 차이 더함) / linear(특징별 a·x+b 회귀). 보정은 시점마다 따로 학습.
판별 모델: 이 묶음 수행자(Z52·Z53)를 뺀 영상용 LogReg (val_raw.py 와 동일).  출력: results/tables/raw15_calibration.csv
"""
import os, re, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_pose as vp
from exp_seq import load_cache
from export_models import features

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    metas, _, _, X3 = load_cache("cache", "paired")
    c = np.load("rf_cache/raw15_pose.npz"); W, ids = c["W"].astype(np.float64), list(c["ids"])
    zs = sorted({metas[i]["z"] for i in ids}); models = json.load(open("app_data/models.json", encoding="utf-8"))["models"]
    exs = np.array([metas[i]["exercise"] for i in ids]); ex = np.array([m["exercise"] for m in metas]); Z = np.array([m["z"] for m in metas])
    Fa = features(vp.observable(X3[ids].astype(np.float64)))                                          # AI Hub 특징 (240,267)
    Fm = np.full((5,) + Fa.shape, np.nan)
    for v in range(5):
        for k in range(len(ids)):
            if np.isfinite(W[k, v]).all(): Fm[v, k] = features(vp.align_yaw(vp.to_aihub(W[k, v]), models[exs[k]]["yaw_mean"])[None])[0]
    clf = {}
    for e in sorted(set(exs)):
        tr = np.where((ex == e) & ~np.isin(Z, zs))[0]; F = features(vp.observable(X3[tr].astype(np.float64)))
        const = F.std(0) < 1e-6; F[:, const] = 0; sc = StandardScaler().fit(F); Y = np.array([metas[i]["vals"] for i in tr])
        clf[e] = (const, sc, [LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(F), Y[:, j]) for j in range(Y.shape[1])])
    def score(e, F, Y):
        const, sc, cs = clf[e]; F = F.copy(); F[:, const] = 0; Zs = np.clip(sc.transform(F), -3, 3)
        return np.mean([balanced_accuracy_score(Y[:, j], cs[j].predict_proba(Zs)[:, 1] >= .5) for j in range(len(cs))])
    rows = []
    for v in range(5):
        okv = np.isfinite(Fm[v]).all(1)
        for e in sorted(set(exs)):
            te = (exs == e) & okv; trn = (exs != e) & okv; Y = np.array([metas[ids[k]]["vals"] for k in np.where(te)[0]])
            X, A = Fm[v][trn], Fa[trn]; xm, am = X.mean(0), A.mean(0)
            cov = ((X - xm) * (A - am)).mean(0); var = X.var(0) + 1e-9; a = cov / var; b = am - a * xm
            for name, F in [("none", Fm[v][te]), ("shift", Fm[v][te] + (am - xm)), ("linear", Fm[v][te] * a + b)]:
                rows.append(dict(view="ABCDE"[v], exercise=e, calib=name, bal_acc=score(e, F, Y)))
    R = pd.DataFrame(rows); R.to_csv("results/tables/raw15_calibration.csv", index=False, encoding="utf-8-sig")
    print(R.pivot_table(index="view", columns="calib", values="bal_acc").round(3)[["none", "shift", "linear"]].to_string())
    print(R[R.exercise == "덤벨 컬"].pivot_table(index="view", columns="calib", values="bal_acc").round(3)[["none", "shift", "linear"]].to_string())

if __name__ == "__main__":
    main()
