#!/usr/bin/env python3
"""Streamlit 앱용 데이터 내보내기 (원본 AI Hub 데이터 없이 앱이 동작하도록 파생물만 저장).

사용:  python export_models.py [--cache cache] [--out app_data]
출력:
  app_data/models.json          운동별 LogReg(3D 좌표+관절각 요약 267특징, 전체 클립 학습) 계수 + 교차검증 성능
  app_data/mean_skeletons.npz   운동×조건×라벨별 평균 정규화 골격 (16,24,3) — 개별 클립이 아닌 집계값
  app_data/counts.json          데이터 정제 집계 (logs/counts.json 복사)
예측은 numpy 만으로 재현 가능하도록 표준화 평균/척도와 계수를 그대로 저장한다(sklearn 버전 무관).
"""
import os, sys, json, shutil, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_seq import load_cache, norm3d, angles3d, summary

def features(X3):
    """(N,16,24,3) 원좌표 -> (N,267) = 요약(3D 정규화 72 + 관절각 17)"""
    return summary(np.concatenate([norm3d(X3), angles3d(X3.astype(np.float64)).astype(np.float32)], -1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="app_data")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    os.makedirs(a.out, exist_ok=True)
    metas, X2, A, X3 = load_cache(a.cache, "paired")
    F = features(X3); S = norm3d(X3).reshape(len(X3), 16, 24, 3)
    cv = pd.read_csv("results/all_logreg_3d_ang_paired_metrics.csv").set_index(["exercise", "condition"])
    ex = np.array([m["exercise"] for m in metas]); models, skel = {}, {}
    for e in sorted(set(ex)):
        ids = np.where(ex == e)[0]; names = metas[ids[0]]["names"]
        Y = np.array([metas[i]["vals"] for i in ids]); sc = StandardScaler().fit(F[ids]); Z = sc.transform(F[ids])
        conds = []
        for j, cn in enumerate(names):
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(Z, Y[:, j])
            w, b = clf.coef_[0], float(clf.intercept_[0])
            p_np = 1 / (1 + np.exp(-(((F[ids] - sc.mean_) / sc.scale_) @ w + b)))   # numpy 재현 검증
            assert np.allclose(p_np, clf.predict_proba(Z)[:, 1], atol=1e-6), (e, cn)
            r = cv.loc[(e, cn)]
            conds.append(dict(name=cn, coef=np.round(w, 6).tolist(), intercept=round(b, 6), pos_rate=float(Y[:, j].mean()),
                              cv_bal_acc=float(r["bal_acc"]), cv_auroc=float(r["auroc"]), cv_majority=float(r["majority"])))
            for lab in (0, 1):
                k = Y[:, j] == lab
                if k.any(): skel[f"{e}|{cn}|{lab}"] = S[ids][k].mean(0).astype(np.float16)
        skel[f"{e}|__all__|1"] = S[ids].mean(0).astype(np.float16)
        models[e] = dict(category=metas[ids[0]]["category"], clips=len(ids), performers=len({metas[i]["z"] for i in ids}),
                         mean=np.round(sc.mean_, 6).tolist(), scale=np.round(sc.scale_, 6).tolist(), conditions=conds)
        print(f"{e}: clips={len(ids)} conditions={len(names)}", flush=True)
    json.dump(dict(feature="summary(norm3d 72 + angles 17) = 267", models=models), open(os.path.join(a.out, "models.json"), "w", encoding="utf-8"), ensure_ascii=False)
    np.savez_compressed(os.path.join(a.out, "mean_skeletons.npz"), **skel)
    shutil.copy("logs/counts.json", os.path.join(a.out, "counts.json"))
    print("saved", {f: os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)})

if __name__ == "__main__":
    main()
