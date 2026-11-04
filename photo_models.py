#!/usr/bin/env python3
"""사진(한 프레임) 판별 모델: AI Hub 클립의 프레임을 한 장씩 학습 → 조건별 '사진으로 판별 가능도' 측정 + 앱용 모델 저장.

사용:  python photo_models.py
- 좌표: 영상과 같이 MediaPipe 로 관측 가능한 관절만 (video_pose.observable), 프레임마다 Neck 원점·그 프레임 어깨너비로 정규화
- 특징(89): 정규화 좌표 72 + 관절각 15 + 귀-어깨 거리 2  (시간 통계 없음 — 사진은 한 장이므로)
- 라벨: 클립 라벨을 그 클립의 모든 프레임에 부여. 클립당 8프레임(2장 간격) 사용
- 평가: 수행자(Z) 분리 GroupKFold(8), 프레임 단위 balanced accuracy (사진이 동작의 어느 순간인지 모른다는 조건과 같음)
- 앱 모델: 예시 수행자(export_models.HOLDOUT) 제외 학습. 신뢰도 기준용으로 예시 수행자 프레임의 '학습 범위 밖 특징 수'별 일치율도 계산
출력: results/photo_cv_metrics.csv, results/tables/photo_ood_vs_acc.csv, app_data/photo_models.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_seq import load_cache, angles3d
from build_cache import JOINTS
from export_models import HOLDOUT
from video_pose import observable

NECK, LS, RS = JOINTS.index("Neck"), JOINTS.index("Left Shoulder"), JOINTS.index("Right Shoulder")
FRAMES = list(range(0, 16, 2))

def frame_features(X):
    """(N,24,3) 한 프레임 좌표 → (N,89)"""
    X = X.astype(np.float64)
    sw = np.linalg.norm(X[:, LS] - X[:, RS], axis=-1)[:, None, None] + 1e-6
    n = ((X - X[:, NECK:NECK + 1]) / sw).reshape(len(X), -1)
    return np.concatenate([n, angles3d(X[:, None])[:, 0]], 1).astype(np.float32)

def main():
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    metas, _, _, X3 = load_cache("cache", "paired")
    ex = np.array([m["exercise"] for m in metas]); rows, models, ood = [], {}, []
    for e in sorted(set(ex), key=lambda e: -np.sum(ex == e)):
        ids = np.where(ex == e)[0]; names = metas[ids[0]]["names"]
        O = observable(X3[ids][:, FRAMES].astype(np.float64))                       # (n,8,24,3)
        F = frame_features(O.reshape(-1, 24, 3))
        Y = np.repeat(np.array([metas[i]["vals"] for i in ids]), len(FRAMES), 0)
        G = np.repeat(np.array([metas[i]["z"] for i in ids]), len(FRAMES))
        hold = G == HOLDOUT.get(e, "__none__")
        conds = []
        for j, cn in enumerate(names):
            P = np.zeros(len(F))
            for tr, te in GroupKFold(8).split(F, Y[:, j], G):
                P[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(F[tr], Y[tr, j]).predict_proba(F[te])[:, 1]
            pos = Y[:, j].mean()
            rows.append(dict(exercise=e, condition=cn, frames=len(F), performers=len(set(G)), majority=max(pos, 1 - pos),
                             bal_acc=balanced_accuracy_score(Y[:, j], P >= .5), auroc=roc_auc_score(Y[:, j], P)))
            sc = StandardScaler().fit(F[~hold]); clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(F[~hold]), Y[~hold, j])
            conds.append(dict(name=cn, coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]), cv_bal_acc=rows[-1]["bal_acc"]))
            if hold.any():   # 신뢰도 기준용: 학습에서 뺀 예시 수행자 프레임
                z = sc.transform(F[hold]); p = clf.predict_proba(z)[:, 1]
                ood.append(pd.DataFrame(dict(exercise=e, condition=cn, n_far=(np.abs(z) > 3).sum(1), correct=(p >= .5) == Y[hold, j])))
        sc = StandardScaler().fit(F[~hold])
        models[e] = dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), conditions=conds, holdout_performer=HOLDOUT.get(e))
        print(f"{e}: frames={len(F)} | " + ", ".join(f"{r['condition'][:8]} {r['bal_acc']:.2f}" for r in rows[-len(names):]), flush=True)
    R = pd.DataFrame(rows); R.to_csv("results/photo_cv_metrics.csv", index=False, encoding="utf-8-sig")
    D = pd.concat(ood); bins = pd.cut(D.n_far, [-1, 0, 2, 5, 10, 999], labels=["0", "1~2", "3~5", "6~10", "11+"])
    T = D.groupby(bins, observed=True).correct.agg(["size", "mean"]).rename(columns={"size": "frames_x_conditions", "mean": "match_rate"})
    T.to_csv("results/tables/photo_ood_vs_acc.csv", encoding="utf-8-sig"); print(T)
    json.dump(dict(feature="frame: 정규화 좌표 72(observable) + 관절각 15 + 귀-어깨 2 = 89", models=models),
              open("app_data/photo_models.json", "w", encoding="utf-8"), ensure_ascii=False)
    v = pd.read_csv("results/all_logreg_3d_ang_vid_paired_metrics.csv")
    M = R.merge(v[["exercise", "condition", "bal_acc"]], on=["exercise", "condition"], suffixes=("_photo", "_video"))
    print(f"\n조건 {len(M)}개 | balanced acc 평균: 영상(16프레임) {M.bal_acc_video.mean():.3f} → 사진(1프레임) {M.bal_acc_photo.mean():.3f}")
    print(f"사진 bal_acc ≥ 0.75: {(M.bal_acc_photo >= .75).sum()}개, 0.65~0.75: {((M.bal_acc_photo >= .65) & (M.bal_acc_photo < .75)).sum()}개, < 0.65: {(M.bal_acc_photo < .65).sum()}개")

if __name__ == "__main__":
    main()
