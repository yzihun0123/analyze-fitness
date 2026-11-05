#!/usr/bin/env python3
"""Roboflow 덤벨 컬 라벨이 '자세'가 아니라 '동작 단계(팔꿈치 굽힘)'와 얽혀 있는지 확인 (rf_pose.py → rf_adapt.py 다음에 실행).

출력: results/tables/rf_curl_phase_check.csv
- 기하 특징 1개(팔꿈치 굽힘·위팔 이탈·몸통 기울기)의 AUROC
- 팔꿈치 각도 구간별 정자세 비율
- 팔꿈치 각도가 비슷한 프레임끼리만 비교했을 때 실사 모델(B: 관절 특징 89 → LogReg, 영상 단위 교차검증) AUROC
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_pose as vp
from photo_models import frame_features
from build_cache import JOINTS

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    d = np.load("rf_cache/rf_pose.npz"); ok = np.isfinite(d["W"]).all((1, 2)); y, G = d["y"][ok], d["video"][ok]
    pm = json.load(open("app_data/photo_models.json", encoding="utf-8"))["models"]["덤벨 컬"]
    yaw = json.load(open("app_data/models.json", encoding="utf-8"))["models"]["덤벨 컬"]["yaw_mean"]
    A = np.stack([vp.align_yaw(vp.to_aihub(w[None]), yaw)[0] for w in d["W"][ok].astype(np.float64)])
    Z = np.clip((frame_features(A) - np.array(pm["mean"])) / np.array(pm["scale"]), -3, 3); J = {j: i for i, j in enumerate(JOINTS)}
    def ang(a, b, c):
        u, v = A[:, J[a]] - A[:, J[b]], A[:, J[c]] - A[:, J[b]]
        return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1)))
    el = np.minimum(ang("Left Shoulder", "Left Elbow", "Left Wrist"), ang("Right Shoulder", "Right Elbow", "Right Wrist"))
    feats = {"팔꿈치 굽힘 각(동작 단계)": el, "어깨 각(위팔 이탈)": np.maximum(ang("Left Hip", "Left Shoulder", "Left Elbow"), ang("Right Hip", "Right Shoulder", "Right Elbow")),
             "몸통 기울기": ang("Neck", "Waist", "Left Hip")}
    p = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(Z, y, G):
        p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    rows = [dict(check=f"기하 특징 1개: {n}", frames=len(y), correct_rate=y.mean(), auroc=max(roc_auc_score(y, v), 1 - roc_auc_score(y, v))) for n, v in feats.items()]
    rows.append(dict(check="실사 모델 B (전체)", frames=len(y), correct_rate=y.mean(), auroc=roc_auc_score(y, p)))
    for lo, hi in [(0, 60), (60, 90), (90, 120), (120, 150), (150, 181)]:
        k = (el >= lo) & (el < hi)
        rows.append(dict(check=f"팔꿈치 {lo}~{hi}° 구간", frames=int(k.sum()), correct_rate=y[k].mean(),
                         auroc=roc_auc_score(y[k], p[k]) if len(set(y[k])) > 1 else np.nan))
    R = pd.DataFrame(rows); R.to_csv("results/tables/rf_curl_phase_check.csv", index=False, encoding="utf-8-sig"); print(R.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
