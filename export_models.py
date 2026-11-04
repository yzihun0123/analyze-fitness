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
from video_pose import hip_yaw, observable

# 라이브 데모 예시 클립용 수행자: 해당 운동의 데모 모델 학습에서 제외한다 (예시를 '처음 보는 사람'으로 만들기 위해).
# 조건 조합 32개를 모두 찍은 수행자를 골랐다. 교차검증 수치(cv_*)는 기존 실험 결과 그대로다.
HOLDOUT = {"덤벨 컬": "Z5", "바벨 스쿼트": "Z53", "스텝 포워드 다이나믹 런지": "Z17", "랫풀 다운": "Z4"}

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
    Fv = features(observable(X3.astype(np.float64)))   # 영상용: MediaPipe 로 관측 가능한 관절만으로 재구성한 좌표
    cv = pd.read_csv("results/all_logreg_3d_ang_paired_metrics.csv").set_index(["exercise", "condition"])
    cvv = pd.read_csv("results/all_logreg_3d_ang_vid_paired_metrics.csv").set_index(["exercise", "condition"])

    def fit(Fm, ids, names, Y, cvt, e):
        """표준화 + 조건별 LogReg(C=0.1) → numpy 로 재현 가능한 계수 dict (교차검증 수치는 exp_seq 결과에서).
        분산이 사실상 0인 특징(영상용 재구성에서 Back 이 목-허리 직선 위에 놓여 생기는 상수 각도 등)은 쓰지 않는다:
        mean=0, scale=1, 계수=0 으로 두어 예측에 기여하지 않게 한다 (척도를 반올림하면 0 으로 나누게 되므로)."""
        const = Fm[ids].std(0) < 1e-6
        Fx = Fm[ids].copy(); Fx[:, const] = 0
        sc = StandardScaler().fit(Fx); Z = sc.transform(Fx); conds = []
        mean, scale = sc.mean_.copy(), sc.scale_.copy(); mean[const], scale[const] = 0.0, 1.0
        for j, cn in enumerate(names):
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(Z, Y[:, j])
            w, b = clf.coef_[0].copy(), float(clf.intercept_[0]); w[const] = 0.0
            p_np = 1 / (1 + np.exp(-(((Fm[ids] - mean) / scale) @ w + b)))   # numpy 재현 검증 (원래 특징 그대로 넣어도 같아야 함)
            assert np.allclose(p_np, clf.predict_proba(Z)[:, 1], atol=1e-6), (e, cn)
            r = cvt.loc[(e, cn)]
            conds.append(dict(name=cn, coef=w.tolist(), intercept=b, pos_rate=float(Y[:, j].mean()),
                              cv_bal_acc=float(r["bal_acc"]), cv_auroc=float(r["auroc"]), cv_majority=float(r["majority"])))
        return dict(mean=mean.tolist(), scale=scale.tolist(), conditions=conds)

    ex = np.array([m["exercise"] for m in metas]); models, skel = {}, {}
    for e in sorted(set(ex)):
        all_ids = np.where(ex == e)[0]; names = metas[all_ids[0]]["names"]
        ids = np.array([i for i in all_ids if metas[i]["z"] != HOLDOUT.get(e)])   # 학습용 (예시 수행자 제외)
        Y = np.array([metas[i]["vals"] for i in ids]); Yall = np.array([metas[i]["vals"] for i in all_ids])
        main_m, video_m = fit(F, ids, names, Y, cv, e), fit(Fv, ids, names, Y, cvv, e)
        for j, cn in enumerate(names):
            for lab in (0, 1):   # 평균 골격은 집계값이므로 전체 클립 사용
                k = Yall[:, j] == lab
                if k.any(): skel[f"{e}|{cn}|{lab}"] = S[all_ids][k].mean(0).astype(np.float16)
        skel[f"{e}|__all__|1"] = S[all_ids].mean(0).astype(np.float16)
        # 영상 판별용: 이 운동 클립들의 골반(L→R) 방위각 — 원형 평균·표준편차(도). 영상 골격을 이 방향으로 돌려 맞춘다
        yaw = hip_yaw(X3[all_ids].astype(np.float64)); yc, ys = np.cos(yaw).mean(), np.sin(yaw).mean()
        yaw_mean, yaw_sd = np.degrees(np.arctan2(ys, yc)), np.degrees(np.sqrt(-2 * np.log(np.hypot(yc, ys))))
        models[e] = dict(category=metas[ids[0]]["category"], clips=len(ids), performers=len({metas[i]["z"] for i in ids}),
                         holdout_performer=HOLDOUT.get(e), yaw_mean=round(float(yaw_mean), 2), yaw_sd=round(float(yaw_sd), 2),
                         video=video_m, **main_m)
        print(f"{e}: clips={len(ids)} conditions={len(names)}", flush=True)
    json.dump(dict(feature="summary(norm3d 72 + angles 17) = 267; video = 같은 특징을 video_pose.observable 좌표로", models=models), open(os.path.join(a.out, "models.json"), "w", encoding="utf-8"), ensure_ascii=False)
    np.savez_compressed(os.path.join(a.out, "mean_skeletons.npz"), **skel)
    shutil.copy("logs/counts.json", os.path.join(a.out, "counts.json"))
    print("saved", {f: os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)})

if __name__ == "__main__":
    main()
