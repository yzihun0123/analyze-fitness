#!/usr/bin/env python3
"""실제 영상 프레임(Roboflow 덤벨 컬, MediaPipe 관절)으로 (A) AI Hub 사진 모델 외부 검증, (B~D) 도메인 적응 모델 비교.

사용:  python rf_adapt.py          (rf_pose.py 를 먼저 실행)
- 정자세 점수 (A): AI Hub 사진 모델에서 '사진 정확도 ≥ 0.65' 인 조건들의 P(올바름) 최솟값 (하나라도 틀리면 잘못된 자세)
- (B) 실사 데이터만: 한 프레임 특징 89 → LogReg   (C) AI Hub 조건 로짓 5 → LogReg (가중치 보정)   (D) B + C
- 평가: 영상 단위 교차검증 (영상 하나를 통째로 빼고 학습 → 그 영상 예측), class_weight=balanced
출력: results/rf_curl_eval.csv, results/tables/rf_curl_per_video.csv, rf_cache/rf_curl_model.json (D 를 전체 데이터로 학습 — 앱 미사용)
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_pose as vp
from photo_models import frame_features

EX = "덤벨 컬"; JUDGE = 0.65

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    d = np.load("rf_cache/rf_pose.npz"); ok = np.isfinite(d["W"]).all((1, 2))
    W, y, G = d["W"][ok].astype(np.float64), d["y"][ok], d["video"][ok]
    print(f"검출된 프레임 {ok.sum()}/{len(ok)} | 정자세 {y.mean():.3f} | 영상 {len(set(G))}")
    pm = json.load(open("app_data/photo_models.json", encoding="utf-8"))["models"][EX]
    yaw = json.load(open("app_data/models.json", encoding="utf-8"))["models"][EX]["yaw_mean"]
    A = np.stack([vp.align_yaw(vp.to_aihub(w[None]), yaw)[0] for w in W])                   # (N,24,3)
    F = frame_features(A).astype(np.float64)                                                  # (N,89)
    Z = np.clip((F - np.array(pm["mean"])) / np.array(pm["scale"]), -3, 3)
    Wc = np.array([c["coef"] for c in pm["conditions"]]); b = np.array([c["intercept"] for c in pm["conditions"]])
    L = Z @ Wc.T + b                                                                           # 조건별 로짓 (N,5)
    P = 1 / (1 + np.exp(-L)); judg = [i for i, c in enumerate(pm["conditions"]) if c["cv_bal_acc"] >= JUDGE]
    names = [c["name"] for c in pm["conditions"]]
    rows = []
    sA = P[:, judg].min(1)
    rows.append(dict(model="A AI Hub 사진 모델 그대로(외부 검증)", auroc=roc_auc_score(y, sA), bal_acc=balanced_accuracy_score(y, sA >= .5)))
    for j, n in enumerate(names):
        rows.append(dict(model=f"  └ 조건 '{n}' P(올바름) 하나만", auroc=roc_auc_score(y, P[:, j]), bal_acc=balanced_accuracy_score(y, P[:, j] >= .5)))
    feats = {"B 실사 데이터만(관절 특징 89)": Z, "C AI Hub 조건 로짓 5개 재보정": L, "D 둘 다(B+C)": np.concatenate([Z, L], 1)}
    oof = {}
    for name, X in feats.items():
        p = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, G):
            if len(set(y[tr])) < 2: p[te] = y[tr].mean(); continue
            p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof[name] = p
        rows.append(dict(model=name, auroc=roc_auc_score(y, p), bal_acc=balanced_accuracy_score(y, p >= .5)))
    R = pd.DataFrame(rows); R.to_csv("results/rf_curl_eval.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200); print(R.round(3).to_string(index=False))
    # 영상별 (D) 성능 — 특정 영상에만 잘 되는지 확인
    pv = []
    for g in sorted(set(G)):
        k = G == g
        if len(set(y[k])) < 2: continue
        pv.append(dict(video=g[:45], frames=int(k.sum()), correct_rate=y[k].mean(), auroc_A=roc_auc_score(y[k], sA[k]), auroc_D=roc_auc_score(y[k], oof["D 둘 다(B+C)"][k])))
    PV = pd.DataFrame(pv); PV.to_csv("results/tables/rf_curl_per_video.csv", index=False, encoding="utf-8-sig"); print(PV.round(3).to_string(index=False))
    # 앱용 (D) 모델: 전체 데이터로 학습
    X = feats["D 둘 다(B+C)"]; sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    best = R[R.model.str.startswith("D")].iloc[0]
    json.dump(dict(exercise=EX, source="Roboflow Dumbbell Biceps Curl Detection v12 (humanarm, CC BY 4.0)", frames=int(len(y)), videos=int(len(set(G))),
                   feature="photo 특징 89 (사진 모델 기준 표준화·±3 자르기) + AI Hub 조건 로짓 5", mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                   coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]), cv_auroc=float(best.auroc), cv_bal_acc=float(best.bal_acc)),
              open("rf_cache/rf_curl_model.json", "w", encoding="utf-8"), ensure_ascii=False)   # 앱에는 쓰지 않음 (rf_phase_check.py: 라벨이 동작 단계와 얽힘)
    print("saved rf_cache/rf_curl_model.json")

if __name__ == "__main__":
    main()
