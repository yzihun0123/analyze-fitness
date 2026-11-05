#!/usr/bin/env python3
"""제출용 노트북 fitness_pose_project.ipynb 생성 (셀 내용은 여기서 관리).

사용:  python make_notebook.py  →  jupyter nbconvert --to notebook --execute --inplace fitness_pose_project.ipynb
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# 헬스 운동 자세 오류의 조건별 판별 가능성 분석
### 3D 키포인트 기반 딥러닝과 수행자 분리 평가

- 데이터: AI Hub 「피트니스 자세 이미지」(dataSetSn=231) **라벨 JSON**(기구 · 바벨/덤벨 · 맨몸, 41종 운동)
- 과제: 운동별 자세 조건(척추 중립, 손목 중립 등 3~5개)의 **정/오류 다중 라벨 분류**
- 평가: **수행자(img_key의 Z코드) 분리** GroupKFold(8) — 같은 사람의 클립은 학습/평가에 동시에 들어가지 않음
- 발표·시연 앱: `app.py` (Streamlit), 보고서: `REPORT.md`

| 연구 질문 | 내용 |
|---|---|
| RQ1 | 3D 키포인트만으로 조건별 정/오류를 얼마나 판별할 수 있는가 |
| RQ2 | 시퀀스 딥러닝(1D-CNN, GRU)이 요약 특징 + 선형 모델보다 어려운 조건에서 나은가 |
| RQ3 | 3D 좌표 vs 2D 5시점 입력 |
| RQ4 | 어려운 조건은 왜 어려운가 |

> 이 노트북은 오래 걸리는 단계(전체 실험 ≈ 1.5시간)를 `.py` 스크립트로 실행한 결과를 불러오고, 핵심 단계(전처리·기준 모델·딥러닝 1개 운동)는 노트북 안에서 직접 실행합니다."""),
code("""import os, sys, json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
pd.set_option("display.precision", 3); pd.set_option("display.max_colwidth", 40)
sys.path.insert(0, os.getcwd())
from build_cache import JOINTS
from exp_seq import load_cache, norm3d, norm2d, angles3d, summary, fit_predict, metrics, make_keras, REP6
print("작업 폴더:", os.getcwd())"""),

md("""## 1. 데이터 준비와 검증

tar → 내부 zip → `D{day}-{slot}-{code}.json`(2D 5시점 + 라벨) / `...-3d.json`(3D 좌표). 풀기와 집계는 스크립트로 실행했습니다.

```
python prepare_data.py data dataset/기구.tar "dataset/바벨,덤벨.tar" dataset/맨몸운동.tar ...
python count_data.py data --out logs/counts.json
```

> 2~4절은 원본 데이터 캐시(`cache/`)가 있어야 실행됩니다. 저장소에는 AI Hub 원본 데이터가 없으므로 저장된 출력으로 확인하세요.

**정제 규칙**: ① 16프레임 아닌 클립 제외 ② 라벨·Z코드 없음 제외 ③ 3D 짝 없음 제외(3D 실험) ④ **3D 이상값**(|좌표|>10⁴ 또는 정규화 특징 |f|>50) 제외"""),
code("""cnt = json.load(open("app_data/counts.json", encoding="utf-8"))   # = logs/counts.json (count_data.py 출력)
pd.DataFrame({k: {"2D JSON": v["json_2d"], "3D JSON": v["json_3d"], "2D만": v["only_2d"], "3D만(라벨 없음)": v["only_3d"],
                  "16프레임 아님": v["excluded"].get("2D 프레임 16 아님", 0), "3D 이상값": v["excluded"].get("3D 이상값", 0),
                  "사용 가능 클립": v["usable"], "수행자(Z)": v["performers"], "운동": v["exercises"]} for k, v in cnt.items()}).T"""),
md("""핸드오프 문서의 기대값(기구 4,262/4,253/32, 바벨·덤벨 14,745/14,653/95, 맨몸 15,461/13,076/62)과 **정확히 일치**합니다.

기존 기준 모델(`analyze_fitness.py`)도 재현했습니다 — 176개 운동×조건 모두 ±0.02 이내(최대 차이 0.0018)."""),
code("""cmp = pd.read_csv("results/baseline_repro_compare.csv")
print(f"재현 조건 수 {len(cmp)} | 클립 수 일치 {(cmp.clips == cmp.clips_repro).mean():.0%} | |정확도 차이| 최대 {cmp['diff'].abs().max():.4f}")"""),

md("""## 2. 캐시와 전처리

모든 실험은 `build_cache.py`가 만든 캐시(클립별 3D `(16,24,3)`, 2D `(16,5,24,2)`, 라벨, Z코드)를 읽습니다.

- **3D**: Neck 원점 이동 → 클립 평균 어깨너비로 나눔 → `(16, 72)`
- **2D 5시점**: 시점마다 Neck 원점 이동 → 몸통 길이(Neck–Waist)로 나눔 → `(16, 240)` (측면 시점은 어깨너비가 0에 가까워 몸통 길이 사용)
- **관절각**: 프레임별 관절각 15개 + 귀–어깨 거리 2개 → `(16, 17)`"""),
code("""metas, X2, A, X3 = load_cache("cache", "paired")
ex = np.array([m["exercise"] for m in metas])
print("클립", len(metas), "| 운동", len(set(ex)), "| 수행자", len({m['z'] for m in metas}))
print("3D", X3.shape, "→ norm3d", norm3d(X3[:2]).shape, "| 2D", X2.shape, "→ norm2d", norm2d(X2[:2]).shape, "| 관절각", angles3d(X3[:2].astype(float)).shape)"""),
code("""# 한 클립의 정규화 3D 골격 (프레임 1, 9, 16)
BONES = [("Neck","Nose"),("Neck","Left Shoulder"),("Neck","Right Shoulder"),("Left Shoulder","Left Elbow"),("Left Elbow","Left Wrist"),
         ("Left Wrist","Left Palm"),("Right Shoulder","Right Elbow"),("Right Elbow","Right Wrist"),("Right Wrist","Right Palm"),("Neck","Back"),
         ("Back","Waist"),("Waist","Left Hip"),("Waist","Right Hip"),("Left Hip","Left Knee"),("Left Knee","Left Ankle"),("Left Ankle","Left Foot"),
         ("Right Hip","Right Knee"),("Right Knee","Right Ankle"),("Right Ankle","Right Foot")]
i = int(np.where(ex == "바벨 스쿼트")[0][0]); a = norm3d(X3[[i]])[0].reshape(16, 24, 3)
fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
for ax, (u, v, nm) in zip(axs, [(0, 1, "정면 x-y"), (2, 1, "측면 z-y")]):
    for t, c in zip([0, 8, 15], ["#c3c2b7", "#86b6ef", "#1c5cab"]):
        for b0, b1 in BONES:
            p = a[t, [JOINTS.index(b0), JOINTS.index(b1)]]; ax.plot(p[:, u], p[:, v], color=c, lw=1.5)
    ax.set_aspect("equal"); ax.set_title(nm)
fig.suptitle(f"{metas[i]['exercise']} · 수행자 {metas[i]['z']} · 라벨 {dict(zip(metas[i]['names'], metas[i]['vals']))}", fontsize=9); plt.show()"""),

md("""## 3. 기준 모델 — 요약 특징 + 로지스틱 회귀 (수행자 분리 8-fold)

시퀀스 `(16, F)` → 프레임 축의 **평균 · 표준편차 · 가동범위** `(3F)` → 표준화 + LogReg(C=0.1), 조건마다 이진 분류.
아래는 **랫풀 다운** 한 운동으로 직접 실행한 예입니다 (3D 좌표 vs 3D 좌표 + 관절각)."""),
code("""from sklearn.model_selection import GroupKFold
def run(exercise, S_fn, model="logreg", seed=0):
    ids = np.where(ex == exercise)[0]; S = S_fn(ids); Y = np.array([metas[i]["vals"] for i in ids]); G = np.array([metas[i]["z"] for i in ids])
    P = np.zeros(Y.shape); t0 = time.time()
    for fi, (tr, te) in enumerate(GroupKFold(8).split(S, Y, G)):
        P[te], _ = fit_predict(model, S[tr], Y[tr], G[tr], S[te], seed + fi, 150)
    names = metas[ids[0]]["names"]
    return pd.DataFrame([dict(condition=n, **metrics(Y[:, j], P[:, j])) for j, n in enumerate(names)]).set_index("condition"), time.time() - t0

S3 = lambda ids: norm3d(X3[ids])
S3a = lambda ids: np.concatenate([norm3d(X3[ids]), angles3d(X3[ids].astype(float)).astype(np.float32)], -1)
r1, t1 = run("랫풀 다운", S3); r2, t2 = run("랫풀 다운", S3a)
pd.concat({"3D 좌표": r1[["bal_acc", "auroc"]], "3D 좌표+관절각": r2[["bal_acc", "auroc"]]}, axis=1)"""),
md("""→ **손목의 중립**이 관절각을 넣으면 크게 좋아집니다(표현의 문제), **이완 시 숄더패킹**은 그대로입니다(관측의 문제). 41종 전체 결과는 5절."""),

md("""## 4. 딥러닝 모델 — 1D-CNN, GRU (운동별 다중 라벨)

- 입력 `(16 프레임, F)`, 출력 = 조건 수만큼 **sigmoid** (binary cross-entropy, Adam 1e-3, batch 32)
- 학습 폴드 안에서 **수행자 15%** 를 검증으로 떼어 조기 종료(patience 15) — 테스트 수행자 누수 없음
- 기준 모델과 **같은 폴드**로 비교"""),
code("""for kind in ["cnn", "gru"]:
    make_keras(kind, 16, 72, 4, 0).summary()"""),
code("""res = {"LogReg": r1["bal_acc"]}
for kind in ["cnn", "gru"]:
    r, t = run("랫풀 다운", S3, kind); res[kind.upper()] = r["bal_acc"]; print(f"{kind}: 8-fold {t:.0f}s")
pd.DataFrame(res).assign(**{"(평균)": lambda d: d.mean(1)})"""),
md("""### 4-1. 학습 곡선 (에폭별 loss)
랫풀 다운, 수행자 분리 폴드 1개에서 1D-CNN·GRU 를 학습하며 에폭마다 train/val loss(binary cross-entropy)를 기록합니다. 점선 = 조기 종료가 고른 최적 에폭(val loss 최소)."""),
code("""import keras
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
ids = np.where(ex == "랫풀 다운")[0]; S = norm3d(X3[ids]); Y = np.array([metas[i]["vals"] for i in ids]); G = np.array([metas[i]["z"] for i in ids])
tr, te = next(GroupKFold(8).split(S, Y, G))
mu, sd = S[tr].mean((0, 1)), S[tr].std((0, 1)) + 1e-6; Str = (S[tr] - mu) / sd
itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(Str, groups=G[tr]))
hist = {}
for kind in ["cnn", "gru"]:
    m = make_keras(kind, 16, Str.shape[2], Y.shape[1], 0)
    h = m.fit(Str[itr], Y[tr][itr].astype("float32"), validation_data=(Str[iva], Y[tr][iva].astype("float32")), epochs=150, batch_size=32,
              callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)], verbose=0)
    hist[kind] = pd.DataFrame(h.history).rename_axis("epoch")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (k, h) in zip(axs, hist.items()):
    ax.plot(h.index + 1, h["loss"], label="train loss"); ax.plot(h.index + 1, h["val_loss"], label="val loss (검증 수행자)")
    best = int(h["val_loss"].idxmin()) + 1; ax.axvline(best, ls="--", c="gray")
    ax.set_title(f"{k.upper()} — 최적 에폭 {best} / 총 {len(h)}"); ax.set_xlabel("epoch"); ax.set_ylabel("binary cross-entropy"); ax.legend()
plt.tight_layout(); plt.show()
curves = pd.concat(hist, names=["model"]).reset_index(); curves.to_csv("results/tables/training_curves.csv", index=False, encoding="utf-8-sig")
pd.concat({k: h.iloc[[0, int(h["val_loss"].idxmin()), len(h) - 1]] for k, h in hist.items()}).round(4)"""),
md("### 4-2. 모델 비교 — 단일 시드 vs 3시드 앙상블 (대표 6종, 3D, 조건 26개 평균)"),
code("""sr = pd.read_csv("results/tables/rep6_3d_seed_robustness.csv"); W6 = pd.read_csv("results/tables/rep6_3d_model_compare.csv")
m = sr.groupby("model")[["seed0", "seed1", "seed2"]].mean()
tab = pd.DataFrame({"시드0": [W6.logreg_bal_acc.mean(), *m.loc[["cnn", "gru"], "seed0"]], "시드1": [np.nan, *m.loc[["cnn", "gru"], "seed1"]],
                    "시드2": [np.nan, *m.loc[["cnn", "gru"], "seed2"]], "3시드 앙상블": [W6.logreg_bal_acc.mean(), W6.cnn_bal_acc.mean(), W6.gru_bal_acc.mean()]},
                   index=["LogReg(결정적)", "1D-CNN", "GRU"])
ax = tab.plot.bar(figsize=(8, 3.8), rot=0, ylim=(0.78, 0.86), width=0.8); ax.set_ylabel("balanced accuracy"); ax.set_title("시드 1개로는 LogReg 보다 낮지만, 3시드 앙상블은 비슷하거나 높음")
plt.tight_layout(); plt.show(); tab.round(3)"""),

md("""한 번의 학습(시드 1개)은 무작위성이 커서, 본 실험에서는 대표 6종에 **시드 3개 × 8폴드**로 학습하고 확률을 평균(앙상블)했습니다:

```
python exp_seq.py --input 3d --exercises rep6 --models logreg,cnn,gru --out results/rep6_3d
python exp_seq.py --input 3d --exercises rep6 --models cnn,gru --seed 100 --out results/rep6_3d_seed1   # seed 200 → seed2
python exp_seq.py --input 2d --exercises rep6 --models logreg,cnn,gru --out results/rep6_2d
python exp_seq.py --input 3d_ang --exercises rep6 --models logreg,cnn,gru --out results/rep6_3d_ang
python exp_seq.py --input {3d,2d,3d_ang,2dv1..2dv5} --exercises all --models logreg --out results/all_logreg_...
python analyze_results.py      # 표 results/tables/, 그림 figures/
```"""),

md("### 4-3. 분류 보고서와 혼동 행렬 — 덤벨 컬 (수행자 분리 아웃오브폴드 예측, `make_ppt_figs.py`)"),
code("""rep = pd.read_csv("results/tables/classification_report_dumbbell_curl.csv")
display(rep[rep.cls != "accuracy"].pivot_table(index=["model", "condition"], columns="cls", values=["precision", "recall", "f1"]).round(3))
display(Image("figures/fig9_confusion_dumbbell_curl.png", width=950))"""),
md("## 5. 결과\n### 탐색적 데이터 분석(EDA)과 구축 프로세스"),
code("""display(Image("figures/fig10_eda.png", width=950)); display(Image("figures/fig11_pipeline.png", width=950))"""),
md("### RQ1 — 41종 운동별 판별 성능 (3D 요약특징 + LogReg)"),
code("""A3 = pd.read_csv("results/all_logreg_3d_paired_metrics.csv")
print(f"조건 {len(A3)}개 | balanced acc 평균 {A3.bal_acc.mean():.3f} (중앙값 {A3.bal_acc.median():.3f}) | AUROC 평균 {A3.auroc.mean():.3f} | 0.80 이상 {(A3.bal_acc >= .8).sum()}개")
display(Image("figures/fig1_exercise_acc_vs_majority.png", width=650))"""),
code("""display(pd.read_csv("results/tables/condition_type_summary.csv").sort_values("bal_acc", ascending=False))
display(Image("figures/fig2_condition_type.png", width=650))"""),
md("### RQ2 — 시퀀스 딥러닝 vs 기준 모델 (대표 6종, CNN·GRU는 3시드 앙상블)"),
code("""sr = pd.read_csv("results/tables/rep6_3d_seed_robustness.csv")
W = pd.read_csv("results/tables/rep6_3d_model_compare.csv")
display(sr.groupby("model")[["seed0", "seed1", "seed2"]].mean().assign(앙상블=[W.cnn_bal_acc.mean(), W.gru_bal_acc.mean()]).T.assign(logreg=W.logreg_bal_acc.mean()).T)
display(W[["exercise", "condition", "logreg_bal_acc", "cnn_bal_acc", "gru_bal_acc", "cnn-logreg_CI", "gru-logreg_CI", "hard"]])
display(Image("figures/fig3_rep6_models_3d.png", width=700))"""),
code("""display(Markdown("**어려운 조건 vs 나머지** (3시드 앙상블)")); display(pd.read_csv("results/tables/rep6_3d_hard_vs_rest.csv"))
display(Markdown("**입력 표현 변경: 3D vs 3D+관절각** (시드 0)")); display(pd.read_csv("results/tables/rep6_3d_vs_3d_ang_mean.csv"))"""),
md("### RQ3 — 3D vs 2D 5시점 vs 카메라 1대 (41종 LogReg, 같은 클립·같은 폴드)"),
code("""display(pd.read_csv("results/tables/all41_2d_vs_3d_overall.csv"))
display(pd.read_csv("results/tables/all41_single_view_by_type.csv"))
display(Image("figures/fig4_2d_vs_3d_rep6.png", width=800))"""),
md("### RQ4 — 어려운 조건은 왜 어려운가\n조건 정의에 대응하는 **기하 특징 1개**(학습 없음)의 AUROC vs 216특징 LogReg AUROC"),
code("""display(pd.read_csv("results/tables/geometric_feature_auroc.csv"))
display(Image("figures/fig7_geometric_features_dumbbell_curl.png", width=850))"""),
code("""display(pd.read_csv("results/tables/all41_3d_vs_3d_ang_by_type.csv"))
display(Image("figures/fig8_angle_features_by_type.png", width=650))
display(Image("figures/fig5_joint_importance.png", width=850))"""),

md("""## 6. 결론

1. 3D 좌표만으로 167개 조건 평균 balanced accuracy **0.792**, 89개 조건이 0.80 이상 — 시선·고개, 어깨 으쓱, 팔꿈치, 무릎 정렬은 자동 피드백 후보.
2. 1D-CNN·GRU(3시드 앙상블)는 LogReg와 평균이 비슷(0.841/0.837 vs 0.836). 큰 운동에서 낫고 작은 운동에서 못하며, **어려운 조건은 개선하지 못함**. 단일 시드로는 결론이 뒤집힐 수 있음.
3. **손목 중립 = 표현의 문제**: 관절각 특징으로 0.675 → 0.761.
4. **긴장 유지·숄더패킹 = 관측의 문제**: 관절각 차이 2° 미만, 어떤 모델·입력으로도 뚜렷한 개선 없음 → 키포인트 기반 피드백 대상에서 빼거나 다른 센서 필요.
5. 같은 클립에서 2D 5시점(0.802) ≥ 3D(0.792) > 카메라 1대(0.755~0.783).

**한계**: 연기된 오류(실제보다 쉬움), 24관절(손가락·견갑골 없음), 16프레임 고정, 시퀀스 모델은 6종·작은 구조만, Z코드=동일인 가정. 자세한 내용은 `REPORT.md`."""),

md("""## 7. 시연 앱용 모델 내보내기

Streamlit 앱(`app.py`)은 원본 데이터 없이 동작하도록 **파생물만** 사용합니다 — `python export_models.py`
- `app_data/models.json`: 41종 × 조건별 LogReg 계수(3D 좌표 + 관절각 267특징, 전체 클립 학습) + 교차검증 성능
- `app_data/mean_skeletons.npz`: 조건별 올바름/오류 **평균** 골격(집계값)"""),
code("""mj = json.load(open("app_data/models.json", encoding="utf-8"))["models"]
print("운동", len(mj), "| 조건", sum(len(m["conditions"]) for m in mj.values()), "| 파일 크기", {f: f"{os.path.getsize('app_data/' + f) / 1e6:.2f} MB" for f in os.listdir("app_data")})"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                  "language_info": {"name": "python"}})
nbf.write(nb, "fitness_pose_project.ipynb")
print("written fitness_pose_project.ipynb", len(cells), "cells")
