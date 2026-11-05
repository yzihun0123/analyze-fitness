#!/usr/bin/env python3
"""PPT 양식(산출물 예시)에 필요한 추가 그림: 혼동 행렬·분류 보고서, EDA, 구축 프로세스 도식.

사용:  python make_ppt_figs.py
출력: figures/fig9_confusion_dumbbell_curl.png, results/tables/classification_report_dumbbell_curl.csv,
      figures/fig10_eda.png, figures/fig11_pipeline.png
"""
import os, sys, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_seq import load_cache
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb"})

def confusion():
    from sklearn.metrics import confusion_matrix, classification_report
    z = np.load("results/rep6_3d_oof.npz"); ex = "덤벨 컬"; Y = z[f"{ex}|Y"]; names = list(z[f"{ex}|names"]); rows = []
    fig, axs = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6.4))
    for r, m in enumerate(["logreg", "cnn"]):
        P = z[f"{ex}|{m}"]
        for j, n in enumerate(names):
            cm = confusion_matrix(Y[:, j], P[:, j] >= .5, labels=[0, 1]); ax = axs[r, j]
            ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1], ["예측 오류", "예측 올바름"], fontsize=8); ax.set_yticks([0, 1], ["실제 오류", "실제 올바름"], fontsize=8)
            for a in range(2):
                for b in range(2): ax.text(b, a, cm[a, b], ha="center", va="center", color="white" if cm[a, b] > cm.max() / 2 else "#0b0b0b", fontsize=11)
            ax.set_title(f"{'LogReg' if m == 'logreg' else '1D-CNN'} · {n}", fontsize=8.5)
            rep = classification_report(Y[:, j], P[:, j] >= .5, labels=[0, 1], target_names=["오류", "올바름"], output_dict=True, zero_division=0)
            for k in ["오류", "올바름"]:
                rows.append(dict(model=m, condition=n, cls=k, precision=rep[k]["precision"], recall=rep[k]["recall"], f1=rep[k]["f1-score"], support=int(rep[k]["support"])))
            rows.append(dict(model=m, condition=n, cls="accuracy", f1=rep["accuracy"], support=len(Y)))
    fig.suptitle(f"{ex} 조건별 혼동 행렬 (수행자 분리 8-fold 아웃오브폴드 예측, 클립 {len(Y)}개)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig("figures/fig9_confusion_dumbbell_curl.png", dpi=130); plt.close(fig)
    pd.DataFrame(rows).to_csv("results/tables/classification_report_dumbbell_curl.csv", index=False, encoding="utf-8-sig")

def eda():
    metas, *_ = load_cache("cache", "paired")
    ex = collections.Counter(m["exercise"] for m in metas).most_common()
    pos = [np.mean(v) for e in {m["exercise"] for m in metas} for v in np.array([m["vals"] for m in metas if m["exercise"] == e]).T]
    perf = collections.Counter(m["z"] for m in metas)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    axs[0].barh([e for e, _ in ex][::-1], [n for _, n in ex][::-1], color="#2a78d6"); axs[0].tick_params(axis="y", labelsize=7)
    axs[0].set_title(f"운동별 사용 클립 수 (41종, 총 {len(metas):,})", loc="left"); axs[0].set_xlabel("클립")
    axs[1].hist(pos, bins=20, color="#eb6834"); axs[1].axvline(0.5, ls="--", c="gray")
    axs[1].set_title(f"조건별 '올바름' 비율 분포 ({len(pos)}개 조건)", loc="left"); axs[1].set_xlabel("올바름 비율 (0.5 = 균형)")
    axs[2].hist(list(perf.values()), bins=25, color="#1baf7a"); axs[2].set_title(f"수행자(Z코드)별 클립 수 ({len(perf)}명)", loc="left"); axs[2].set_xlabel("클립")
    fig.tight_layout(); fig.savefig("figures/fig10_eda.png", dpi=130); plt.close(fig)

def pipeline():
    steps = [("AI Hub 라벨 tar", "기구·바벨덤벨·맨몸\n(영상 미사용)"), ("JSON 풀기·검증", "prepare_data\ncount_data"), ("정제", "16프레임·라벨·Z코드\n3D 이상값 제외"),
             ("캐시", "3D (16,24,3)\n2D 5시점 (16,5,24,2)"), ("특징 / 시퀀스", "요약 특징 216·관절각\n시퀀스 (16,72)"), ("모델", "LogReg\n1D-CNN · GRU\n운동별 다중 라벨"),
             ("평가", "수행자(Z) 분리\nGroupKFold 8\n부트스트랩 CI"), ("분석", "조건 유형·기하 특징\n2D vs 3D\n원시 이미지 검증"), ("서비스", "Streamlit 앱\nJSON·영상·사진 판별")]
    fig, ax = plt.subplots(figsize=(17, 3.2)); ax.axis("off"); w = 1 / len(steps)
    for i, (t, s) in enumerate(steps):
        x = i * w + 0.005
        ax.add_patch(plt.Rectangle((x, 0.18), w - 0.02, 0.64, fc="#e8f0fb" if i < 6 else "#fde9df", ec="#2a78d6" if i < 6 else "#eb6834", lw=1.5, transform=ax.transAxes))
        ax.text(x + (w - 0.02) / 2, 0.66, t, ha="center", va="center", fontsize=11, weight="bold", transform=ax.transAxes)
        ax.text(x + (w - 0.02) / 2, 0.38, s, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
        if i < len(steps) - 1: ax.annotate("", xy=(x + w + 0.003, 0.5), xytext=(x + w - 0.018, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("AI 예측 분석모델 구축 프로세스", loc="left", fontsize=13)
    fig.savefig("figures/fig11_pipeline.png", dpi=130, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    confusion(); eda(); pipeline(); print("done")
