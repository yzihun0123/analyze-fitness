#!/usr/bin/env python3
"""실험 결과 -> 그림(figures/) + 표(results/tables/, results/report_tables.md).

사용:  python analyze_results.py
입력: results/all_logreg_{3d_paired,2d_paired,2d_all2d}_metrics.csv, results/rep6_{3d,2d}_metrics.csv / _oof.npz, cache/
보고서 수치는 모두 이 스크립트가 쓴 표에서 가져온다.
"""
import os, sys, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cache import JOINTS
from exp_seq import REP6, load_cache, norm3d, summary, metrics

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
C = {"logreg": "#2a78d6", "cnn": "#eb6834", "gru": "#1baf7a", "maj": "#a3a29c", "3d": "#2a78d6", "2d": "#eb6834"}
MN = {"logreg": "요약특징+LogReg", "cnn": "1D-CNN", "gru": "GRU"}
os.makedirs("figures", exist_ok=True); os.makedirs("results/tables", exist_ok=True)
MD = []   # report_tables.md 조각

# 조건 유형: 위에서부터 처음 맞는 규칙 하나 (조건 이름 키워드 기반, 대략적 분류)
TYPE_RULES = [("손목 중립/각도", ["손목의"]), ("이완/긴장 유지", ["긴장"]), ("어깨 으쓱", ["으쓱"]),
              ("견갑(숄더패킹·하강·고정)", ["숄더패킹", "견갑골 하강", "견갑대"]), ("시선·고개", ["시선", "고개", "경추"]),
              ("반동·흔들림", ["반동", "흔들림"]), ("척추·상체·허리", ["척추", "상체", "허리", "등의", "아치", "몸통과 엉덩이", "견갑골이 지면"]),
              ("무릎·다리·발", ["무릎", "다리", "발", "허벅지"]), ("팔꿈치·팔", ["팔꿈치", "팔", "전완", "상완", "상관"])]
def ctype(name):
    for t, kws in TYPE_RULES:
        if any(k in name for k in kws): return t
    return "기타(궤적·위치 등)"
HARD = lambda n: ctype(n) in ("손목 중립/각도", "이완/긴장 유지") or "숄더패킹" in n

def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):   # iterrows 는 int/float 혼합 행을 float 로 올려 버린다
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)

def boot_diff(Y, P1, P0, G, B=1000, seed=0):
    """수행자 단위 부트스트랩: balanced accuracy(P1) - balanced accuracy(P0) 의 95% CI"""
    from sklearn.metrics import balanced_accuracy_score as ba
    rng = np.random.default_rng(seed); zs = np.unique(G); idx = {z: np.where(G == z)[0] for z in zs}; d = []
    for _ in range(B):
        ii = np.concatenate([idx[z] for z in rng.choice(zs, len(zs))])
        if len(set(Y[ii])) < 2: continue
        d.append(ba(Y[ii], P1[ii] >= .5) - ba(Y[ii], P0[ii] >= .5))
    return np.percentile(d, [2.5, 97.5])

def load_oof(path):
    z = np.load(path, allow_pickle=False); d = {}
    for k in z.files:
        ex, f = k.split("|"); d.setdefault(ex, {})[f] = z[k]
    return d

# ---------------- 1. 41종 전체 기준 모델(3D) ----------------
A3 = pd.read_csv("results/all_logreg_3d_paired_metrics.csv")
A3["type"] = A3["condition"].map(ctype)
A3[["exercise", "condition", "type"]].to_csv("results/tables/condition_types.csv", index=False, encoding="utf-8-sig")
ex_tab = A3.groupby("exercise").agg(clips=("clips", "first"), performers=("performers", "first"), conditions=("condition", "count"),
                                    acc=("acc", "mean"), majority=("majority", "mean"), bal_acc=("bal_acc", "mean"),
                                    macro_f1=("macro_f1", "mean"), auroc=("auroc", "mean"),
                                    bal_acc_min=("bal_acc", "min"), bal_acc_max=("bal_acc", "max")).sort_values("bal_acc")
ex_tab.to_csv("results/tables/exercise_summary_3d_logreg.csv", encoding="utf-8-sig")
MD.append(f"## T1. 41종 운동별 기준 모델(3D 요약특징+LogReg) — 조건 평균, 수행자 분리 8-fold\n\n운동 {len(ex_tab)}종, 조건 {len(A3)}개, 클립 {int(ex_tab['clips'].sum())}개\n\n"
          + md_table(ex_tab.reset_index().sort_values("bal_acc", ascending=False)))

fig, ax = plt.subplots(figsize=(8, 11)); e = ex_tab.sort_values("acc"); y = np.arange(len(e))
ax.barh(y, e["acc"], height=0.62, color=C["logreg"], label="정확도 (LogReg, 3D)")
ax.scatter(e["majority"], y, marker="|", s=140, color="#0b0b0b", linewidths=2, zorder=3, label="다수 클래스 기준")
ax.scatter(e["bal_acc"], y, marker="o", s=18, color="#eda100", edgecolor="#fcfcfb", zorder=4, label="balanced accuracy")
ax.set_yticks(y, [f"{n} ({c})" for n, c in zip(e.index, e["clips"])], fontsize=8); ax.set_xlim(0.4, 1.0)
ax.set_xlabel("조건 평균 (수행자 분리 8-fold)"); ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.set_title("운동별 판별 정확도 vs 다수 클래스 기준 (41종)", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig("figures/fig1_exercise_acc_vs_majority.png"); plt.close(fig)

s0 = {"조건 수": len(A3), "balanced acc 평균": A3.bal_acc.mean(), "balanced acc 중앙값": A3.bal_acc.median(),
      "AUROC 평균": A3.auroc.mean(), "bal_acc ≥ 0.80 조건": int((A3.bal_acc >= 0.8).sum()), "bal_acc < 0.65 조건": int((A3.bal_acc < 0.65).sum()),
      "정확도 ≤ 다수 클래스 기준 조건": int((A3.acc <= A3.majority).sum()), "AUROC < 0.6 조건": int((A3.auroc < 0.6).sum())}
MD.append("## T0. 41종 기준 모델 요약 (3D LogReg)\n\n" + md_table(pd.DataFrame([s0])) + "\n\n정확도가 다수 클래스 기준 이하인 조건:\n\n"
          + md_table(A3[A3.acc <= A3.majority][["exercise", "condition", "performers", "pos_rate", "majority", "acc", "bal_acc", "auroc"]])
          + "\n\nbal_acc 하위 10:\n\n" + md_table(A3.nsmallest(10, "bal_acc")[["exercise", "condition", "performers", "majority", "bal_acc", "auroc"]]))
ty = A3.assign(gain=A3["acc"] - A3["majority"]).groupby("type").agg(n=("condition", "count"), acc=("acc", "mean"), bal_acc=("bal_acc", "mean"),
                                                                  auroc=("auroc", "mean"), gain_over_majority=("gain", "mean")).sort_values("bal_acc")
ty.to_csv("results/tables/condition_type_summary.csv", encoding="utf-8-sig")
MD.append("## T2. 조건 유형별 평균 (3D LogReg, 41종; 유형은 조건 이름 키워드 기반 대략 분류)\n\n" + md_table(ty.reset_index().sort_values("bal_acc", ascending=False)))
fig, ax = plt.subplots(figsize=(7, 4.2)); y = np.arange(len(ty))
ax.barh(y, ty["bal_acc"], height=0.6, color=[C["cnn"] if t in ("손목 중립/각도", "이완/긴장 유지") else C["logreg"] for t in ty.index])
for i, (v, n) in enumerate(zip(ty["bal_acc"], ty["n"])): ax.text(v + 0.005, i, f"{v:.2f} (n={n})", va="center", fontsize=8, color="#52514e")
ax.set_yticks(y, ty.index, fontsize=9); ax.set_xlim(0.5, 1.0); ax.set_xlabel("balanced accuracy 평균")
ax.set_title("조건 유형별 판별 성능 (3D 요약특징+LogReg)", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig("figures/fig2_condition_type.png"); plt.close(fig)

# ---------------- 2. 대표 6종: 기준 vs 시퀀스 모델 ----------------
done = [inp for inp in ["3d", "2d"] if os.path.exists(f"results/rep6_{inp}_oof.npz")]   # oof 는 실행 끝에 저장됨 = 완료 표시
R = {inp: pd.read_csv(f"results/rep6_{inp}_metrics.csv") for inp in done}
O = {inp: load_oof(f"results/rep6_{inp}_oof.npz") for inp in done}
R3_seed0 = R.get("3d")   # 2D 는 시드 1개이므로 2D vs 3D 비교는 시드 0끼리 한다
SEEDS = [f"results/rep6_3d_seed{s}_oof.npz" for s in (1, 2)]
if "3d" in R and all(os.path.exists(p) for p in SEEDS):
    # 시드 견고성: CNN·GRU 를 시드 3개(0,100,200)로 학습. 주 결과는 3시드 평균 확률(앙상블)로 교체한다.
    seeds = [O["3d"]] + [load_oof(p) for p in SEEDS]; rows = []
    for ex, d in O["3d"].items():
        for m in ["cnn", "gru"]:
            Ps = [s[ex][m] for s in seeds]
            for j, cn in enumerate(d["names"]):
                per = [metrics(d["Y"][:, j], P[:, j])["bal_acc"] for P in Ps]
                rows.append(dict(exercise=ex, condition=cn, model=m, **{f"seed{k}": v for k, v in enumerate(per)}))
            d[m] = np.mean(Ps, 0)
    sd = pd.DataFrame(rows); sd["mean"] = sd[["seed0", "seed1", "seed2"]].mean(1); sd["sd"] = sd[["seed0", "seed1", "seed2"]].std(1)
    sd.to_csv("results/tables/rep6_3d_seed_robustness.csv", index=False, encoding="utf-8-sig")
    new = []
    for ex, d in O["3d"].items():
        for m in ["cnn", "gru"]:
            for j, cn in enumerate(d["names"]):
                base = R["3d"].query("exercise==@ex and condition==@cn and model==@m").iloc[0].to_dict()
                base.update(metrics(d["Y"][:, j], d[m][:, j])); new.append(base)
    R["3d"] = pd.concat([R["3d"][R["3d"].model == "logreg"], pd.DataFrame(new)], ignore_index=True)
    ss = sd.groupby("model")[["seed0", "seed1", "seed2"]].mean().assign(**{"3시드 앙상블": R["3d"].groupby("model")["bal_acc"].mean()[["cnn", "gru"]]})
    ss.loc["logreg(결정적)"] = [R["3d"].query("model=='logreg'")["bal_acc"].mean()] * 4
    MD.append("## T3b. 시드 견고성 (3D, 대표 6종; 조건 평균 balanced accuracy). 이후 표의 CNN·GRU는 3시드 앙상블\n\n" + md_table(ss.reset_index())
              + f"\n\n조건별 시드 간 표준편차: CNN 평균 {sd.query('model==\"cnn\"')['sd'].mean():.3f}(최대 {sd.query('model==\"cnn\"')['sd'].max():.3f}), "
              + f"GRU 평균 {sd.query('model==\"gru\"')['sd'].mean():.3f}(최대 {sd.query('model==\"gru\"')['sd'].max():.3f})")
if "3d" in R:
    r3 = R["3d"]; wide = r3.pivot_table(index=["exercise", "condition"], columns="model", values=["bal_acc", "macro_f1", "auroc"], sort=False)
    wide.columns = [f"{m}_{k}" for k, m in wide.columns]; wide = wide.reset_index()
    maj = r3[r3.model == "logreg"].set_index(["exercise", "condition"])["majority"]
    wide.insert(2, "majority", [maj[(e, c)] for e, c in zip(wide.exercise, wide.condition)])
    for m in ["cnn", "gru"]:
        ci = []
        for e, c in zip(wide.exercise, wide.condition):
            d = O["3d"][e]; j = list(d["names"]).index(c)
            ci.append(boot_diff(d["Y"][:, j], d[m][:, j], d["logreg"][:, j], d["G"]))
        wide[f"{m}-logreg_bal_acc"] = wide[f"{m}_bal_acc"] - wide["logreg_bal_acc"]
        wide[f"{m}-logreg_CI"] = [f"[{a:+.3f}, {b:+.3f}]" for a, b in ci]
        wide[f"{m}_sig"] = [("+" if a > 0 else "-" if b < 0 else "") for a, b in ci]
    wide["hard"] = wide["condition"].map(HARD); wide["type"] = wide["condition"].map(ctype)
    wide.to_csv("results/tables/rep6_3d_model_compare.csv", index=False, encoding="utf-8-sig")
    cols = ["exercise", "condition", "majority", "logreg_bal_acc", "cnn_bal_acc", "gru_bal_acc", "cnn-logreg_CI", "gru-logreg_CI"]
    MD.append("## T3. 대표 6종 조건별 balanced accuracy — 3D 입력, 같은 폴드 (차이의 95% CI는 수행자 단위 부트스트랩 1000회)\n\n" + md_table(wide[cols]))
    s = r3.groupby("model")[["acc", "bal_acc", "macro_f1", "auroc"]].mean().loc[["logreg", "cnn", "gru"]]
    s.to_csv("results/tables/rep6_3d_model_mean.csv", encoding="utf-8-sig")
    sig = ", ".join(f"{MN[m]}: 유의하게 좋음 {(wide[f'{m}_sig'] == '+').sum()} / 유의하게 나쁨 {(wide[f'{m}_sig'] == '-').sum()} / 차이 없음 {(wide[f'{m}_sig'] == '').sum()}" for m in ["cnn", "gru"])
    MD.append("## T4. 대표 6종 모델별 평균 (3D, 조건 " + str(len(wide)) + "개)\n\n" + md_table(s.reset_index())
              + f"\n\nLogReg 대비 조건별 95% CI(다중비교 보정 없음) — {sig}")
    hard = wide[wide.hard]; easy = wide[~wide.hard]
    hs = pd.DataFrame({g: {f"{m}_bal_acc": df[f"{m}_bal_acc"].mean() for m in ["logreg", "cnn", "gru"]} | {"n": len(df)} for g, df in [("어려운 조건(손목·긴장 유지·숄더패킹)", hard), ("나머지 조건", easy)]}).T
    hs["n"] = hs["n"].astype(int); hs.to_csv("results/tables/rep6_3d_hard_vs_rest.csv", encoding="utf-8-sig")
    MD.append("## T5. 어려운 조건 vs 나머지 (3D, 대표 6종)\n\n" + md_table(hard[["exercise", "condition", "logreg_bal_acc", "cnn_bal_acc", "gru_bal_acc", "cnn-logreg_CI", "gru-logreg_CI"]])
              + "\n\n" + md_table(hs.reset_index().rename(columns={"index": "group"})))

    fig, ax = plt.subplots(figsize=(8.5, 9)); w = wide.iloc[::-1].reset_index(drop=True); y = np.arange(len(w))
    for k, (m, dy) in enumerate([("logreg", -0.22), ("cnn", 0), ("gru", 0.22)]):
        ax.scatter(w[f"{m}_bal_acc"], y + dy, s=26, color=C[m], edgecolor="#fcfcfb", linewidth=0.8, label=MN[m], zorder=3)
    ax.hlines(y, w[["logreg_bal_acc", "cnn_bal_acc", "gru_bal_acc"]].min(1), w[["logreg_bal_acc", "cnn_bal_acc", "gru_bal_acc"]].max(1), color="#d6d5d0", lw=1, zorder=2)
    ax.set_yticks(y, [("★ " if h else "") + f"{e} · {c}" for e, c, h in zip(w.exercise, w.condition, w.hard)], fontsize=7.5)
    ax.axvline(0.5, color="#a3a29c", lw=1, ls="--"); ax.set_xlabel("balanced accuracy (수행자 분리 8-fold)")
    ax.legend(loc="lower right", fontsize=8, frameon=False); ax.set_title("대표 6종: 요약특징+LogReg vs 1D-CNN vs GRU (3D; CNN·GRU는 3시드 앙상블)  ★=어려운 조건", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig("figures/fig3_rep6_models_3d.png"); plt.close(fig)

# ---------------- 3. 2D 5시점 vs 3D ----------------
if "3d" in R and "2d" in R:
    m = R3_seed0.merge(R["2d"], on=["exercise", "condition", "model"], suffixes=("_3d", "_2d"))
    assert (m["clips_3d"] == m["clips_2d"]).all()
    s = m.groupby("model")[["bal_acc_3d", "bal_acc_2d", "auroc_3d", "auroc_2d"]].mean().loc[["logreg", "cnn", "gru"]]
    s.to_csv("results/tables/rep6_2d_vs_3d_mean.csv", encoding="utf-8-sig")
    tab = m.pivot_table(index=["exercise", "condition"], columns="model", values=["bal_acc_3d", "bal_acc_2d"], sort=False)
    tab.columns = [f"{mm}_{k[8:]}" for k, mm in tab.columns]; tab = tab.reset_index()
    tab = tab[["exercise", "condition"] + [f"{mm}_{i}" for mm in ["logreg", "cnn", "gru"] for i in ["3d", "2d"]]]
    tab.to_csv("results/tables/rep6_2d_vs_3d.csv", index=False, encoding="utf-8-sig")
    MD.append("## T6. 대표 6종 2D 5시점 vs 3D (같은 클립·같은 폴드·CNN/GRU 시드 0끼리; 6종은 2D 전용 클립이 없어 '전체 기준'과 동일)\n\n" + md_table(s.reset_index()) + "\n\n" + md_table(tab))
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True, sharey=True)
    for ax, mm in zip(axs, ["logreg", "cnn", "gru"]):
        d = m[m.model == mm]; ax.plot([0.5, 1], [0.5, 1], color="#a3a29c", lw=1, ls="--")
        ax.scatter(d["bal_acc_3d"], d["bal_acc_2d"], s=22, color=C[mm], edgecolor="#fcfcfb", lw=0.8)
        ax.set_title(f"{MN[mm]}  (평균 3D {d.bal_acc_3d.mean():.3f} / 2D {d.bal_acc_2d.mean():.3f})", fontsize=9, loc="left"); ax.set_xlabel("3D balanced acc")
    axs[0].set_ylabel("2D 5시점 balanced acc"); axs[0].set_xlim(0.5, 1); axs[0].set_ylim(0.5, 1)
    fig.suptitle("조건별 3D vs 2D 5시점 (대각선 위 = 2D가 더 좋음)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig("figures/fig4_2d_vs_3d_rep6.png"); plt.close(fig)

L = {k: pd.read_csv(f"results/all_logreg_{k}_metrics.csv") for k in ["3d_paired", "2d_paired", "2d_all2d"]}
mm = L["3d_paired"].merge(L["2d_paired"], on=["exercise", "condition"], suffixes=("_3d", "_2dp")).merge(
    L["2d_all2d"].rename(columns={c: c + "_2da" for c in L["2d_all2d"].columns if c not in ("exercise", "condition")}), on=["exercise", "condition"])
ex41 = mm.groupby("exercise").agg(clips_paired=("clips_3d", "first"), clips_all2d=("clips_2da", "first"), bal_acc_3d=("bal_acc_3d", "mean"),
                                  bal_acc_2d_paired=("bal_acc_2dp", "mean"), bal_acc_2d_all2d=("bal_acc_2da", "mean"))
ex41["2d-3d (같은 클립)"] = ex41["bal_acc_2d_paired"] - ex41["bal_acc_3d"]
ex41.to_csv("results/tables/all41_2d_vs_3d_logreg.csv", encoding="utf-8-sig")
tot = pd.DataFrame({"클립 집합": ["3D (paired)", "2D 5시점 (paired, 같은 클립)", "2D 5시점 (all2d, 2D 전용 포함)"],
                    "clips": [int(ex41.clips_paired.sum()), int(ex41.clips_paired.sum()), int(ex41.clips_all2d.sum())],
                    "bal_acc 평균(조건)": [mm.bal_acc_3d.mean(), mm.bal_acc_2dp.mean(), mm.bal_acc_2da.mean()],
                    "AUROC 평균(조건)": [mm.auroc_3d.mean(), mm.auroc_2dp.mean(), mm.auroc_2da.mean()]})
tot.to_csv("results/tables/all41_2d_vs_3d_overall.csv", index=False, encoding="utf-8-sig")
MD.append("## T7. 41종 전체 LogReg: 3D vs 2D 5시점 (같은 클립 기준 + 전체 2D 기준)\n\n" + md_table(tot) + "\n\n"
          + md_table(ex41[ex41.clips_all2d != ex41.clips_paired].reset_index()))

V = {f"2dv{v}": f"results/all_logreg_2dv{v}_paired_metrics.csv" for v in range(1, 6)}
if all(os.path.exists(p) for p in V.values()):
    sv = A3[["exercise", "condition", "type", "bal_acc"]].rename(columns={"bal_acc": "3D"})
    sv = sv.merge(L["2d_paired"][["exercise", "condition", "bal_acc"]].rename(columns={"bal_acc": "2D 5시점"}), on=["exercise", "condition"])
    for k, p in V.items():
        sv = sv.merge(pd.read_csv(p)[["exercise", "condition", "bal_acc"]].rename(columns={"bal_acc": f"2D 시점{'ABCDE'[int(k[-1]) - 1]}"}), on=["exercise", "condition"])
    one = [f"2D 시점{c}" for c in "ABCDE"]; sv["단일 시점 평균"] = sv[one].mean(1)
    sv.to_csv("results/tables/all41_single_view.csv", index=False, encoding="utf-8-sig")
    vt = sv.groupby("type")[["3D", "2D 5시점"] + one + ["단일 시점 평균"]].mean(); vt.loc["전체"] = sv[["3D", "2D 5시점"] + one + ["단일 시점 평균"]].mean()
    vt.insert(0, "n", sv.groupby("type").size().tolist() + [len(sv)]); vt.to_csv("results/tables/all41_single_view_by_type.csv", encoding="utf-8-sig")
    MD.append("## T12. 41종 LogReg: 3D vs 2D 5시점 결합 vs 2D 단일 시점(A~E; 카메라 위치는 문서에 없음) — 조건 유형별 balanced accuracy, 같은 클립·같은 폴드\n\n"
              + md_table(vt.reset_index().sort_values("3D")))

# ---------------- 4. 어려운 조건 분석: 관절 중요도 + 오분류 사례 ----------------
metas, X2, A, X3 = load_cache("cache", "paired")
exa = np.array([r["exercise"] for r in metas])
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
CASES = [("덤벨 컬", "손목의 중립"), ("덤벨 컬", "이완 시 팔 긴장 유지"), ("랫풀 다운", "이완 시 숄더패킹"), ("덤벨 컬", "팔꿈치 위치 고정"), ("랫풀 다운", "수축 시 몸통-팔꿈치 사이 모아줌")]
imp = {}
for ex, cn in CASES:
    ids = np.where(exa == ex)[0]; j = metas[ids[0]]["names"].index(cn)
    S = norm3d(X3[ids]); F = StandardScaler().fit_transform(summary(S)); Y = np.array([metas[i]["vals"][j] for i in ids])
    w = np.abs(LogisticRegression(C=0.1, max_iter=3000).fit(F, Y).coef_[0]).reshape(3, 24, 3)   # (통계, 관절, 축)
    v = w.sum((0, 2)); imp[f"{ex} · {cn}"] = v / v.sum()
I = pd.DataFrame(imp, index=JOINTS).T; I.to_csv("results/tables/joint_importance.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(10, 3.4)); im = ax.imshow(I.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(24), JOINTS, rotation=60, ha="right", fontsize=7.5); ax.set_yticks(range(len(I)), I.index, fontsize=8); ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.02, label="|계수| 비중"); ax.set_title("관절별 중요도 (표준화 LogReg 계수 절댓값 합의 비중, 전체 데이터 적합)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/fig5_joint_importance.png"); plt.close(fig)
MD.append("## T8. 관절 중요도 상위 5 (어려운 조건 3개 vs 쉬운 조건 2개)\n\n" + "\n".join(f"- {k}: " + ", ".join(f"{JOINTS[i]} {I.loc[k].iloc[i]:.3f}" for i in np.argsort(-I.loc[k].values)[:5]) for k in I.index))

# 오분류 사례: 3D OOF 에서 가장 좋은 모델 기준, 확신 높은 오답 vs 정답 (덤벨 컬 손목의 중립)
if "3d" in O:
    BONES = [("Neck", "Nose"), ("Neck", "Left Shoulder"), ("Neck", "Right Shoulder"), ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
             ("Left Wrist", "Left Palm"), ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"), ("Right Wrist", "Right Palm"),
             ("Neck", "Back"), ("Back", "Waist"), ("Waist", "Left Hip"), ("Waist", "Right Hip"), ("Left Hip", "Left Knee"), ("Left Knee", "Left Ankle"),
             ("Left Ankle", "Left Foot"), ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"), ("Right Ankle", "Right Foot")]
    path2i = {r["path"]: i for i, r in enumerate(metas)}; case_rows = []
    for ex, cn in CASES[:3]:
        d = O["3d"][ex]; j = list(d["names"]).index(cn); best = max(["logreg", "cnn", "gru"], key=lambda k: R["3d"].query("exercise==@ex and condition==@cn and model==@k")["bal_acc"].iloc[0])
        p, y = d[best][:, j], d["Y"][:, j]
        pick = {"정답: 올바름(True)": np.argmax(np.where(y == 1, p, -1)), "정답: 오류(False)": np.argmin(np.where(y == 0, p, 2)),
                "오분류: 올바름→오류로 예측": np.argmin(np.where(y == 1, p, 2)), "오분류: 오류→올바름으로 예측": np.argmax(np.where(y == 0, p, -1))}
        fig, axs = plt.subplots(2, 4, figsize=(11, 5.6))
        for c, (lab, k) in enumerate(pick.items()):
            a = norm3d(X3[[path2i[d["paths"][k]]]])[0].reshape(16, 24, 3)
            for r, (ax_i, nm) in enumerate([((0, 1), "정면 x-y"), ((2, 1), "측면 z-y")]):
                ax = axs[r, c]; ax.grid(False)
                for t in [0, 8, 15]:
                    for b0, b1 in BONES:
                        q = a[t, [JOINTS.index(b0), JOINTS.index(b1)]]
                        ax.plot(q[:, ax_i[0]], q[:, ax_i[1]], color=["#c3c2b7", "#86b6ef", "#1c5cab"][[0, 8, 15].index(t)], lw=1.2)
                for wr, col in [("Left Wrist", C["cnn"]), ("Right Wrist", C["gru"])]:
                    tr = a[:, JOINTS.index(wr)]; ax.plot(tr[:, ax_i[0]], tr[:, ax_i[1]], color=col, lw=1.4, marker="o", ms=2)
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
                if r == 0: ax.set_title(f"{lab}\np(올바름)={p[k]:.2f}  {d['paths'][k].split(os.sep)[-1]}", fontsize=8)
                if c == 0: ax.set_ylabel(nm, fontsize=8)
            case_rows.append(dict(exercise=ex, condition=cn, model=best, case=lab, path=d["paths"][k], performer=d["G"][k], p_true=float(p[k]), label=int(y[k])))
        fig.suptitle(f"{ex} · {cn} ({MN[best]} 아웃오브폴드 예측) — 골격: 프레임 1·9·16(연→진), 궤적: 왼손목(주황)·오른손목(청록)", fontsize=9.5, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(f"figures/fig6_cases_{ex}_{cn}.png".replace(" ", "_")); plt.close(fig)
    pd.DataFrame(case_rows).to_csv("results/tables/misclassified_cases.csv", index=False, encoding="utf-8-sig")

# 조건 정의에 대응하는 기하 특징 1개로 정/오류가 갈리는가 (단일 특징 AUROC, 방향 무관 = max(AUC, 1-AUC))
from sklearn.metrics import roc_auc_score
J = {j: JOINTS.index(j) for j in JOINTS}
def angle(a, b, c):
    """b 에서의 a-b-c 각도(도), a,b,c: (...,3)"""
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9), -1, 1)))
def geo(a, kind):
    """a: (N,16,24,3) 정규화 좌표 -> 클립당 스칼라"""
    if kind == "wrist":   # 전완 연장선과 손(Wrist->Palm) 사이 꺾임 = 180 - (Elbow-Wrist-Palm 각)
        return np.mean([180 - angle(a[:, :, J[f"{s} Elbow"]], a[:, :, J[f"{s} Wrist"]], a[:, :, J[f"{s} Palm"]]).mean(1) for s in ("Left", "Right")], 0)
    if kind == "elbow_max":   # 가장 펴졌을 때 팔꿈치 각도
        return np.mean([angle(a[:, :, J[f"{s} Shoulder"]], a[:, :, J[f"{s} Elbow"]], a[:, :, J[f"{s} Wrist"]]).max(1) for s in ("Left", "Right")], 0)
    if kind == "elbow_drift":   # 어깨 기준 팔꿈치 위치의 프레임 간 흔들림
        return np.mean([np.linalg.norm((a[:, :, J[f"{s} Elbow"]] - a[:, :, J[f"{s} Shoulder"]]).std(1), axis=-1) for s in ("Left", "Right")], 0)
GEO = [("wrist", "손목 꺾임 각(평균)", lambda c: c.startswith("손목의")),
       ("elbow_max", "팔꿈치 최대 각", lambda c: "팔 긴장 유지" in c),
       ("elbow_drift", "팔꿈치 흔들림", lambda c: c == "팔꿈치 위치 고정")]
lr_auc = A3.set_index(["exercise", "condition"])["auroc"]; geo_rows = []; dist = {}
for kind, label, sel in GEO:
    for ex in sorted(set(exa)):
        ids = np.where(exa == ex)[0]; names = metas[ids[0]]["names"]
        for j, cn in enumerate(names):
            if not sel(cn): continue
            a = norm3d(X3[ids]).reshape(len(ids), 16, 24, 3); v = geo(a, kind); y = np.array([metas[i]["vals"][j] for i in ids])
            auc = roc_auc_score(y, v); geo_rows.append(dict(feature=label, exercise=ex, condition=cn, clips=len(ids), single_feature_auroc=max(auc, 1 - auc),
                                                            mean_true=v[y == 1].mean(), mean_false=v[y == 0].mean(), logreg_auroc=lr_auc[(ex, cn)]))
            if ex == "덤벨 컬": dist[label] = (v, y, cn)
G_ = pd.DataFrame(geo_rows); G_.to_csv("results/tables/geometric_feature_auroc.csv", index=False, encoding="utf-8-sig")
MD.append("## T9. 조건 정의에 대응하는 기하 특징 1개의 AUROC (3D, 전체 클립; 방향 무관) vs 216특징 LogReg AUROC\n\n" + md_table(G_))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, (label, (v, y, cn)) in zip(axs, dist.items()):
    bins = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 30)
    ax.hist(v[y == 1], bins, color=C["logreg"], histtype="step", lw=2, label="올바름(True)"); ax.hist(v[y == 0], bins, color=C["cnn"], histtype="step", lw=2, label="오류(False)")
    r = G_.query("exercise=='덤벨 컬' and condition==@cn").iloc[0]
    ax.set_title(f"{cn}\n{label}: AUROC {r.single_feature_auroc:.2f} (LogReg {r.logreg_auroc:.2f})", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
axs[0].legend(fontsize=8, frameon=False); fig.suptitle("덤벨 컬: 조건 정의에 대응하는 기하 특징의 분포", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figures/fig7_geometric_features_dumbbell_curl.png"); plt.close(fig)

# ---------------- 5. 관절각 특징 추가 (3D 좌표 + 관절각 17) ----------------
if os.path.exists("results/all_logreg_3d_ang_paired_metrics.csv"):
    B = pd.read_csv("results/all_logreg_3d_ang_paired_metrics.csv")
    mb = A3.merge(B, on=["exercise", "condition"], suffixes=("_3d", "_ang"))
    mb["diff"] = mb["bal_acc_ang"] - mb["bal_acc_3d"]
    tb = mb.groupby("type").agg(n=("condition", "count"), bal_acc_3d=("bal_acc_3d", "mean"), bal_acc_3d_ang=("bal_acc_ang", "mean"),
                                auroc_3d=("auroc_3d", "mean"), auroc_3d_ang=("auroc_ang", "mean"), improved=("diff", lambda d: int((d > 0).sum()))).sort_values("bal_acc_3d")
    tb.loc["전체"] = [len(mb), mb.bal_acc_3d.mean(), mb.bal_acc_ang.mean(), mb.auroc_3d.mean(), mb.auroc_ang.mean(), int((mb["diff"] > 0).sum())]
    tb.to_csv("results/tables/all41_3d_vs_3d_ang_by_type.csv", encoding="utf-8-sig")
    mb[["exercise", "condition", "type", "performers_3d", "bal_acc_3d", "bal_acc_ang", "diff"]].to_csv("results/tables/all41_3d_vs_3d_ang.csv", index=False, encoding="utf-8-sig")
    MD.append("## T10. 41종 LogReg: 3D 좌표 vs 3D 좌표+관절각 (같은 폴드; 조건 유형별 balanced accuracy·AUROC 평균)\n\n" + md_table(tb.reset_index())
              + "\n\n개선 상위 8 / 하락 상위 5:\n\n" + md_table(pd.concat([mb.nlargest(8, "diff"), mb.nsmallest(5, "diff")])[["exercise", "condition", "performers_3d", "bal_acc_3d", "bal_acc_ang", "diff"]]))
    t = tb.drop("전체"); fig, ax = plt.subplots(figsize=(7.5, 4.4)); y = np.arange(len(t))
    ax.barh(y - 0.19, t["bal_acc_3d"], height=0.36, color=C["3d"], label="3D 좌표 요약 (216)")
    ax.barh(y + 0.19, t["bal_acc_3d_ang"], height=0.36, color=C["gru"], label="3D 좌표 + 관절각 요약 (267)")
    ax.set_yticks(y, [f"{i} (n={n})" for i, n in zip(t.index, t["n"])], fontsize=8.5); ax.set_xlim(0.5, 1.0); ax.set_xlabel("balanced accuracy 평균 (LogReg, 41종)")
    ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.set_title("관절각 특징 추가 효과 (조건 유형별)", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig("figures/fig8_angle_features_by_type.png"); plt.close(fig)
if os.path.exists("results/rep6_3d_ang_oof.npz") and "3d" in R:
    Rang = pd.read_csv("results/rep6_3d_ang_metrics.csv")
    x = pd.concat([R3_seed0.assign(input="3d"), Rang.assign(input="3d_ang")])
    pv = x.pivot_table(index=["exercise", "condition"], columns=["input", "model"], values="bal_acc", sort=False)
    pv.columns = [f"{m}_{i}" for i, m in pv.columns]; pv = pv.reset_index(); pv["hard"] = pv["condition"].map(HARD)
    pv.to_csv("results/tables/rep6_3d_vs_3d_ang.csv", index=False, encoding="utf-8-sig")
    g = pv.groupby("hard")[[c for c in pv.columns if c.endswith(("_3d", "_3d_ang"))]].mean().rename(index={True: "어려운 조건", False: "나머지 조건"})
    g.loc["전체"] = pv[[c for c in pv.columns if c.endswith(("_3d", "_3d_ang"))]].mean()
    g.to_csv("results/tables/rep6_3d_vs_3d_ang_mean.csv", encoding="utf-8-sig")
    MD.append("## T11. 대표 6종: 입력(3D vs 3D+관절각) × 모델 (balanced accuracy, 시드 0, 같은 폴드)\n\n" + md_table(g.reset_index().rename(columns={"hard": "group"}))
              + "\n\n" + md_table(pv[pv.hard].drop(columns="hard")))

open("results/report_tables.md", "w", encoding="utf-8").write("# 자동 생성 결과 표 (analyze_results.py)\n\n" + "\n\n".join(MD) + "\n")
print("done:", sorted(os.listdir("figures")))
