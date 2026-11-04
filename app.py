"""헬스 운동 자세 오류의 조건별 판별 가능성 — Streamlit 발표·시연 앱.

실행:  streamlit run app.py
원본 AI Hub 데이터 없이 동작한다. 사용하는 파일:
  app_data/        export_models.py 가 만든 모델 계수·평균 골격·데이터 집계
  results/         실험 결과 CSV (exp_seq.py, analyze_results.py)
  figures/         analyze_results.py 가 만든 그림
라이브 데모는 사용자가 업로드한 JSON 을 메모리에서만 처리한다(저장하지 않음).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from build_cache import JOINTS
from exp_seq import norm3d, angles3d, summary

BASE = Path(__file__).parent
RES, TAB, FIG, APP = BASE / "results", BASE / "results" / "tables", BASE / "figures", BASE / "app_data"
C = {"logreg": "#2a78d6", "cnn": "#eb6834", "gru": "#1baf7a", "true": "#2a78d6", "false": "#eb6834", "muted": "#a3a29c"}
MN = {"logreg": "요약특징+LogReg", "cnn": "1D-CNN", "gru": "GRU"}
BONES = [("Neck", "Nose"), ("Nose", "Left Eye"), ("Nose", "Right Eye"), ("Left Eye", "Left Ear"), ("Right Eye", "Right Ear"),
         ("Neck", "Left Shoulder"), ("Neck", "Right Shoulder"), ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
         ("Left Wrist", "Left Palm"), ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"), ("Right Wrist", "Right Palm"),
         ("Neck", "Back"), ("Back", "Waist"), ("Waist", "Left Hip"), ("Waist", "Right Hip"), ("Left Hip", "Left Knee"),
         ("Left Knee", "Left Ankle"), ("Left Ankle", "Left Foot"), ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"), ("Right Ankle", "Right Foot")]
JI = {j: i for i, j in enumerate(JOINTS)}

st.set_page_config(page_title="헬스 자세 오류 판별", page_icon="🏋️", layout="wide")

# ---------------- 데이터 로드 ----------------
@st.cache_data
def csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_models():
    return json.load(open(APP / "models.json", encoding="utf-8"))["models"]

@st.cache_data
def load_skeletons():
    z = np.load(APP / "mean_skeletons.npz")
    return {k: z[k].astype(np.float32) for k in z.files}

def show_fig(name, caption=None):
    p = FIG / name
    if p.exists(): st.image(str(p), caption=caption, width="stretch")

def fmt(df, cols=None, digits=3):
    cols = cols or [c for c in df.columns if df[c].dtype.kind == "f"]
    return df.style.format({c: f"{{:.{digits}f}}" for c in cols})

# ---------------- 골격 시각화 ----------------
def skeleton_fig(seqs, title="", height=520):
    """seqs: [(이름, 색, (16,24,3) 정규화 좌표)] -> 프레임 애니메이션 3D 골격. 화면 축: x=좌우, y=깊이(z), z=위아래(y)"""
    def traces(t):
        out = []
        for name, color, a in seqs:
            xs, ys, zs = [], [], []
            for b0, b1 in BONES:
                p, q = a[t, JI[b0]], a[t, JI[b1]]
                xs += [p[0], q[0], None]; ys += [p[2], q[2], None]; zs += [p[1], q[1], None]
            out.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=6), name=name, hoverinfo="skip"))
            out.append(go.Scatter3d(x=a[t, :, 0], y=a[t, :, 2], z=a[t, :, 1], mode="markers", marker=dict(size=3, color=color),
                                    text=JOINTS, hovertemplate="%{text}<extra>" + name + "</extra>", showlegend=False))
        return out
    allp = np.concatenate([a.reshape(-1, 3) for _, _, a in seqs])
    lo, hi = np.nanmin(allp, 0), np.nanmax(allp, 0); c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.05
    rng = lambda k: [c[k] - r, c[k] + r]
    fig = go.Figure(data=traces(0), frames=[go.Frame(data=traces(t), name=str(t)) for t in range(16)])
    fig.update_layout(
        title=title, height=height, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.02, x=0),
        scene=dict(xaxis=dict(range=rng(0), title="좌우"), yaxis=dict(range=rng(2), title="깊이"), zaxis=dict(range=rng(1), title="위아래"),
                   aspectmode="cube", camera=dict(eye=dict(x=0.2, y=-2.0, z=0.4), up=dict(x=0, y=0, z=1))),
        updatemenus=[dict(type="buttons", x=0, y=0, xanchor="left", yanchor="bottom", buttons=[
            dict(label="▶ 재생", method="animate", args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=True, mode="immediate")]),
            dict(label="❚❚ 정지", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.15, len=0.85, y=0, currentvalue=dict(prefix="프레임 "),
                      steps=[dict(label=str(t + 1), method="animate", args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for t in range(16)])])
    return fig

# ---------------- 예측 ----------------
def predict(model, X3):
    """X3: (1,16,24,3) 원좌표 -> 조건별 P(올바름)"""
    f = summary(np.concatenate([norm3d(X3), angles3d(X3.astype(np.float64)).astype(np.float32)], -1))[0]
    z = (f - np.array(model["mean"])) / np.array(model["scale"])
    return [1 / (1 + np.exp(-(z @ np.array(c["coef"]) + c["intercept"]))) for c in model["conditions"]]

def parse_3d(raw):
    fr = json.loads(raw)["frames"]
    if len(fr) != 16: raise ValueError(f"프레임이 {len(fr)}개입니다 (16개여야 함)")
    X = np.array([[[f["pts"][j]["x"], f["pts"][j]["y"], f["pts"][j]["z"]] for j in JOINTS] for f in fr], np.float64)[None]
    if not np.isfinite(X).all() or np.abs(X).max() > 1e4: raise ValueError("3D 좌표 이상값(|좌표|>10⁴) — 복원 오류 클립입니다")
    return X

# ---------------- 페이지 ----------------
def page_overview():
    st.title("🏋️ 헬스 운동 자세 오류의 조건별 판별 가능성")
    st.markdown("**3D 키포인트 기반 딥러닝과 수행자 분리 평가** — AI Hub 「피트니스 자세 이미지」 라벨 JSON (41종 운동)")
    c = st.columns(4)
    c[0].metric("사용 클립", "31,982"); c[1].metric("운동", "41종"); c[2].metric("운동×조건", "167개"); c[3].metric("수행자(Z코드)", "111명")
    st.markdown("""
#### 연구 질문
- **RQ1.** 운동·조건별로 3D 키포인트만으로 정/오류를 얼마나 판별할 수 있는가?
- **RQ2.** 요약 특징 + 선형 모델 대비 시퀀스 딥러닝(1D-CNN, GRU)은 어려운 조건에서 개선되는가?
- **RQ3.** 3D 좌표 대 2D 5시점 입력의 차이는?
- **RQ4.** 판별이 어려운 조건은 왜 어려운가?

#### 핵심 결과 (수행자 분리 8-fold)
1. 3D 좌표만으로 조건 평균 **balanced accuracy 0.792** (AUROC 0.858), 167개 중 89개 조건이 0.80 이상.
   **손목 중립(0.675)·이완 시 긴장 유지(0.657)** 가 가장 어렵다.
2. 1D-CNN·GRU(3시드 앙상블)는 LogReg와 평균이 비슷(0.841 / 0.837 vs 0.836). 큰 운동에서는 낫고 작은 운동에서는 못하며, **어려운 조건은 개선하지 못함**.
3. **손목 중립 = 표현의 문제**: 관절각 특징을 넣으면 0.675 → 0.761.
4. **긴장 유지·숄더패킹 = 관측의 문제**: 관절각 차이 2° 미만, 어떤 모델·입력으로도 뚜렷한 개선 없음.
5. 같은 클립에서 2D 5시점 결합(0.802) ≥ 3D(0.792) > 카메라 1대(0.755~0.783).
""")
    st.info("평가는 항상 **수행자(img_key의 Z코드) 단위로 분리**했습니다 — 같은 사람의 클립은 학습과 평가에 동시에 들어가지 않습니다.")

def page_data():
    st.header("📦 데이터와 정제")
    st.markdown("클립 = JSON 1개 = **16프레임** × (2D 5시점 또는 3D 1벌) × 관절 24개. 운동마다 조건 3~5개가 True(올바름)/False(오류)로 라벨링되어 있고, 오류는 수행자가 지시에 따라 **연기**한 것입니다.")
    cnt = json.load(open(APP / "counts.json", encoding="utf-8"))
    rows = [dict(데이터=k.replace(",", "·"), **{"2D JSON": v["json_2d"], "3D JSON": v["json_3d"], "2D만": v["only_2d"], "3D만(라벨 없음)": v["only_3d"],
                 "16프레임 아님": v["excluded"].get("2D 프레임 16 아님", 0), "3D 이상값": v["excluded"].get("3D 이상값", 0),
                 "사용 가능 클립": v["usable"], "수행자(Z)": v["performers"], "운동": v["exercises"]}) for k, v in cnt.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("""
**제외 규칙**: ① 16프레임이 아닌 클립 ② 라벨·Z코드 없음 ③ 3D 짝 없음(3D 실험) ④ **3D 이상값** — |좌표|>10⁴ 또는 정규화 특징 |f|>50 (맨몸 zip의 3D 복원 오류, 좌표 10¹²~10¹⁴).

**전처리**: 3D는 Neck 원점 이동 후 클립 평균 어깨너비로 나눔. 2D는 시점마다 Neck 원점 이동 후 몸통 길이(Neck–Waist)로 나눔 — 측면 시점에서는 어깨너비가 0에 가까워지기 때문.

**평가**: 운동마다 수행자(Z코드) 분리 GroupKFold(8). 모든 모델·입력이 같은 폴드. 주 지표는 balanced accuracy (라벨 불균형 운동 때문).
""")

def page_rq1():
    st.header("RQ1 · 운동·조건별 판별 성능 (41종, 3D 요약특징 + LogReg)")
    A = csv(RES / "all_logreg_3d_paired_metrics.csv"); B = csv(RES / "all_logreg_3d_ang_paired_metrics.csv")
    ex = A.groupby("exercise").agg(clips=("clips", "first"), performers=("performers", "first"), bal_acc=("bal_acc", "mean"),
                                   majority=("majority", "mean"), acc=("acc", "mean")).sort_values("bal_acc")
    fig = go.Figure()
    fig.add_bar(y=ex.index, x=ex["acc"], orientation="h", name="정확도", marker_color=C["logreg"],
                customdata=np.stack([ex["clips"], ex["performers"], ex["bal_acc"]], 1),
                hovertemplate="%{y}<br>정확도 %{x:.3f}<br>balanced acc %{customdata[2]:.3f}<br>클립 %{customdata[0]} · 수행자 %{customdata[1]}<extra></extra>")
    fig.add_scatter(y=ex.index, x=ex["majority"], mode="markers", name="다수 클래스 기준", marker=dict(symbol="line-ns-open", size=14, color="#0b0b0b", line=dict(width=3)))
    fig.update_layout(height=900, xaxis=dict(range=[0.4, 1], title="조건 평균 (수행자 분리 8-fold)"), margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.01, x=0), bargap=0.3)
    c1, c2 = st.columns([3, 2])
    with c1: st.plotly_chart(fig, width="stretch")
    with c2:
        e = st.selectbox("운동 선택", list(ex.index[::-1]))
        d = A[A.exercise == e][["condition", "pos_rate", "majority", "acc", "bal_acc", "auroc"]].merge(
            B[B.exercise == e][["condition", "bal_acc"]].rename(columns={"bal_acc": "bal_acc(+관절각)"}), on="condition")
        st.dataframe(fmt(d.rename(columns={"condition": "조건", "pos_rate": "올바름 비율"})), hide_index=True, width="stretch")
        st.caption("pos_rate = 올바름(True) 클립 비율, majority = 다수 클래스 기준 정확도")
        st.subheader("조건 유형별")
        t = csv(TAB / "condition_type_summary.csv").sort_values("bal_acc", ascending=False)
        st.dataframe(fmt(t.rename(columns={"type": "유형"})), hide_index=True, width="stretch")

def page_rq2():
    st.header("RQ2 · 시퀀스 딥러닝(1D-CNN, GRU) vs 요약특징 + LogReg (대표 6종)")
    st.markdown("랫풀 다운 · 딥스 · 바벨 스쿼트 · 오버 헤드 프레스 · 덤벨 컬 · 스텝 포워드 다이나믹 런지 (클립 5,577개, 조건 26개). "
                "입력 (16프레임, 72) · 운동별 다중 라벨 sigmoid · 같은 8폴드 · 학습 폴드 내 수행자 15%로 조기 종료.")
    sr = csv(TAB / "rep6_3d_seed_robustness.csv")
    m = sr.groupby("model")[["seed0", "seed1", "seed2"]].mean()
    W = csv(TAB / "rep6_3d_model_compare.csv")
    ens = {k: W[f"{k}_bal_acc"].mean() for k in ["logreg", "cnn", "gru"]}
    c = st.columns(3)
    c[0].metric("요약특징+LogReg", f"{ens['logreg']:.3f}")
    c[1].metric("1D-CNN (3시드 앙상블)", f"{ens['cnn']:.3f}", f"{ens['cnn'] - ens['logreg']:+.3f}")
    c[2].metric("GRU (3시드 앙상블)", f"{ens['gru']:.3f}", f"{ens['gru'] - ens['logreg']:+.3f}")
    st.markdown(f"**시드 견고성** — 단일 시드 CNN {m.loc['cnn'].min():.3f}~{m.loc['cnn'].max():.3f}, GRU {m.loc['gru'].min():.3f}~{m.loc['gru'].max():.3f} "
                f"(모두 LogReg {ens['logreg']:.3f}보다 낮음). 조건별 시드 간 표준편차 평균 {sr['sd'].mean():.3f} → **시드 1개로는 결론을 낼 수 없음**.")
    W = W.copy(); W["label"] = np.where(W["hard"], "★ ", "") + W["exercise"] + " · " + W["condition"]
    fig = go.Figure()
    for k in ["logreg", "cnn", "gru"]:
        fig.add_scatter(x=W[f"{k}_bal_acc"], y=W["label"], mode="markers", name=MN[k], marker=dict(size=10, color=C[k], line=dict(color="white", width=1)),
                        hovertemplate="%{y}<br>" + MN[k] + " %{x:.3f}<extra></extra>")
    fig.update_layout(height=780, xaxis=dict(title="balanced accuracy", range=[0.5, 1]), yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=1.02, x=0), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("★ = 어려운 조건(손목 중립·이완 시 긴장 유지·숄더패킹). CNN·GRU는 3시드 앙상블.")
    st.subheader("조건별 비교와 95% 신뢰구간 (수행자 부트스트랩 1000회, LogReg 대비)")
    cols = ["exercise", "condition", "majority", "logreg_bal_acc", "cnn_bal_acc", "gru_bal_acc", "cnn-logreg_CI", "gru-logreg_CI"]
    st.dataframe(fmt(W[cols]), hide_index=True, width="stretch")
    st.subheader("입력 표현을 바꾸면: 3D 좌표 vs 3D 좌표 + 관절각 (시드 0)")
    st.dataframe(fmt(csv(TAB / "rep6_3d_vs_3d_ang_mean.csv").rename(columns={"hard": "group", "Unnamed: 0": "group"})), hide_index=True, width="stretch")
    st.success("어려운 조건의 개선은 **모델 구조가 아니라 입력 표현(관절각)** 에서 나왔고, 그것도 손목처럼 정보가 키포인트에 있는 조건에 한정됩니다.")

def page_rq3():
    st.header("RQ3 · 3D 좌표 vs 2D 5시점 vs 카메라 1대")
    o = csv(TAB / "all41_2d_vs_3d_overall.csv")
    st.dataframe(fmt(o), hide_index=True, width="stretch")
    v = csv(TAB / "all41_single_view_by_type.csv").rename(columns={"Unnamed: 0": "type"})
    v = v.rename(columns={v.columns[0]: "유형"})
    cols = ["3D", "2D 5시점", "2D 시점A", "2D 시점B", "2D 시점C", "2D 시점D", "2D 시점E"]
    fig = go.Figure()
    order = v[v["유형"] != "전체"].sort_values("3D")
    for k, col in zip(cols, ["#2a78d6", "#eb6834", "#c3c2b7", "#a3a29c", "#86857f", "#6b6a65", "#52514e"]):
        fig.add_bar(y=order["유형"], x=order[k], name=k, orientation="h", marker_color=col, hovertemplate="%{y}<br>" + k + " %{x:.3f}<extra></extra>")
    fig.update_layout(barmode="group", height=720, xaxis=dict(range=[0.55, 0.9], title="balanced accuracy (LogReg, 41종, 같은 클립)"),
                      legend=dict(orientation="h", y=1.02, x=0), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fmt(v), hide_index=True, width="stretch")
    st.markdown("- 같은 클립에서 **2D 5시점 결합이 3D보다 약간 높음** — 3D는 다시점 2D에서 복원된 값으로 보이며 복원 오차가 더해진 것으로 해석(생성 방식은 문서 미기재).\n"
                "- **카메라 1대**로는 0.02~0.05 하락, 손목·긴장 유지 조건에서 더 크게 하락. 시점 A~E의 카메라 위치는 문서에 없음.")
    show_fig("fig4_2d_vs_3d_rep6.png", "대표 6종: 조건별 3D vs 2D 5시점 (모델별)")

def page_rq4():
    st.header("RQ4 · 어려운 조건은 왜 어려운가")
    st.markdown("조건 정의에 대응하는 **기하 특징 1개**(학습 없음)로 정/오류가 갈리는지 AUROC로 확인했습니다.")
    g = csv(TAB / "geometric_feature_auroc.csv")
    st.dataframe(fmt(g.rename(columns={"single_feature_auroc": "기하특징 1개 AUROC", "logreg_auroc": "LogReg(216) AUROC"})), hide_index=True, width="stretch")
    show_fig("fig7_geometric_features_dumbbell_curl.png")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 표현의 문제 — 손목 중립\n손목 꺾임 각 하나가 216특징 LogReg보다 잘 가름(9개 중 8개). **관절각을 특징에 넣으면 0.675 → 0.761.**")
    with c2:
        st.markdown("#### 관측의 문제 — 긴장 유지·숄더패킹\n팔꿈치 최대 각 차이 0.5~1.7° (AUROC 0.52~0.58). 근육 긴장·견갑골은 24관절로 직접 보이지 않음.")
    t = csv(TAB / "all41_3d_vs_3d_ang_by_type.csv").rename(columns={"Unnamed: 0": "type"})
    st.dataframe(fmt(t.rename(columns={t.columns[0]: "유형"})), hide_index=True, width="stretch")
    show_fig("fig5_joint_importance.png")

    st.subheader("조건별 평균 동작 — 올바름 vs 오류 (전체 클립 평균, 정규화 좌표)")
    models, sk = load_models(), load_skeletons()
    c1, c2 = st.columns([1, 2])
    with c1:
        e = st.selectbox("운동", sorted(models), index=sorted(models).index("덤벨 컬"))
        cn = st.selectbox("조건", [c["name"] for c in models[e]["conditions"]])
        info = next(c for c in models[e]["conditions"] if c["name"] == cn)
        st.metric("교차검증 balanced acc (3D+관절각)", f"{info['cv_bal_acc']:.3f}")
        st.caption(f"다수 클래스 기준 정확도 {info['cv_majority']:.3f} · 수행자 분리 8-fold")
        st.caption("평균 골격은 같은 라벨 클립들의 정규화 좌표 평균입니다. 개별 수행자의 움직임이 아니며, 평균이라 동작이 부드럽게 뭉개집니다.")
    with c2:
        seqs = [(f"올바름 (n≈{info['pos_rate'] * models[e]['clips']:.0f})", C["true"], sk.get(f"{e}|{cn}|1")),
                (f"오류 (n≈{(1 - info['pos_rate']) * models[e]['clips']:.0f})", C["false"], sk.get(f"{e}|{cn}|0"))]
        st.plotly_chart(skeleton_fig([s for s in seqs if s[2] is not None], f"{e} · {cn}"), width="stretch")

def page_demo():
    st.header("🎯 라이브 데모 — 클립 JSON 업로드 → 조건별 판별")
    st.markdown("AI Hub 라벨의 **3D 파일(`D*-*-*-3d.json`)** 을 올리면 운동별 LogReg(3D 좌표+관절각, 41종 전체 학습)로 조건마다 P(올바름)를 계산합니다. "
                "같은 이름의 **2D 파일(`D*-*-*.json`)** 도 함께 올리면 운동명을 자동 인식하고 정답 라벨과 비교합니다.")
    st.caption("업로드한 파일은 이 세션의 메모리에서만 처리되며 저장되지 않습니다. 원본 데이터는 저장소에 포함되어 있지 않습니다.")
    models = load_models()
    c1, c2 = st.columns(2)
    f3 = c1.file_uploader("3D JSON (필수)", type="json", key="f3")
    f2 = c2.file_uploader("2D JSON (선택: 운동명·정답 라벨)", type="json", key="f2")
    truth, ex_auto = None, None
    if f2 is not None:
        try:
            ti = json.loads(f2.getvalue())["type_info"]; ex_auto = ti["exercise"].strip()
            truth = {c["condition"].strip(): bool(c["value"]) for c in ti["conditions"]}
        except Exception as e:
            st.warning(f"2D 파일에서 라벨을 읽지 못했습니다: {e}")
    names = sorted(models)
    e = st.selectbox("운동", names, index=names.index(ex_auto) if ex_auto in names else 0, disabled=ex_auto in names)
    if f3 is None:
        st.info("3D JSON을 업로드하세요."); return
    try:
        X = parse_3d(f3.getvalue())
    except Exception as err:
        st.error(f"3D 파일 오류: {err}"); return
    m = models[e]; P = predict(m, X)
    rows = []
    for c, p in zip(m["conditions"], P):
        r = {"조건": c["name"], "P(올바름)": p, "예측": "✅ 올바름" if p >= 0.5 else "❌ 오류", "교차검증 bal_acc": c["cv_bal_acc"]}
        if truth is not None:
            t = truth.get(c["name"]); r["정답"] = "-" if t is None else ("✅ 올바름" if t else "❌ 오류"); r["일치"] = "" if t is None else ("○" if (p >= 0.5) == t else "✕")
        rows.append(r)
    R = pd.DataFrame(rows)
    c1, c2 = st.columns([2, 3])
    with c1:
        fig = go.Figure(go.Bar(x=R["P(올바름)"], y=R["조건"], orientation="h", marker_color=[C["true"] if p >= .5 else C["false"] for p in R["P(올바름)"]],
                               hovertemplate="%{y}<br>P(올바름) %{x:.2f}<extra></extra>"))
        fig.add_vline(x=0.5, line_dash="dash", line_color=C["muted"])
        fig.update_layout(height=80 + 60 * len(R), xaxis=dict(range=[0, 1], title="P(올바름)"), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(fmt(R, ["P(올바름)", "교차검증 bal_acc"], 2), hide_index=True, width="stretch")
        if truth is not None and "일치" in R:
            st.metric("조건 일치", f"{(R['일치'] == '○').sum()} / {(R['일치'] != '').sum()}")
        st.caption("주의: 데모 모델은 41종 전체 클립으로 학습했으므로, AI Hub 학습 데이터 클립을 올리면 '본 적 있는' 데이터입니다. 일반화 성능은 '교차검증 bal_acc' 열(수행자 분리)을 보세요.")
    with c2:
        st.plotly_chart(skeleton_fig([("업로드 클립", "#1c5cab", norm3d(X.astype(np.float32)).reshape(16, 24, 3))], f3.name), width="stretch")

PAGES = {"🏠 개요": page_overview, "📦 데이터": page_data, "RQ1 · 조건별 성능": page_rq1, "RQ2 · 딥러닝 비교": page_rq2,
         "RQ3 · 2D vs 3D": page_rq3, "RQ4 · 어려운 조건": page_rq4, "🎯 라이브 데모": page_demo}
with st.sidebar:
    st.markdown("### 헬스 자세 오류 판별")
    page = st.radio("페이지", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.caption("데이터: AI Hub 피트니스 자세 이미지 (dataSetSn=231) 라벨 JSON · 평가: 수행자 분리 GroupKFold(8)")
PAGES[page]()
