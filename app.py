"""헬스 운동 자세 오류의 조건별 판별 가능성 — Streamlit 발표·시연 앱.

실행:  streamlit run app.py
원본 AI Hub 데이터 없이 동작한다. 사용하는 파일:
  app_data/        export_models.py 가 만든 모델 계수·평균 골격·데이터 집계
  results/         실험 결과 CSV (exp_seq.py, analyze_results.py)
  figures/         analyze_results.py 가 만든 그림
라이브 데모는 사용자가 업로드한 JSON 을 메모리에서만 처리한다(저장하지 않음).
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from build_cache import JOINTS, BONES
from exp_seq import norm3d, angles3d, summary

BASE = Path(__file__).parent
RES, TAB, FIG, APP = BASE / "results", BASE / "results" / "tables", BASE / "figures", BASE / "app_data"
C = {"logreg": "#2a78d6", "cnn": "#eb6834", "gru": "#1baf7a", "true": "#2a78d6", "false": "#eb6834", "muted": "#a3a29c"}
MN = {"logreg": "요약특징+LogReg", "cnn": "1D-CNN", "gru": "GRU"}
JI = {j: i for i, j in enumerate(JOINTS)}

st.set_page_config(page_title="헬스 자세 오류 판별", page_icon="🏋️", layout="wide")

# ---------------- 데이터 로드 ----------------
@st.cache_data
def csv(path):
    return pd.read_csv(path)

# 캐시 키에 파일 수정 시각을 넣어, 모델 파일을 바꾸면 서버를 재시작하지 않아도 새로 읽는다
@st.cache_data
def _json(path, mtime):
    return json.load(open(path, encoding="utf-8"))["models"]

def load_models():
    p = APP / "models.json"; return _json(str(p), p.stat().st_mtime)

@st.cache_data
def _npz(path, mtime):
    z = np.load(path); return {k: z[k].astype(np.float32) for k in z.files}

def load_skeletons():
    p = APP / "mean_skeletons.npz"; return _npz(str(p), p.stat().st_mtime)

def show_fig(name, caption=None):
    p = FIG / name
    if p.exists(): st.image(str(p), caption=caption, width="stretch")

def fmt(df, cols=None, digits=3):
    cols = cols or [c for c in df.columns if df[c].dtype.kind == "f"]
    return df.style.format({c: f"{{:.{digits}f}}" for c in cols})

# 판별 결과처럼 꼭 보여야 하는 것은 markdown·내장 이미지로 그린다: st.dataframe·st.image 는 따로 받는 부품/파일이라
# Cloud 연결이 불안정할 때(streamlit.app 정적 파일 요청이 간헐적으로 멈춤) 회색 상자로 남는다. markdown 은 기본 번들에 포함.
def md_table(df, digits=2):
    """DataFrame → markdown 표 문자열 (실수는 digits 자리, NaN 은 —)"""
    cell = lambda v: "—" if isinstance(v, float) and np.isnan(v) else f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))

def img_html(rgb, caption=""):
    """RGB 배열 → 페이지에 직접 넣는 <img> (base64 JPEG, 추가 요청 없음)"""
    import base64, io
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    cap = f"<div style='font-size:0.8rem;color:#808495;text-align:center'>{caption}</div>" if caption else ""
    return f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}' style='width:100%;border-radius:6px'/>{cap}"

# ---------------- 골격 시각화 ----------------
VIEWS = {"정면": (0.2, -2.0, 0.4), "측면": (2.0, 0.2, 0.4), "위": (0.1, -0.4, 2.1), "대각선": (1.4, -1.4, 0.9)}

SKELETON_JS = """
const init = () => {
const gd = document.getElementById('{plot_id}');
const F = __FRAMES__, VIEWS = __VIEWS__, N = F.length;
const up = {x: 0, y: 0, z: 1}, center = {x: 0, y: 0, z: 0};
let t = 0, timer = null, rot = false, ang = -Math.PI / 2;
const slider = document.getElementById('{plot_id}-slider'), lab = document.getElementById('{plot_id}-label');
const playBtn = document.getElementById('{plot_id}-play'), rotBtn = document.getElementById('{plot_id}-rot');
// Plotly 기본 animate 는 매 프레임 레이아웃 카메라로 되돌리므로 직접 재생한다.
// 좌표만 바꾸는 restyle 은 레이아웃 카메라로 다시 그리므로, 사용자가 돌린 카메라(드래그를 끝낼 때 오는
// plotly_relayout 이벤트, 시점 버튼 포함)를 레이아웃에 반영해 둔다. (react 는 프레임마다 장면을 새로 만들어
// 재생 중 드래그가 끊기므로 쓰지 않음)
let cam = JSON.parse(JSON.stringify(gd.layout.scene.camera));
gd.on('plotly_relayout', e => { if (e['scene.camera']) cam = e['scene.camera']; });
function show(k) {
  t = k;
  if (rot) { ang += Math.PI / 16; cam = {eye: {x: 2.1 * Math.cos(ang), y: 2.1 * Math.sin(ang), z: 0.5}, up, center}; }
  gd.layout.scene.camera = cam;
  Plotly.restyle(gd, {x: F[k].map(d => d.x), y: F[k].map(d => d.y), z: F[k].map(d => d.z)});
  slider.value = k + 1; lab.textContent = '프레임 ' + (k + 1) + ' / ' + N;
}
// 드래그·휠 조작 중에는 프레임을 넘기지 않는다: 다시 그리기가 진행 중인 드래그를 끊으면 시점이 저장되지 않으므로.
// 손을 떼면(mouseup → relayout 이벤트로 카메라 저장) 새 시점에서 이어서 재생된다.
let holding = false, wheelAt = 0, release = null;
gd.addEventListener('pointerdown', () => { holding = true; clearTimeout(release); }, true);
// 손을 뗀 뒤 Plotly 가 카메라를 저장(plotly_relayout)하면 바로, 오지 않으면 0.5초 뒤 재개
window.addEventListener('pointerup', () => { if (holding) release = setTimeout(() => { holding = false; }, 500); }, true);
gd.on('plotly_relayout', e => { if (e['scene.camera'] && holding) { clearTimeout(release); holding = false; } });
gd.addEventListener('wheel', () => { wheelAt = Date.now(); }, {capture: true, passive: true});
function play() {
  if (timer) return;
  timer = setInterval(() => { if (!holding && Date.now() - wheelAt > 300) show((t + 1) % N); }, 180);
  playBtn.textContent = '❚❚ 정지';
}
function stop() { clearInterval(timer); timer = null; playBtn.textContent = '▶ 재생'; }
playBtn.onclick = () => timer ? stop() : play();
rotBtn.onclick = () => { rot = !rot; rotBtn.classList.toggle('on', rot); if (rot) play(); };
slider.oninput = () => { stop(); show(+slider.value - 1); };
document.querySelectorAll('[data-view="{plot_id}"]').forEach(b => b.onclick = () => {
  rot = false; rotBtn.classList.remove('on'); const [x, y, z] = VIEWS[b.dataset.name];
  Plotly.relayout(gd, {'scene.camera': {eye: {x, y, z}, up, center}});
});
};
// 재생 컨트롤은 그래프 아래에 있으므로 문서를 다 읽은 뒤 연결
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
"""

def skeleton_html(seqs, title="", height=520):
    """seqs: [(이름, 색, (16,24,3) 정규화 좌표)] -> 재생 중에도 시점을 바꿀 수 있는 3D 골격 애니메이션 HTML.
    화면 축: x=좌우, y=깊이(z), z=위아래(y). 시점 버튼(정면/측면/위/대각선), 회전하며 재생, 드래그 회전을 지원한다."""
    def traces(t):
        out = []
        for name, color, a in seqs:
            xs, ys, zs = [], [], []
            for b0, b1 in BONES:
                p, q = a[t, JI[b0]], a[t, JI[b1]]
                xs += [float(p[0]), float(q[0]), None]; ys += [float(p[2]), float(q[2]), None]; zs += [float(p[1]), float(q[1]), None]
            out.append(dict(x=xs, y=ys, z=zs))
            out.append(dict(x=a[t, :, 0].tolist(), y=a[t, :, 2].tolist(), z=a[t, :, 1].tolist()))
        return out
    t0 = traces(0); fig = go.Figure()
    for (name, color, _), lines, pts in zip(seqs, t0[::2], t0[1::2]):
        fig.add_scatter3d(**lines, mode="lines", line=dict(color=color, width=6), name=name, hoverinfo="skip")
        fig.add_scatter3d(**pts, mode="markers", marker=dict(size=3, color=color), text=JOINTS,
                          hovertemplate="%{text}<extra>" + name + "</extra>", showlegend=False)
    allp = np.concatenate([a.reshape(-1, 3) for _, _, a in seqs])
    lo, hi = np.nanmin(allp, 0), np.nanmax(allp, 0); c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.05
    rng = lambda k: [float(c[k] - r), float(c[k] + r)]
    x0, y0, z0 = VIEWS["정면"]
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.02, x=0), uirevision="keep",
                      font=dict(family="Source Sans Pro, sans-serif"), paper_bgcolor="white",
                      scene=dict(xaxis=dict(range=rng(0), title="좌우"), yaxis=dict(range=rng(2), title="깊이"), zaxis=dict(range=rng(1), title="위아래"),
                                 aspectmode="cube", camera=dict(eye=dict(x=x0, y=y0, z=z0), up=dict(x=0, y=0, z=1))))
    js = SKELETON_JS.replace("__FRAMES__", json.dumps([traces(t) for t in range(16)])).replace("__VIEWS__", json.dumps(VIEWS, ensure_ascii=False))
    pid = "skel"
    buttons = "".join(f'<button data-view="{pid}" data-name="{v}">{v}</button>' for v in VIEWS)
    controls = f"""
<style>
  .sk {{font-family: 'Source Sans Pro', sans-serif; font-size: 14px; color: #31333f; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0}}
  .sk button {{border: 1px solid #d6d5d0; background: #fff; border-radius: 6px; padding: 4px 10px; cursor: pointer; font: inherit}}
  .sk button:hover {{border-color: #2a78d6; color: #2a78d6}}
  .sk button.on {{background: #2a78d6; border-color: #2a78d6; color: #fff}}
  .sk input[type=range] {{flex: 1; min-width: 120px; accent-color: #2a78d6}}
</style>
<div class="sk"><span>시점</span>{buttons}<button id="{pid}-rot">🔄 회전하며 재생</button></div>"""
    player = f"""<div class="sk"><button id="{pid}-play">▶ 재생</button><input id="{pid}-slider" type="range" min="1" max="16" value="1">
<span id="{pid}-label">프레임 1 / 16</span></div>"""
    return controls + fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=pid, config={"displaylogo": False},
                                  post_script=js.replace("{plot_id}", pid)) + player

# ---------------- 예측 ----------------
def zscore(model, X3):
    """X3: (1,16,24,3) 원좌표 -> 학습 분포 기준 표준화 특징 (267,)"""
    f = summary(np.concatenate([norm3d(X3), angles3d(X3.astype(np.float64)).astype(np.float32)], -1))[0]
    return (f - np.array(model["mean"])) / np.array(model["scale"])

def predict(model, X3, clip=None):
    """X3: (1,16,24,3) 원좌표 -> 조건별 P(올바름). clip: 표준화 값을 ±clip 으로 잘라 학습 범위 밖 극단값의 영향을 제한(영상용)"""
    z = zscore(model, X3)
    if clip: z = np.clip(z, -clip, clip)
    return [1 / (1 + np.exp(-(z @ np.array(c["coef"]) + c["intercept"]))) for c in model["conditions"]]

KO_JOINT = {"Nose": "코", "Left Eye": "왼눈", "Right Eye": "오른눈", "Left Ear": "왼귀", "Right Ear": "오른귀", "Left Shoulder": "왼어깨",
            "Right Shoulder": "오른어깨", "Left Elbow": "왼팔꿈치", "Right Elbow": "오른팔꿈치", "Left Wrist": "왼손목", "Right Wrist": "오른손목",
            "Left Hip": "왼골반", "Right Hip": "오른골반", "Left Knee": "왼무릎", "Right Knee": "오른무릎", "Left Ankle": "왼발목",
            "Right Ankle": "오른발목", "Neck": "목", "Left Palm": "왼손", "Right Palm": "오른손", "Back": "등", "Waist": "허리",
            "Left Foot": "왼발", "Right Foot": "오른발"}

def feature_names():
    """summary 특징 267개의 사람이 읽는 이름 (평균·흔들림·움직임 범위 × [관절 좌표 72 + 관절각 15 + 귀-어깨 2])"""
    from exp_seq import ANGLES
    ax = {"x": "좌우 위치", "y": "높이", "z": "앞뒤 위치"}
    base = [f"{KO_JOINT[j]} {ax[a]}" for j in JOINTS for a in "xyz"]
    base += [f"{KO_JOINT[b]} 각도({KO_JOINT[a]}-{KO_JOINT[b]}-{KO_JOINT[c]})" for a, b, c in ANGLES] + ["왼귀-어깨 거리", "오른귀-어깨 거리"]
    return [f"{n}의 {s}" for s in ["평균", "흔들림", "움직임 범위"] for n in base]

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
    with c1: st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
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
        components.html(skeleton_html([s for s in seqs if s[2] is not None], f"{e} · {cn}"), height=640)
        st.caption("▶ 재생 중에도 그래프를 마우스로 드래그하면 시점을 돌릴 수 있습니다(휠: 확대/축소). '회전하며 재생'은 카메라가 몸 주위를 자동으로 돕니다.")

def page_demo():
    st.header("🎯 라이브 데모 — 클립 JSON 업로드 → 조건별 판별")
    st.markdown("AI Hub 라벨의 **3D 파일(`D*-*-*-3d.json`)** 을 올리면 운동별 LogReg(3D 좌표+관절각)로 조건마다 P(올바름)를 계산합니다. "
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
        st.plotly_chart(fig, use_container_width=True)
        if truth is not None and "일치" in R:
            st.metric("조건 일치", f"{(R['일치'] == '○').sum()} / {(R['일치'] != '').sum()}")
        z = m.get("holdout_performer")
        if z:
            st.caption(f"이 운동의 데모 모델은 **수행자 {z}를 학습에서 제외**했습니다. {z}의 클립(예시 파일)은 처음 보는 사람의 자세입니다. "
                       "다른 수행자의 클립은 학습에 쓰인 데이터이므로, 일반화 성능은 '교차검증 bal_acc' 열(수행자 분리)을 보세요.")
        else:
            st.caption("주의: 이 운동의 데모 모델은 전체 클립으로 학습했으므로, AI Hub 클립을 올리면 '본 적 있는' 데이터입니다. "
                       "일반화 성능은 '교차검증 bal_acc' 열(수행자 분리)을 보세요.")
    with c2:
        components.html(skeleton_html([("업로드 클립", "#1c5cab", norm3d(X.astype(np.float32)).reshape(16, 24, 3))], f3.name), height=640)
    st.markdown(md_table(R))

@st.cache_data(max_entries=3, show_spinner=False)
def analyze_video(data):
    """영상 바이트 → MediaPipe world landmarks (T,33,3), 2D (T,33,2), fps, 확인용 썸네일. 같은 영상은 다시 계산하지 않음"""
    import video_pose as vp
    frames, fps = vp.read_video(data)
    W, V, P2 = vp.detect(frames, fps)
    thumbs = [vp.thumbnail(f, p) for f, p in zip(frames, P2)]
    return W, V, fps, thumbs

def page_video():
    st.header("🎬 영상 판별 (실험적) — 운동 영상 업로드 → 조건별 판별")
    st.warning("""**실험적 기능입니다.** 영상에서 MediaPipe 로 3D 관절을 추정해 AI Hub 24관절로 바꾼 뒤 같은 모델로 판별합니다.
모델은 AI Hub 의 **5대 카메라로 복원한 3D 좌표**로 학습했고, 영상은 **카메라 1대로 추정한 3D** 라 좌표의 성질이 다릅니다.
관절 정의는 AI Hub 데이터로 측정한 위치에 맞췄고, 영상에서 볼 수 없는 척추 점(Neck·Back·Waist)에 기대지 않도록 **영상용 모델을 따로 학습**했습니다.
표의 교차검증 bal_acc 는 그 모델을 AI Hub 데이터로 평가한 값이며, **MediaPipe 의 관절 추정 오차는 반영되지 않았습니다**(정답 라벨이 있는 영상이 없어 검증 불가).""")
    with st.expander("촬영 팁 (정확도에 큰 영향)", expanded=False):
        st.markdown("""- **서서 하는 운동** 권장 (덤벨 컬·바벨 컬·스쿼트·런지·오버 헤드 프레스·사이드 레터럴 레이즈 등). 누워서 하는 운동은 AI Hub 의 몸 방향이 일정하지 않아 정렬이 불안정합니다.
- **전신이 화면에 다 나오게**, 한 사람만, 카메라는 **수평으로 고정** (MediaPipe 의 '위쪽'은 카메라 기준이라 기울면 몸이 기운 것으로 계산됩니다).
- 정면보다 **비스듬한 각도(약 45°)** 가 깊이 추정에 유리합니다. 영상은 앞 8초까지만 분석합니다(서버 메모리 제한).
- 아래 구간 슬라이더로 **동작 1회**(AI Hub 클립은 원본 31장 중 16장, 약 1~2초 분량으로 추정)를 고르세요.""")
    models = load_models()
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("운동 영상 (mp4 / mov / avi)", type=["mp4", "mov", "avi", "m4v"], key="vid")
    names = sorted(models)
    e = c2.selectbox("어떤 운동인가요?", names, index=names.index("덤벨 컬"), key="vid_ex")
    if up is None:
        st.info("영상을 업로드하세요."); return
    try:
        import mediapipe  # noqa: F401  설치 여부 확인
        import video_pose as vp
        assert os.path.exists(vp.MODEL), f"포즈 모델 파일이 없습니다: {vp.MODEL}"
    except Exception as err:
        st.error("이 서버에서 영상 분석(MediaPipe)을 불러오지 못했습니다. requirements.txt 의 mediapipe, packages.txt(libgl1·libglib2.0-0·libportaudio2), "
                 f"models/pose_landmarker_full.task 가 배포에 포함됐는지 확인하세요. 로컬 실행(`streamlit run app.py`)에서는 동작합니다.\n\n세부: {err}"); return
    with st.spinner("관절 검출 중… (영상 길이에 따라 수십 초)"):
        try:
            W, V, fps, thumbs = analyze_video(up.getvalue())
        except Exception as err:
            st.error(f"영상 처리 실패: {err}"); return
    import video_pose as vp
    T = len(W); found = np.isfinite(W).all((1, 2))
    st.caption(f"분석 프레임 {T}장 ({T / fps:.1f}초, {fps:.0f}fps) · 사람 검출 {found.mean():.0%}")
    if found.sum() < 8:
        st.error("사람이 거의 검출되지 않았습니다. 전신이 보이게 다시 찍어 주세요."); return
    t0, t1 = st.slider("분석 구간 (초) — 동작 1회", 0.0, round(T / fps, 1), (0.0, round(T / fps, 1)), step=0.1, key="vid_seg")
    try:
        idx = vp.sample16(W, int(t0 * fps), min(T - 1, int(t1 * fps)))
    except ValueError as err:
        st.error(str(err)); return
    m = models[e]; mv = m["video"]   # 영상용 모델: AI Hub 데이터를 영상으로 관측 가능한 관절만으로 재구성해 학습
    X = vp.align_yaw(vp.to_aihub(W[idx]), m["yaw_mean"])[None]
    vis = V[idx][:, vp.KEY_MP].mean()
    P = predict(mv, X, clip=3)
    if vis < 0.6: st.warning(f"주요 관절 신뢰도가 낮습니다({vis:.2f}). 가려진 관절이 있으면 결과를 믿기 어렵습니다.")
    # 학습 분포에서 벗어난 정도: |z|>3 특징이 많으면 결과가 그 차이에 끌려갈 수 있음 (Firefly 생성 영상: 23개 → 판정이 발 간격·고개 방향에 좌우됨)
    z = zscore(mv, X); far = np.abs(z) > 3; names_f = feature_names()
    top = [i for i in np.argsort(-np.abs(z)) if far[i]][:5]
    # 기준 근거: 학습에서 뺀 AI Hub 수행자 128클립에서 범위 밖 특징 0~2개 → 조건 일치율 0.92, 3~9개 → 0.86, 10~19개 → 0.69
    detail = ", ".join(f"{names_f[i]} ({z[i]:+.1f}σ)" for i in top)
    if far.sum() >= 10:
        st.error(f"**이 영상은 학습 데이터(AI Hub)와 자세·촬영 조건이 크게 다릅니다** — 267개 특징 중 **{far.sum()}개**가 학습 범위(±3 표준편차)를 벗어났습니다. "
                 "아래 판정은 이 차이에 끌려갔을 수 있어 **신뢰하기 어렵습니다** (학습에 쓰지 않은 AI Hub 클립에서도 이 개수가 10개 이상이면 "
                 f"조건 일치율이 0.92 → 0.69 로 떨어졌습니다).\n\n크게 다른 부분: {detail}")
    elif far.sum() >= 3:
        st.info(f"학습 범위를 벗어난 특징 {far.sum()}개 (AI Hub 검증 클립 기준 3~9개일 때 조건 일치율 약 0.86): {detail}")
    R = pd.DataFrame([{"조건": c["name"], "P(올바름)": p, "예측": "✅ 올바름" if p >= 0.5 else "❌ 오류", "교차검증 bal_acc": c["cv_bal_acc"]}
                      for c, p in zip(mv["conditions"], P)])
    c1, c2 = st.columns([2, 3])
    with c1:
        fig = go.Figure(go.Bar(x=R["P(올바름)"], y=R["조건"], orientation="h", marker_color=[C["true"] if p >= .5 else C["false"] for p in R["P(올바름)"]],
                               hovertemplate="%{y}<br>P(올바름) %{x:.2f}<extra></extra>"))
        fig.add_vline(x=0.5, line_dash="dash", line_color=C["muted"])
        fig.update_layout(height=80 + 60 * len(R), xaxis=dict(range=[0, 1], title="P(올바름)"), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"영상용 모델: 척추 점(Neck·Back·Waist)처럼 영상에서 직접 볼 수 없는 관절을 영상과 같은 공식으로 재구성한 AI Hub 좌표로 학습 "
                   f"(교차검증 bal_acc 는 이 모델 기준). 주요 관절 평균 신뢰도 {vis:.2f} · 골반 방향을 AI Hub {e} 평균({m['yaw_mean']:.0f}°, "
                   f"표준편차 {m['yaw_sd']:.0f}°)으로 정렬")
    with c2:
        components.html(skeleton_html([("영상에서 추정한 골격", "#1c5cab", norm3d(X.astype(np.float32)).reshape(16, 24, 3))], f"{e} · {up.name}"), height=640)
    st.markdown(md_table(R))
    st.markdown("**검출 확인** — 분석에 쓴 16프레임 중 4장 (파란 선이 몸과 맞지 않으면 결과를 믿기 어렵습니다)")
    cols = st.columns(4)
    for col, k in zip(cols, [0, 5, 10, 15]):
        col.markdown(img_html(thumbs[idx[k]], f"{idx[k] / fps:.1f}초"), unsafe_allow_html=True)

def load_photo_models():
    p = APP / "photo_models.json"; return _json(str(p), p.stat().st_mtime)

@st.cache_data(max_entries=5, show_spinner=False)
def analyze_photo(data):
    import video_pose as vp
    img, W, V, P2 = vp.detect_image(data)
    return W, V, vp.thumbnail(img, P2, 480)

# 사진 판별 기준 (photo_models.py 결과로 정함): 한 프레임 교차검증 bal_acc 와 학습 범위 밖 특징 수
PHOTO_OK, PHOTO_MAYBE = 0.75, 0.65
PHOTO_FAR_WARN = 3   # 학습에서 뺀 수행자 프레임: 범위 밖 0개 → 일치율 0.76, 1~2개 0.74, 3~5개 0.68, 6~10개 0.63

def page_photo():
    st.header("📷 사진 판별 (실험적) — 운동 사진 한 장 → 자세 조건 판별")
    st.warning("""**실험적 기능입니다.** 사진 한 장에는 '움직임'이 없어서 **반동·팔꿈치 고정·이완 시 긴장 유지처럼 동작으로 정의되는 조건은 판별할 수 없습니다.**
AI Hub 클립의 프레임을 한 장씩 학습한 모델로, 조건마다 **사진으로 판별 가능한 정도(수행자 분리 교차검증)** 를 함께 보여 줍니다.
영상 판별과 같이 MediaPipe 로 관절을 추정하므로 그 오차는 반영되지 않았습니다.""")
    with st.expander("촬영 팁", expanded=False):
        st.markdown("- 전신이 다 나오게, 한 사람만, 카메라 수평 · 서서 하는 운동 권장 · 판별하고 싶은 **자세의 순간**(예: 스쿼트 가장 낮은 지점)을 찍으세요.")
    pm, models = load_photo_models(), load_models()
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("운동 사진 (jpg / png)", type=["jpg", "jpeg", "png", "webp"], key="photo")
    names = sorted(pm); e = c2.selectbox("어떤 운동인가요?", names, index=names.index("덤벨 컬"), key="photo_ex")
    if up is None:
        st.info("사진을 업로드하세요."); return
    try:
        import mediapipe  # noqa: F401
        import video_pose as vp
        assert os.path.exists(vp.MODEL), f"포즈 모델 파일이 없습니다: {vp.MODEL}"
    except Exception as err:
        st.error("이 서버에서 사진 분석(MediaPipe)을 불러오지 못했습니다. requirements.txt 의 mediapipe, packages.txt, models/pose_landmarker_full.task "
                 f"가 배포에 포함됐는지 확인하세요. 로컬 실행에서는 동작합니다.\n\n세부: {err}"); return
    with st.spinner("관절 검출 중…"):
        try:
            W, V, thumb = analyze_photo(up.getvalue())
        except Exception as err:
            st.error(f"사진 처리 실패: {err}"); return
    if W is None:
        st.error("사람이 검출되지 않았습니다. 전신이 보이는 사진을 올려 주세요."); return
    from photo_models import frame_features
    m = pm[e]; A = vp.align_yaw(vp.to_aihub(W[None]), models[e]["yaw_mean"])         # (1,24,3)
    z = (frame_features(A)[0] - np.array(m["mean"])) / np.array(m["scale"])
    zc = np.clip(z, -3, 3); vis = V[vp.KEY_MP].mean()
    rows = []
    for c in m["conditions"]:
        p = 1 / (1 + np.exp(-(zc @ np.array(c["coef"]) + c["intercept"]))); acc = c["cv_bal_acc"]
        level = "✅ 가능" if acc >= PHOTO_OK else "⚠️ 참고만" if acc >= PHOTO_MAYBE else "⛔ 어려움"
        rows.append({"조건": c["name"], "사진 판별": level, "P(올바름)": p if acc >= PHOTO_MAYBE else np.nan,
                     "예측": ("✅ 올바름" if p >= .5 else "❌ 오류") if acc >= PHOTO_MAYBE else "—", "사진 정확도": acc,
                     "영상 정확도": next(x["cv_bal_acc"] for x in models[e]["video"]["conditions"] if x["name"] == c["name"])})
    R = pd.DataFrame(rows)
    far = np.abs(z) > 3; fn = [n.replace("의 평균", "") for n in feature_names()[:89]]   # 한 프레임 특징 = 좌표 72 + 각도 17 (평균 항목과 같은 순서)
    top = [i for i in np.argsort(-np.abs(z)) if far[i]][:5]; detail = ", ".join(f"{fn[i]} ({z[i]:+.1f}σ)" for i in top)
    if vis < 0.6: st.warning(f"주요 관절 신뢰도가 낮습니다({vis:.2f}). 가려진 관절이 있으면 결과를 믿기 어렵습니다.")
    if far.sum() >= PHOTO_FAR_WARN:
        st.error(f"**이 사진은 학습 데이터와 자세·촬영 조건이 다릅니다** — 89개 특징 중 **{far.sum()}개**가 학습 범위(±3σ)를 벗어나 판정을 신뢰하기 어렵습니다 "
                 f"(학습에 쓰지 않은 AI Hub 프레임에서도 3개 이상이면 일치율이 0.76 → 0.63~0.68 로 떨어졌습니다).\n\n크게 다른 부분: {detail}")
    elif far.sum() > 0:
        st.info(f"학습 범위를 벗어난 특징 {far.sum()}개: {detail}")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(img_html(thumb, f"검출된 관절 · 주요 관절 신뢰도 {vis:.2f}"), unsafe_allow_html=True)
    with c2:
        judged = R[R["예측"] != "—"]
        st.markdown(f"#### {e} — 판별한 조건 {len(judged)}개 중 **오류 {int((judged['예측'] == '❌ 오류').sum())}개**")
        for r in judged.itertuples(index=False):
            st.markdown(f"- {r[3]} **{r[0]}** (P(올바름) {r[2]:.2f}, 사진 정확도 {r[4]:.2f})")
        hard = R[R["예측"] == "—"]["조건"].tolist()
        if hard: st.markdown("- ⛔ 사진으로 판별 어려움: " + ", ".join(hard))
    st.markdown(md_table(R))
    st.caption("사진 정확도: AI Hub 프레임 한 장씩으로 학습·평가한 수행자 분리 교차검증 balanced accuracy. "
               "영상 정확도(16프레임)와 비교하면 '움직임 정보'가 얼마나 필요한지 보입니다. ✅ ≥ 0.75, ⚠️ 0.65~0.75, ⛔ < 0.65")

PAGES = {"🏠 개요": page_overview, "📦 데이터": page_data, "RQ1 · 조건별 성능": page_rq1, "RQ2 · 딥러닝 비교": page_rq2,
         "RQ3 · 2D vs 3D": page_rq3, "RQ4 · 어려운 조건": page_rq4, "🎯 라이브 데모": page_demo, "🎬 영상 판별 (실험적)": page_video,
         "📷 사진 판별 (실험적)": page_photo}
with st.sidebar:
    st.markdown("### 헬스 자세 오류 판별")
    page = st.radio("페이지", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.caption("데이터: AI Hub 피트니스 자세 이미지 (dataSetSn=231) 라벨 JSON · 평가: 수행자 분리 GroupKFold(8)")
PAGES[page]()
