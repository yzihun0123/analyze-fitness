#!/usr/bin/env python3
"""운동별 조건 다중 라벨 분류: 요약 특징+LogReg(기준) vs 1D-CNN vs GRU, 입력 3D 또는 2D 5시점. 수행자(Z) 분리 GroupKFold.

사용:  python exp_seq.py --cache cache --input 3d --clipset paired --models logreg,cnn,gru --out results/seq_3d
  --exercises  쉼표 구분 운동명, 'rep6'(대표 6종) 또는 'all'
  --clipset    paired = 2D·3D 모두 사용 가능(3D 이상값 제외)한 클립 / all2d = 2D만 사용 가능해도 포함(2D 입력 전용)
같은 (운동, 클립 집합)이면 입력·모델과 무관하게 같은 폴드를 쓴다.
출력: <out>_metrics.csv (운동×조건×모델), <out>_oof.npz (아웃오브폴드 확률), <out>_timing.csv
"""
import os, sys, json, time, argparse, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cache import JOINTS

REP6 = ["랫풀 다운", "딥스", "바벨 스쿼트", "오버 헤드 프레스", "덤벨 컬", "스텝 포워드 다이나믹 런지"]
NECK, LS, RS, WAIST = (JOINTS.index(j) for j in ["Neck", "Left Shoulder", "Right Shoulder", "Waist"])

def load_cache(cache, clipset):
    metas, X2, A, X3 = [], [], [], []
    for f in sorted(os.listdir(cache)):
        if not f.endswith(".json"): continue
        ds = f[:-5]; m = json.load(open(os.path.join(cache, f), encoding="utf-8")); z = np.load(os.path.join(cache, ds + ".npz"))
        keep = [i for i, r in enumerate(m) if r["ok2d"] and (clipset == "all2d" or (r["ok3d"] and not r["outlier3d"]))]
        for i in keep: m[i]["dataset"] = ds
        metas += [m[i] for i in keep]; X2.append(z["X2"][keep]); A.append(z["act"][keep]); X3.append(z["X3"][keep])
    return metas, np.concatenate(X2), np.concatenate(A), np.concatenate(X3)

def norm3d(X3):
    """(N,16,24,3) -> Neck 기준 이동, 클립 평균 어깨너비로 나눔 -> (N,16,72)"""
    sc = np.linalg.norm(X3[:, :, LS] - X3[:, :, RS], axis=-1).mean(1)[:, None, None, None] + 1e-6
    return ((X3 - X3[:, :, NECK:NECK + 1]) / sc).reshape(len(X3), 16, -1)

def norm2d(X2):
    """(N,16,5,24,2) -> 시점마다 Neck 기준 이동, 그 시점의 클립 평균 몸통 길이(Neck-Waist)로 나눔 -> (N,16,5*48)
    측면 시점은 어깨가 겹쳐 2D 어깨너비가 0에 가까워지므로(최소 12.6px) 2D 에서는 몸통 길이를 척도로 쓴다."""
    sc = np.linalg.norm(X2[:, :, :, NECK] - X2[:, :, :, WAIST], axis=-1).mean(1)[:, None, :, None, None] + 1e-6
    return ((X2 - X2[:, :, :, NECK:NECK + 1]) / sc).reshape(len(X2), 16, -1)

ANGLES = [(f"{s} {a}", f"{s} {b}", f"{s} {c}") for s in ("Left", "Right") for a, b, c in
          [("Shoulder", "Elbow", "Wrist"), ("Elbow", "Wrist", "Palm"), ("Hip", "Shoulder", "Elbow"),
           ("Shoulder", "Hip", "Knee"), ("Hip", "Knee", "Ankle"), ("Knee", "Ankle", "Foot")]] + \
         [("Nose", "Neck", "Back"), ("Neck", "Back", "Waist"), ("Back", "Waist", "Left Hip")]

def angles3d(X3):
    """(N,16,24,3) -> 프레임별 관절각 15개(/180) + 귀-어깨 거리 2개(어깨너비 단위) -> (N,16,17)"""
    J = {j: X3[:, :, JOINTS.index(j)] for j in JOINTS}; out = []
    for a, b, c in ANGLES:
        u, v = J[a] - J[b], J[c] - J[b]
        out.append(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9), -1, 1)) / np.pi)
    sc = np.linalg.norm(J["Left Shoulder"] - J["Right Shoulder"], axis=-1).mean(1, keepdims=True) + 1e-6
    out += [np.linalg.norm(J[f"{s} Ear"] - J[f"{s} Shoulder"], axis=-1) / sc for s in ("Left", "Right")]
    return np.stack(out, -1)

def summary(S):
    """(N,16,F) -> 평균·표준편차·가동범위 (N,3F)  (analyze_fitness 의 216 특징과 같은 구성)"""
    return np.concatenate([S.mean(1), S.std(1), S.max(1) - S.min(1)], 1)

def make_keras(kind, T, F, K, seed):
    import keras
    from keras import layers
    keras.utils.set_random_seed(seed)
    x = inp = keras.Input((T, F))
    x = layers.GaussianNoise(0.05)(x)
    if kind == "cnn":
        for _ in range(2):
            x = layers.Conv1D(64, 3, padding="same", use_bias=False)(x)
            x = layers.ReLU()(layers.BatchNormalization()(x))
        x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    else:
        x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(K, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy")
    return m

def fit_predict(kind, Str, Ytr, Gtr, Ste, seed, max_epochs):
    """학습 폴드 안에서 수행자 15%를 조기 종료용 검증으로 떼어 학습 -> 테스트 확률 (n_te, K)"""
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        Xtr, Xte = summary(Str), summary(Ste); P = np.zeros((len(Ste), Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            if len(set(Ytr[:, j])) < 2: P[:, j] = Ytr[0, j]; continue
            P[:, j] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr, Ytr[:, j]).predict_proba(Xte)[:, 1]
        return P, 0
    import keras
    from sklearn.model_selection import GroupShuffleSplit
    mu, sd = Str.mean((0, 1)), Str.std((0, 1)) + 1e-6
    Str, Ste = (Str - mu) / sd, (Ste - mu) / sd
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Str, groups=Gtr))
    m = make_keras(kind, Str.shape[1], Str.shape[2], Ytr.shape[1], seed)
    es = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    h = m.fit(Str[itr], Ytr[itr].astype("float32"), validation_data=(Str[iva], Ytr[iva].astype("float32")),
              epochs=max_epochs, batch_size=32, callbacks=[es], verbose=0)
    return m.predict(Ste, verbose=0), len(h.history["loss"])

def metrics(y, p):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    yh = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh), macro_f1=f1_score(y, yh, average="macro"),
                auroc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache"); ap.add_argument("--input", choices=["3d", "2d", "3d_ang", "3d_ang_vid"] + [f"2dv{v}" for v in range(1, 6)], default="3d",
                                                                 help="3d_ang = 3D 좌표 72 + 프레임별 관절각·귀-어깨 거리 17 / 3d_ang_vid = 3d_ang 을 영상(MediaPipe)으로 "
                                                                      "관측 가능한 관절만으로 재구성 / 2dvN = 2D 단일 시점 N")
    ap.add_argument("--clipset", choices=["paired", "all2d"], default="paired")
    ap.add_argument("--models", default="logreg,cnn,gru"); ap.add_argument("--exercises", default="rep6")
    ap.add_argument("--folds", type=int, default=8); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max-folds-run", type=int, default=0, help="시간 측정용: 앞의 n개 폴드만 실행")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.input.startswith("2d") or a.clipset == "paired", "3D 입력은 paired 클립 집합만 가능"
    from sklearn.model_selection import GroupKFold
    from video_pose import observable
    metas, X2, A, X3 = load_cache(a.cache, a.clipset)
    S_all = {"3d": lambda: norm3d(X3), "2d": lambda: norm2d(X2),
             "3d_ang": lambda: np.concatenate([norm3d(X3), angles3d(X3.astype(np.float64)).astype(np.float32)], -1),
             "3d_ang_vid": lambda: (lambda O: np.concatenate([norm3d(O.astype(np.float32)), angles3d(O).astype(np.float32)], -1))(observable(X3.astype(np.float64))),
             **{f"2dv{v + 1}": (lambda v=v: norm2d(X2[:, :, [v]])) for v in range(5)}}[a.input]()
    byex = collections.defaultdict(list)
    for i, r in enumerate(metas): byex[r["exercise"]].append(i)
    exs = REP6 if a.exercises == "rep6" else sorted(byex, key=lambda e: -len(byex[e])) if a.exercises == "all" else a.exercises.split(",")
    models = a.models.split(","); rows, timing, oof = [], [], {}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ex in exs:
        ids = np.array(byex[ex]); names = metas[ids[0]]["names"]
        bad = [i for i in ids if metas[i]["names"] != names]
        if bad: print(f"  [경고] {ex}: 조건 이름이 다른 클립 {len(bad)}개 제외"); ids = np.array([i for i in ids if metas[i]["names"] == names])
        S = S_all[ids]; Y = np.array([metas[i]["vals"] for i in ids]); G = np.array([metas[i]["z"] for i in ids])
        if not np.isfinite(S).all():
            ok = np.isfinite(S).all((1, 2)); print(f"  [경고] {ex}: 좌표 NaN/inf 클립 {(~ok).sum()}개 제외"); S, Y, G, ids = S[ok], Y[ok], G[ok], ids[ok]
        k = min(a.folds, len(set(G))); folds = list(GroupKFold(n_splits=k).split(S, Y, G))
        print(f"=== {ex}: clips={len(ids)} performers={len(set(G))} conditions={len(names)} input={a.input} clipset={a.clipset}", flush=True)
        oof[ex] = dict(Y=Y, G=G, paths=np.array([metas[i]["path"] for i in ids]), fold=np.zeros(len(ids), int), names=np.array(names))
        for fi, (tr, te) in enumerate(folds): oof[ex]["fold"][te] = fi
        for mn in models:
            P = np.full(Y.shape, np.nan); t0 = time.time()
            for fi, (tr, te) in enumerate(folds):
                if a.max_folds_run and fi >= a.max_folds_run: break
                tf0 = time.time()
                P[te], ep = fit_predict(mn, S[tr], Y[tr], G[tr], S[te], a.seed + fi, a.epochs)
                timing.append(dict(exercise=ex, model=mn, fold=fi, n_train=len(tr), epochs=ep, sec=round(time.time() - tf0, 2)))
            oof[ex][mn] = P; done = np.isfinite(P[:, 0])
            for j, cn in enumerate(names):
                pos = Y[done, j].mean()
                fold_acc = [((P[te, j] >= 0.5) == Y[te, j]).mean() for _, te in folds if np.isfinite(P[te, j]).all()]
                rows.append(dict(exercise=ex, condition=cn, model=mn, input=a.input, clipset=a.clipset, clips=int(done.sum()),
                                 performers=len(set(G[done])), pos_rate=pos, majority=max(pos, 1 - pos),
                                 fold_mean_acc=np.mean(fold_acc), **metrics(Y[done, j], P[done, j])))
            print(f"  {mn:<6} {time.time() - t0:6.1f}s | " + " | ".join(f"{r['condition'][:12]} {r['bal_acc']:.2f}" for r in rows[-len(names):]), flush=True)
        pd.DataFrame(rows).to_csv(a.out + "_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(timing).to_csv(a.out + "_timing.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(a.out + "_oof.npz", **{f"{ex}|{k}": v for ex, d in oof.items() for k, v in d.items()})
    print("saved", a.out + "_metrics.csv")

if __name__ == "__main__":
    main()
