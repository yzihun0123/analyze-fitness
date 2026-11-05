#!/usr/bin/env python3
"""Roboflow 'Dumbbell Biceps Curl Detection' v12 (CC BY 4.0, humanarm) → 프레임별 MediaPipe 관절 캐시.

사용:  python rf_pose.py "<데이터셋 폴더>" [--out rf_cache/rf_pose.npz]
- 밝기만 바꾼 Roboflow 사본(.rf.<해시>)은 원본당 1장만 사용
- 라벨: 'Dumbbell Bicep Curl'(0)만 → 정자세 1, 'incorrect'(2)만 → 잘못된 자세 0, 둘 다/없음 → 제외
- 그룹: 파일명의 영상 이름(…_mp4-0014 의 앞부분) — 평가는 영상 단위로 나눈다(Roboflow 분할은 같은 영상 프레임이 섞여 있음)
"""
import os, re, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_pose as vp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", default="rf_cache/rf_pose.npz")
    a = ap.parse_args()
    import mediapipe as mp, cv2
    from mediapipe.tasks.python import vision, BaseOptions
    items, seen = [], set()
    for split in ["train", "valid", "test"]:
        lab = os.path.join(a.root, split, "labels")
        for f in sorted(os.listdir(lab)):
            base = re.sub(r"\.rf\.[0-9a-f]+\.txt$", "", f)
            if base in seen: continue
            seen.add(base)
            cls = {int(l.split()[0]) for l in open(os.path.join(lab, f), encoding="utf-8").read().splitlines() if l.strip()}
            if (0 in cls) == (2 in cls): continue                       # 둘 다 / 둘 다 없음 → 모호
            items.append((os.path.join(a.root, split, "images", f[:-4] + ".jpg"), int(0 in cls), re.sub(r"-\d+_(jpg|png|jpeg)$", "", base)))
    print("사용 프레임", len(items), "| 정자세 비율", round(np.mean([y for _, y, _ in items]), 3), flush=True)
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=vp.MODEL), running_mode=vision.RunningMode.IMAGE, num_poses=1)
    W = np.full((len(items), 33, 3), np.nan, np.float32); V = np.zeros((len(items), 33), np.float32)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i, (p, _, _) in enumerate(items):
            img = cv2.imread(p)
            if img is None: continue
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
            if r.pose_world_landmarks:
                W[i] = [[q.x, q.y, q.z] for q in r.pose_world_landmarks[0]]; V[i] = [q.visibility for q in r.pose_world_landmarks[0]]
            if i % 500 == 0: print(i, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, W=W, V=V, y=np.array([y for _, y, _ in items]), video=np.array([g for _, _, g in items]),
                        path=np.array([os.path.relpath(p, a.root) for p, _, _ in items]))
    print("검출 성공", int(np.isfinite(W).all((1, 2)).sum()), "/", len(items), "→", a.out)

if __name__ == "__main__":
    main()
