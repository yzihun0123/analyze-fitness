"""운동 영상(mp4) → MediaPipe 3D 포즈 → AI Hub 24관절 (16프레임) 변환. app.py 의 '영상 판별(실험적)' 페이지에서 사용.

MediaPipe PoseLandmarker(world landmarks, 33점, 미터, 골반 중심)를 AI Hub 3D 관절 정의에 맞춘다.
매핑 계수는 AI Hub 3D 데이터에서 직접 측정한 값이다(클립 31,982개 중앙값):
  Neck  = 어깨중점 + 0.162·(어깨중점-골반중점)      Waist = 골반중점 + 0.322·(어깨중점-골반중점)
  Back  = Neck + 0.549·(Waist-Neck)                  Palm  = 손목 + 0.22·전완길이 (검지·새끼 뿌리 방향)
  Foot  = 발목 + 0.22·정강이길이 (발끝 방향)
좌표계: AI Hub 는 y 가 위인 오른손 좌표계, MediaPipe world 는 x 오른쪽·y 아래·z 화면 안쪽(오른손) → (x, -y, -z).
"""
import os, tempfile
import numpy as np
from build_cache import JOINTS

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task")
MP = {"nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8, "l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14,
      "l_wr": 15, "r_wr": 16, "l_pinky": 17, "r_pinky": 18, "l_index": 19, "r_index": 20, "l_hip": 23, "r_hip": 24,
      "l_kn": 25, "r_kn": 26, "l_an": 27, "r_an": 28, "l_toe": 31, "r_toe": 32}
DIRECT = {"Nose": "nose", "Left Eye": "l_eye", "Right Eye": "r_eye", "Left Ear": "l_ear", "Right Ear": "r_ear",
          "Left Shoulder": "l_sh", "Right Shoulder": "r_sh", "Left Elbow": "l_el", "Right Elbow": "r_el",
          "Left Wrist": "l_wr", "Right Wrist": "r_wr", "Left Hip": "l_hip", "Right Hip": "r_hip",
          "Left Knee": "l_kn", "Right Knee": "r_kn", "Left Ankle": "l_an", "Right Ankle": "r_an"}
KEY_MP = [MP[k] for k in ["l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr", "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"]]

# 미리보기용 MediaPipe 33점 연결 (몸통·팔·다리)
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 31),
              (24, 26), (26, 28), (28, 32), (15, 19), (16, 20), (0, 11), (0, 12)]

def read_video(data, max_side=640, max_sec=8, max_fps=15):
    """영상 바이트 → (프레임 RGB 리스트, 실제 사용 fps). 긴 쪽 max_side, 앞 max_sec 초, 초당 max_fps 장까지만(메모리 제한)."""
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data); path = f.name
    try:
        cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / max_fps))); frames, i = [], 0
        while len(frames) < max_sec * fps / step:
            ok, img = cap.read()
            if not ok: break
            if i % step == 0:
                h, w = img.shape[:2]; s = max_side / max(h, w)
                if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)))
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
    finally:
        os.remove(path)
    if not frames: raise ValueError("영상을 읽지 못했습니다 (mp4/mov, H.264 권장).")
    return frames, float(fps / step)

def thumbnail(img, p2, width=240):
    """프레임 + 2D 관절 오버레이 축소 이미지"""
    import cv2
    h, w = img.shape[:2]; s = width / w; im = cv2.resize(img, (width, int(h * s)))
    if np.isfinite(p2).all():
        pts = (p2 * [width, h * s]).astype(int)
        for a, b in POSE_EDGES: cv2.line(im, tuple(pts[a]), tuple(pts[b]), (42, 120, 214), 2)
        for k in KEY_MP: cv2.circle(im, tuple(pts[k]), 3, (235, 104, 52), -1)
    return im

def detect(frames, fps):
    """프레임마다 MediaPipe world landmarks (T,33,3) 와 visibility (T,33). 사람이 안 잡힌 프레임은 NaN."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO, num_poses=1)
    W = np.full((len(frames), 33, 3), np.nan); V = np.zeros((len(frames), 33)); P2 = np.full((len(frames), 33, 2), np.nan)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for t, img in enumerate(frames):
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)), int(t * 1000 / fps))
            if r.pose_world_landmarks:
                W[t] = [[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]]
                V[t] = [p.visibility for p in r.pose_world_landmarks[0]]
                P2[t] = [[p.x, p.y] for p in r.pose_landmarks[0]]   # 화면 정규화 좌표(미리보기용)
    return W, V, P2

def derive(out, hand_tip, toe):
    """직접 관측 관절 dict(17개) + 손끝 방향점·발끝 방향점 → Neck·Waist·Back·Palm·Foot 을 채운 (…,24,3).
    영상(to_aihub)과 영상용 모델 학습(observable)이 같은 공식을 쓰도록 한 곳에 둔다."""
    out = dict(out)
    SM = (out["Left Shoulder"] + out["Right Shoulder"]) / 2; HM = (out["Left Hip"] + out["Right Hip"]) / 2; T = SM - HM
    out["Neck"] = SM + 0.162 * T; out["Waist"] = HM + 0.322 * T; out["Back"] = out["Neck"] + 0.549 * (out["Waist"] - out["Neck"])
    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    for s in ["Left", "Right"]:
        fore = np.linalg.norm(out[f"{s} Wrist"] - out[f"{s} Elbow"], axis=-1, keepdims=True)
        out[f"{s} Palm"] = out[f"{s} Wrist"] + 0.22 * fore * unit(hand_tip[s] - out[f"{s} Wrist"])
        shank = np.linalg.norm(out[f"{s} Ankle"] - out[f"{s} Knee"], axis=-1, keepdims=True)
        out[f"{s} Foot"] = out[f"{s} Ankle"] + 0.22 * shank * unit(toe[s] - out[f"{s} Ankle"])
    return np.stack([out[j] for j in JOINTS], -2)

def detect_image(data, max_side=1024):
    """사진 바이트 → (RGB 이미지, world landmarks (33,3) 또는 None, visibility (33,), 2D (33,2))"""
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise ValueError("사진을 읽지 못했습니다 (jpg/png 권장).")
    h, w = img.shape[:2]; s = max_side / max(h, w)
    if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    if not r.pose_world_landmarks: return img, None, np.zeros(33), np.full((33, 2), np.nan)
    W = np.array([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]])
    V = np.array([p.visibility for p in r.pose_world_landmarks[0]])
    P2 = np.array([[p.x, p.y] for p in r.pose_landmarks[0]])
    return img, W, V, P2

def to_aihub(W):
    """(T,33,3) MediaPipe world → (T,24,3) AI Hub 관절 (y 위, 오른손 좌표계)"""
    W = W * np.array([1, -1, -1])
    g = lambda k: W[..., MP[k], :]
    return derive({j: g(k) for j, k in DIRECT.items()},
                  {s: (g(f"{p}_index") + g(f"{p}_pinky")) / 2 for s, p in [("Left", "l"), ("Right", "r")]},
                  {s: g(f"{p}_toe") for s, p in [("Left", "l"), ("Right", "r")]})

def observable(X):
    """AI Hub (…,24,3) → MediaPipe 로 관측 가능한 정보만 남긴 (…,24,3): Neck·Waist·Back·Palm·Foot 을 영상과 같은 공식으로 재구성.
    영상용 모델은 이 좌표로 학습한다 (척추 점처럼 영상에서 볼 수 없는 정보에 기대지 않도록)."""
    g = lambda j: X[..., JOINTS.index(j), :]
    return derive({j: g(j) for j in DIRECT}, {s: g(f"{s} Palm") for s in ["Left", "Right"]}, {s: g(f"{s} Foot") for s in ["Left", "Right"]})

def sample16(W, t0, t1):
    """[t0, t1] 프레임 구간에서 사람이 잡힌 프레임만 골라 고르게 16장(AI Hub 클립도 원본 31장 중 2장 간격 16장).
    잡힌 프레임이 16장보다 적으면(8장 이상) 일부를 반복해 쓴다. 반환: 사용한 프레임 번호 16개"""
    ok = [t for t in range(t0, t1 + 1) if np.isfinite(W[t]).all()]
    if len(ok) < 8: raise ValueError(f"구간에서 사람이 잡힌 프레임이 {len(ok)}장뿐입니다(8장 이상 필요). 전신이 보이게 찍거나 구간을 넓혀 주세요.")
    return [ok[i] for i in np.linspace(0, len(ok) - 1, 16).round().astype(int)]

def align_yaw(X, yaw_mean_deg):
    """골반 방향을 그 운동의 AI Hub 평균 방향(yaw_mean)에 맞춰 수직축 기준으로 회전"""
    return rotate_yaw(X, np.radians(yaw_mean_deg) - hip_yaw(X))

def hip_yaw(X):
    """(…,16,24,3) → 골반(L→R) 수평 방위각(라디안) — export_models 의 yaw 분포와 같은 정의"""
    hv = X[..., JOINTS.index("Right Hip"), :] - X[..., JOINTS.index("Left Hip"), :]
    return np.arctan2(hv[..., 2].mean(-1), hv[..., 0].mean(-1))

def rotate_yaw(X, th):
    """수직축(y) 기준으로 방위각을 th 만큼 증가시키는 회전"""
    c, s = np.cos(th), np.sin(th); x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([x * c - z * s, y, x * s + z * c], -1)
