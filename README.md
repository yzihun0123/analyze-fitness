# 헬스 운동 자세 오류의 조건별 판별 가능성 분석

3D 키포인트 기반 딥러닝과 수행자 분리 평가 — AI Hub 「피트니스 자세 이미지」 라벨 JSON (41종 운동, 167개 자세 조건)

- **시연 앱**: `streamlit run app.py` (Streamlit Community Cloud 배포용)
- **노트북**: `fitness_pose_project.ipynb` (전체 흐름 + 핵심 단계 실행 결과)
- **보고서**: `REPORT.md`

## 핵심 결과 (수행자 분리 8-fold)

| | 결과 |
|---|---|
| RQ1 | 3D 좌표만으로 조건 평균 balanced accuracy 0.792, 167개 중 89개 조건 ≥ 0.80. 손목 중립(0.675)·이완 시 긴장 유지(0.657)가 가장 어려움 |
| RQ2 | 1D-CNN·GRU(3시드 앙상블) 0.841 / 0.837 vs LogReg 0.836 — 평균은 비슷, 어려운 조건은 개선 못 함 |
| RQ3 | 같은 클립에서 2D 5시점 0.802 ≥ 3D 0.792 > 카메라 1대 0.755~0.783 |
| RQ4 | 손목 중립 = 표현의 문제(관절각 추가 0.675 → 0.761), 긴장 유지·숄더패킹 = 관측의 문제(개선 없음) |

## 파일 구성

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 발표·시연 앱 (원본 데이터 없이 `app_data/`, `results/`, `figures/`만 사용) |
| `fitness_pose_project.ipynb` | 제출용 노트북 (`make_notebook.py`로 생성·실행) |
| `prepare_data.py` | tar → 풀린 JSON |
| `count_data.py` | 데이터 정제 집계 |
| `analyze_fitness.py`, `compare_baseline.py` | 핸드오프 기준 모델과 재현 비교 |
| `build_cache.py` | JSON → 클립 캐시(npz) |
| `exp_seq.py` | 실험: 요약특징+LogReg / 1D-CNN / GRU, 입력 3D·2D·3D+관절각·단일 시점, 수행자 분리 GroupKFold |
| `analyze_results.py` | 표(`results/tables/`)·그림(`figures/`)·`results/report_tables.md` 생성 |
| `export_models.py` | 앱용 모델 계수·평균 골격 내보내기 (`app_data/`) |

## 데이터

AI Hub 원본 데이터(`dataset/`, `data/`, `cache/`)는 이용약관상 재배포할 수 없어 **저장소에 포함하지 않습니다**. 재현하려면 AI Hub에서 라벨 데이터를 받아 `dataset/`에 두고 아래를 실행하세요.

```
pip install -r requirements-train.txt
python prepare_data.py data dataset/기구.tar "dataset/바벨,덤벨.tar" dataset/맨몸운동.tar
python build_cache.py data --out cache
python exp_seq.py --input 3d --exercises all --models logreg --out results/all_logreg_3d_paired
python exp_seq.py --input 3d --exercises rep6 --models logreg,cnn,gru --out results/rep6_3d
python analyze_results.py
python export_models.py
```

전체 명령은 `REPORT.md` §7 참고.
