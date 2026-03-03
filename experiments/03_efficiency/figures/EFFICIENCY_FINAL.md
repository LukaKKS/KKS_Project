# ViCo-Lite 효율 실험 최종 정리

**기준**: 에피소드당 평균 **620 스텝** (ablation 로그 12 에피소드, 스텝 범위 334~1233), **2 에이전트**, **LLM 비사용** (heuristic 100%).  
**지연 측정**: `experiments/01_computational_cost/benchmark_results.json` (CPU, 100회 측정, mean ms).  
**스텝 정의**: `env.step()` 1회 = 에이전트 행동 결정 1회. 호출 횟수·지연은 모두 **스텝** 기준.

---

## Experiment 1. 호출 횟수 + 지연 시간 (전체)

### 에피소드 1개 기준

| 항목 | 값 |
|------|-----|
| **에피소드당 총 호출 (유의미 모듈)** | **약 4,464** |
| **에피소드당 ViCo 연산 지연 (CPU)** | **약 90.8 s (1.5 min)** |
| **스텝당 평균 지연 (ViCo 연산만)** | **약 146 ms** |

- LLM 호출: **0회**.  
- 위 지연은 TDW 시뮬·I/O 제외, ViCo 모듈 연산만 합한 값.

---

## Experiment 2. 모듈별 호출 횟수 + 지연 시간

### (A) 1회 호출당 지연 (benchmark, CPU)

| 모듈 | mean (ms) | p95 (ms) | 비고 |
|------|-----------|----------|------|
| CLIP image encode | 43.08 | 48.90 | perception 내부 |
| CLIP text encode | 25.93 | 29.33 | perception 내부 |
| **Full perception (CLIP+심볼)** | **72.61** | 81.91 | 1회 = 이미지+텍스트 인코딩 등 |
| EMA memory update | 0.012 | 0.015 | latent_dim=1024 |
| Team hub update | 0.037 | 0.043 | n_agents=2 |
| Heuristic scoring | 0.066 | 0.070 | n_candidates=10 |
| A* pathfinding 120×120 | 1.20 | 1.65 | — |
| A* pathfinding **240×240** | **4.91** | 5.28 | 기본 사용 |
| A* pathfinding 480×480 | 29.53 | 31.02 | — |

### (B) 에피소드 1개 기준 — 모듈별 호출 횟수 + 지연

**가정**: 스텝당 perception·EMA·heuristic 각 2회(에이전트당 1회), team_hub 1회, A*는 스텝의 약 20%에서 1회(240×240).

| 모듈 | 호출 횟수/에피소드 | 1회 mean (ms) | 에피소드 합계 (ms) | 합계 (초) |
|------|---------------------|---------------|---------------------|-----------|
| Full perception (CLIP+심볼) | 1,240 | 72.61 | 90,036 | **90.0** |
| EMA memory update | 1,240 | 0.012 | 15 | 0.02 |
| Team hub update | 620 | 0.037 | 23 | 0.02 |
| Heuristic scoring | 1,240 | 0.066 | 82 | 0.08 |
| A* pathfinding (240×240) | 124 | 4.91 | 609 | **0.6** |
| **합계** | **~4,464** | — | **90,765** | **~90.8** |

### (C) 스텝당 호출 (2 에이전트 기준)

| 모듈 | 호출/스텝 |
|------|-----------|
| Full perception | 2 |
| EMA memory update | 2 |
| Team hub update | 1 |
| Heuristic scoring | 2 |
| A* pathfinding | ≈ 0.2 (20% 스텝) |

---

## 요약 표 (논문/보고용)

**에피소드 1개 (평균 620 스텝, 2 에이전트, heuristic-only)**

| 구분 | 호출 횟수 | 지연 (ms) | 지연 (초) |
|------|-----------|-----------|-----------|
| Perception (CLIP+심볼) | 1,240 | 90,036 | 90.0 |
| Memory (EMA + Team hub) | 1,860 | 38 | 0.04 |
| Heuristic scoring | 1,240 | 82 | 0.08 |
| A* pathfinding | 124 | 609 | 0.6 |
| **합계** | **~4,464** | **~90,765** | **~90.8 (약 1.5 min)** |

---

## 데이터 출처

| 항목 | 출처 |
|------|------|
| 1회 호출당 지연 (각 모듈) | `experiments/01_computational_cost/benchmark_results.json` |
| 에피소드당 평균 스텝 (620) | `experiments/02_ablation` 로그 "done at step N" 12 에피소드 평균 |
| 의사 결정 소스 (heuristic vs LLM) | `experiments/03_efficiency/figures/efficiency_metrics.json` |

**참고**: 모듈별 에피소드 호출 횟수는 “스텝당 2 에이전트, A* 20%” 등 **가정**에 따른 추정이며, 코드에 per-module 호출 계측은 없음. 논문/보고 시 각주 권장.
