# Dual Arm Imitation Learning (모방 학습) 프로젝트

기존 강화학습(RL, `dual_arm0`) 환경과 완벽히 격리된 **독립 모방 학습(Imitation Learning, IL)** 프로젝트입니다.  
Isaac Sim 환경에서의 텔레오퍼레이션(수동 조작) 시연 데이터 수집부터 최신 로봇 조작 모방 학습 3대 알고리즘(**BC, Diffusion Policy, ACT**) 학습 및 시뮬레이션 평가까지 전 과정을 지원합니다.

---

## 1. 프로젝트 구조

```text
/home/optimus/isaac_lab/dual_arm_il/
├── README.md                  # 프로젝트 사용 가이드 (본 문서)
├── requirements.txt           # 필요한 파이썬 라이브러리 목록
├── configs/                   # 환경 및 모델 하이퍼파라미터
│   ├── env_cfg.py             # 텔레옵/IL 맞춤형 Isaac Lab 환경 설정
│   ├── bc_cfg.yaml            # Behavior Cloning 설정
│   ├── diffusion_cfg.yaml     # Diffusion Policy 설정
│   └── act_cfg.yaml           # ACT (Action Chunking with Transformers) 설정
├── teleop/                    # 텔레오퍼레이션 & 데이터 수집
│   ├── dual_arm_teleop.py     # Active-Arm Toggle SE(3) 키보드 제어기
│   └── collect_demos.py       # Isaac Sim 텔레옵 및 HDF5 에피소드 저장 스크립트
├── dataset/                   # 데이터셋 파이프라인
│   └── il_dataset.py          # PyTorch HDF5 Dataset (정규화, Horizon 슬라이싱, Chunking)
├── models/                    # 모방학습 3대 알고리즘 구현체
│   ├── bc/                    # MLP / RNN Behavior Cloning
│   ├── diffusion/             # 1D Temporal UNet Diffusion Policy (Chi et al. 2023)
│   └── act/                   # CVAE + Transformer ACT (Zhao et al. 2023)
├── scripts/                   # 실행 스크립트
│   ├── generate_scripted_demos.py  # [추천] 스크립트 기반 고품질 단일 데모 자동 생성기
│   ├── generate_scripted_demos2.py # [강력 추천] 대규모 텐서 병렬화 초고속 자동 생성기
│   ├── replay_demos.py         # 수집된 HDF5 데모 시뮬레이션 재생 검증
│   ├── train.py               # 3종 알고리즘 통합 고속 GPU 학습 스크립트
│   └── eval.py                # Isaac Sim 환경에서 정책 롤아웃 평가
├── data/                      # 수집된 데모 파일 (.hdf5) 저장 경로
└── checkpoints/               # 훈련된 모델 체크포인트 저장 경로
```

---

## 2. 텔레오퍼레이션 (키보드 조작법)

양팔(14자유도 + 2개 그리퍼)을 직관적으로 다룰 수 있도록 **활성 팔 전환(Tab 키)** 방식을 제공합니다.

| 분류 | 키 | 동작 설명 |
| :--- | :---: | :--- |
| **팔 선택** | `TAB` | 오른팔(Pick) <-> 왼팔(Place) 전환 토글 |
| | `1` / `2` | 1번: 오른팔 즉시 선택, 2번: 왼팔 즉시 선택 |
| **위치 이동** | `W` / `S` | X축 전진(+) / 후진(-) |
| | `A` / `D` | Y축 좌측(+) / 우측(-) |
| | `Q` / `E` | Z축 상승(+) / 하강(-) |
| **자세 회전** | `Z` / `X` | Roll (X축 회전) |
| | `T` / `G` | Pitch (Y축 회전) |
| | `C` / `V` | Yaw (Z축 회전) |
| **그리퍼** | `SPACE` or `K` | 현재 활성 팔의 그리퍼 열기/닫기 토글 |
| | `O` / `P` | 현재 활성 팔의 그리퍼 강제 열기(`O`) / 닫기(`P`) |
| **데이터 수집** | `ENTER` or `Y` | **[성공 저장]** 현재 에피소드를 HDF5에 확정 저장 후 리셋 |
| | `BACKSPACE` or `N` | **[폐기]** 실수한 에피소드를 저장하지 않고 폐기 후 리셋 |
| | `R` | 환경 즉시 리셋 |
| **속도 조절** | `UP` / `DOWN` | 1회 입력당 이동 속도 증가 / 감소 |

---

## 3. 사용 단계별 가이드

### ① 1단계: 데모 데이터 수집 (3가지 방법 중 선택)

#### [방법 A: 강력 추천] 대규모 텐서 병렬화 데모 자동 생성기 (`generate_scripted_demos2.py`)
역운동학(IK) 솔버와 상태 기계(State Machine)를 100% PyTorch GPU 텐서 연산으로 병렬화하여, **수십 대의 로봇이 동시에 각기 다른 데모를 0초의 병목 없이 쏟아내는 초고속 파이프라인**입니다.
손목 특이점(Singularity) 방지, 대기 시간 최적화, Baton 드롭 자동 감지 등 가장 진보된 패치가 모두 적용되어 있습니다.

```bash
# 50대의 로봇이 50개의 데모를 단 한 번의 에피소드 재생(수 초)만에 수집!
python scripts/generate_scripted_demos2.py --num_demos=50 --num_envs=50 --headless
```

#### [방법 B] 스크립트 기반 단일 데모 자동 생성기 (`generate_scripted_demos.py`)
디버깅이나 시각적 확인이 필요할 때 1대의 로봇이 순차적으로 완벽한 고품질 데모를 생성합니다. (대기 시간 최적화 패치 적용 완료)

```bash
# GUI 화면을 보면서 50개 데모 자동 수집
python scripts/generate_scripted_demos.py --num_demos=50
```

#### [방법 C] 키보드 텔레오퍼레이션 수동 수집 (`collect_demos.py`)
Isaac Sim GUI를 띄우고 직접 키보드로 조작하여 성공 에피소드를 수동 녹화합니다.

```bash
# 기본 20개 데모 수집 (data/demos.hdf5에 자동 누적 저장)
python teleop/collect_demos.py --num_demos=20
```

> **팁**: 조작 중 실수가 발생했을 때 `BACKSPACE`나 `N`을 누르면 데이터셋에 오염되지 않고 즉시 버려지며 새 에피소드가 시작됩니다.

---

### ② 2단계: 수집된 데모 검증 (`replay_demos.py`)
녹화된 HDF5 궤적이 시뮬레이션 물리 환경에서 안정적으로 동작하는지 재생해 봅니다.

```bash
# 0번 데모 재생
python scripts/replay_demos.py --demo_idx=0

# 전체 수집된 데모 순차 재생
python scripts/replay_demos.py --demo_idx=-1
```

---

### ③ 3단계: 모방학습 모델 훈련 (`train.py`)
Isaac Sim을 켜지 않고 순수 PyTorch 환경에서 고속 GPU 병렬 학습을 수행합니다.

```bash
# 1. Behavior Cloning (MLP 베이스라인)
python scripts/train.py --algo=bc --epochs=100

# 2. Diffusion Policy (1D Temporal UNet)
python scripts/train.py --algo=diffusion --epochs=150

# 3. ACT (Action Chunking with Transformers)
python scripts/train.py --algo=act --epochs=150
```

- 학습 중 최저 Validation Loss를 기록한 가중치는 `checkpoints/{algo}/best_model.pt`로 자동 보존됩니다.
- 입력 관측치/액션의 정규화 통계치는 `checkpoints/{algo}/stats.pkl`에 저장됩니다.

---

### ④ 4단계: Isaac Sim 시뮬레이션 평가 (`eval.py`)
학습된 모델을 시뮬레이션 로봇에 연결하여 실제 클로즈드 루프 성공률을 측정합니다.

```bash
# Diffusion Policy 평가
python scripts/eval.py --algo=diffusion --num_episodes=10

# ACT 평가 (Temporal Ensembling 적용)
python scripts/eval.py --algo=act --num_episodes=10

# Behavior Cloning 평가
python scripts/eval.py --algo=bc --num_episodes=10
```

---

## 4. 기존 RL 환경과의 관계

- 본 폴더(`dual_arm_il`)의 모든 코드는 `/home/optimus/isaac_lab/dual_arm0`의 파일을 수정하지 않습니다.
- 시뮬레이션 씬(`DualArmSceneCfg`) 및 로봇 정의는 환경에 설치된 설정을 안전하게 상속(`configs/env_cfg.py`)받아 재사용하므로, RL 쪽에서 현재 진행 중인 학습이나 튜닝에 아무런 간섭을 주지 않습니다.
