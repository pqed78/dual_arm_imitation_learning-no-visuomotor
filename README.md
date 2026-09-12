# Dual Arm Imitation Learning Project

This is an **independent Imitation Learning (IL)** project that is completely isolated from the existing Reinforcement Learning (RL, `dual_arm0`) environment.
It supports the entire pipeline, from teleoperation demonstration data collection in the Isaac Sim environment to the training and simulation evaluation of three state-of-the-art robot manipulation imitation learning algorithms (**BC, Diffusion Policy, ACT**).

---

## 1. Project Structure

```text
/home/optimus/isaac_lab/dual_arm_il/
├── README.md                  # Project usage guide (This document)
├── requirements.txt           # Required Python libraries list
├── configs/                   # Environment and model hyperparameters
│   ├── env_cfg.py             # Teleop/IL custom Isaac Lab environment settings
│   ├── bc_cfg.yaml            # Behavior Cloning settings
│   ├── diffusion_cfg.yaml     # Diffusion Policy settings
│   └── act_cfg.yaml           # ACT (Action Chunking with Transformers) settings
├── teleop/                    # Teleoperation & Data collection
│   ├── dual_arm_teleop.py     # Active-Arm Toggle SE(3) keyboard controller
│   └── collect_demos.py       # Isaac Sim teleop and HDF5 episode saving script
├── dataset/                   # Dataset pipeline
│   └── il_dataset.py          # PyTorch HDF5 Dataset (Normalization, Horizon slicing, Chunking)
├── models/                    # 3 Major Imitation Learning Algorithms
│   ├── bc/                    # MLP / RNN Behavior Cloning
│   ├── diffusion/             # 1D Temporal UNet Diffusion Policy (Chi et al. 2023)
│   └── act/                   # CVAE + Transformer ACT (Zhao et al. 2023)
├── scripts/                   # Execution scripts
│   ├── generate_scripted_demos_parallel.py  # [Recommended] Script-based high-quality single demo auto-generator

│   ├── replay_demos.py         # Verify collected HDF5 demos via simulation replay
│   ├── train.py               # Integrated high-speed GPU training script for the 3 algorithms
│   └── eval.py                # Policy rollout evaluation in Isaac Sim environment
├── data/                      # Directory for collected demo files (.hdf5)
└── checkpoints/               # Directory for trained model checkpoints
```

---

## 2. Teleoperation (Keyboard Controls)

To intuitively control both arms (14 DoF + 2 grippers), an **Active-Arm Toggle (Tab key)** method is provided.

| Category | Key | Description |
| :--- | :---: | :--- |
| **Arm Selection** | `TAB` | Toggle between Right Arm (Pick) <-> Left Arm (Place) |
| | `1` / `2` | 1: Force select Right Arm, 2: Force select Left Arm |
| **Translation** | `W` / `S` | Move X-axis Forward(+) / Backward(-) |
| | `A` / `D` | Move Y-axis Left(+) / Right(-) |
| | `Q` / `E` | Move Z-axis Up(+) / Down(-) |
| **Rotation** | `Z` / `X` | Roll (X-axis rotation) |
| | `T` / `G` | Pitch (Y-axis rotation) |
| | `C` / `V` | Yaw (Z-axis rotation) |
| **Gripper** | `SPACE` or `K` | Toggle open/close gripper of the active arm |
| | `O` / `P` | Force open (`O`) / close (`P`) gripper of the active arm |
| **Data Collection** | `ENTER` or `Y` | **[Save Success]** Save the current episode to HDF5 and reset |
| | `BACKSPACE` or `N` | **[Discard]** Discard the mistaken episode without saving and reset |
| | `R` | Reset environment immediately |
| **Speed Control** | `UP` / `DOWN` | Increase / Decrease movement speed per input |

---

## 3. Step-by-Step Usage Guide

### ① Step 1: Demo Data Collection (Choose 1 of 2 methods)

#### [Method A] Script-based Single Demo Auto-generator (`generate_scripted_demos_parallel.py`)
Generates perfect, high-quality demos sequentially using a single robot when debugging or visual confirmation is needed. (Wait time optimization patch applied)

```bash
# Auto-collect 50 demos while watching the GUI screen
python scripts/generate_scripted_demos_parallel.py --num_demos=50
```

#### [Method B] Manual Keyboard Teleoperation Collection (`collect_demos.py`)
Launches the Isaac Sim GUI and allows manual recording of successful episodes by controlling the robot directly with a keyboard.

```bash
# Collect 20 baseline demos (auto-accumulated and saved in data/demos.hdf5)
python teleop/collect_demos.py --num_demos=20
```

> **Tip**: If you make a mistake during operation, pressing `BACKSPACE` or `N` will immediately discard the data without polluting the dataset and start a new episode.

---

### ② Step 2: Verify Collected Demos (`replay_demos.py`)
Replay the recorded HDF5 trajectories in the simulation physics environment to ensure they operate stably.

```bash
# Replay demo 0
python scripts/replay_demos.py --demo_idx=0

# Sequentially replay all collected demos
python scripts/replay_demos.py --demo_idx=-1
```

---

### ③ Step 3: Imitation Learning Model Training (`train.py`)
Performs high-speed GPU parallel training in a pure PyTorch environment without launching Isaac Sim.

```bash
# 1. Behavior Cloning (MLP Baseline)
python scripts/train.py --algo=bc --epochs=100

# 2. Diffusion Policy (1D Temporal UNet)
python scripts/train.py --algo=diffusion --epochs=150

# 3. ACT (Action Chunking with Transformers)
python scripts/train.py --algo=act --epochs=150
```

- Weights recording the lowest Validation Loss during training are automatically preserved at `checkpoints/{algo}/best_model.pt`.
- Normalization statistics for input observations/actions are saved at `checkpoints/{algo}/stats.pkl`.
- **Resume Training**: You can resume training from a saved checkpoint by passing the `--resume` flag:
  ```bash
  python scripts/train.py --algo=diffusion --epochs=100 --resume="checkpoints/diffusion/best_model.pt"
  ```

---

### ④ Step 4: Isaac Sim Simulation Evaluation (`eval.py`)
Connects the trained model to the simulation robot to measure the actual closed-loop success rate.

```bash
# Evaluate Diffusion Policy
python scripts/eval.py --algo=diffusion --num_episodes=10

# Evaluate ACT (Temporal Ensembling applied)
python scripts/eval.py --algo=act --num_episodes=10

# Evaluate Behavior Cloning
python scripts/eval.py --algo=bc --num_episodes=10
```

---

## 4. Relationship with Existing RL Environment

- All code in this folder (`dual_arm_il`) does not modify any files in `/home/optimus/isaac_lab/dual_arm0`.
- Simulation scenes (`DualArmSceneCfg`) and robot definitions are safely inherited (`configs/env_cfg.py`) and reused, meaning it causes absolutely no interference with ongoing training or tuning on the RL side.

---

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
│   ├── generate_scripted_demos_parallel.py  # [추천] 스크립트 기반 고품질 단일 데모 자동 생성기

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

### ① 1단계: 데모 데이터 수집 (2가지 방법 중 선택)

#### [방법 A] 스크립트 기반 단일 데모 자동 생성기 (`generate_scripted_demos_parallel.py`)
디버깅이나 시각적 확인이 필요할 때 1대의 로봇이 순차적으로 완벽한 고품질 데모를 생성합니다. (대기 시간 최적화 패치 적용 완료)

```bash
# GUI 화면을 보면서 50개 데모 자동 수집
python scripts/generate_scripted_demos_parallel.py --num_demos=50
```

#### [방법 B] 키보드 텔레오퍼레이션 수동 수집 (`collect_demos.py`)
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
- **학습 이어하기 (Resume)**: `--resume` 옵션을 사용해 기존에 학습된 가중치부터 이어서 학습할 수 있습니다.
  ```bash
  python scripts/train.py --algo=diffusion --epochs=100 --resume="checkpoints/diffusion/best_model.pt"
  ```

---

### ④ 4단계: Isaac Sim 시뮬레이션 평가 (`eval.py` / `eval_parallel.py`)
학습된 모델을 시뮬레이션 로봇에 연결하여 실제 클로즈드 루프 성공률을 측정합니다. 모델의 성능을 빠르고 정확하게 검증하기 위해 두 가지 평가 모드를 제공합니다.

1. **병렬 고속 평가 (`eval_parallel.py`) [추천]**
   여러 환경(Environment)을 동시에 띄워 단기간에 대량의 에피소드를 평가합니다.
   ```bash
   # 16개의 환경에서 총 100개의 에피소드를 병렬로 평가
   python scripts/eval_parallel.py --algo=diffusion --num_episodes=100 --num_envs=16 --headless
   ```

2. **단일 환경 순차 평가 (`eval.py`)**
   시각적 확인을 위해 하나의 환경에서 에피소드를 순차적으로 실행하며 평가합니다. ACT, BC 등 다른 알고리즘도 동일하게 적용 가능합니다.
   ```bash
   # 단일 환경에서 순차적으로 10개의 에피소드 평가 (GUI 시각화 모드)
   python scripts/eval.py --algo=diffusion --num_episodes=10
   ```

---

## 4. 기존 RL 환경과의 관계

- 본 폴더(`dual_arm_il`)의 모든 코드는 `/home/optimus/isaac_lab/dual_arm0`의 파일을 수정하지 않습니다.
- 시뮬레이션 씬(`DualArmSceneCfg`) 및 로봇 정의는 환경에 설치된 설정을 안전하게 상속(`configs/env_cfg.py`)받아 재사용하므로, RL 쪽에서 현재 진행 중인 학습이나 튜닝에 아무런 간섭을 주지 않습니다.

---

## 5. Antigravity AI 프로젝트 재구성 프롬프트

차후에 Antigravity AI(또는 다른 LLM 에이전트)에게 현재 프로젝트와 동일한 구조와 기능을 가진 모방 학습(IL) 파이프라인을 처음부터 구축해 달라고 요청할 때 사용할 수 있는 프롬프트 템플릿입니다. 복사해서 사용하시면 현재 프로젝트의 구조를 그대로 복원할 수 있습니다.

### 프롬프트 복사하기
> **역할 및 목표:**
> 너는 Isaac Lab과 PyTorch를 기반으로 로봇 팔 제어 및 모방 학습(Imitation Learning) 환경을 구축하는 전문가야.
> 기존의 강화학습(RL) 환경(예: `dual_arm0`) 코드를 전혀 건드리지 않고, 독립적인 모방 학습 전용 프로젝트 폴더(`dual_arm_il`)를 생성해서 완전한 IL 파이프라인을 구성해 줘.
> 
> **주요 요구사항:**
> 1. **프로젝트 구조:** `configs/`, `teleop/`, `dataset/`, `models/`, `scripts/`, `data/`, `checkpoints/`, `runs/` 폴더로 기능별 역할을 완벽히 분리해 줘.
> 2. **텔레오퍼레이션 (`teleop/`):** Isaac Sim에서 키보드를 통해 양팔 로봇(14 DoF + 2 Gripper)을 조작하고, 성공한 에피소드(관측치 및 액션 궤적)를 HDF5 포맷으로 저장할 수 있는 스크립트를 만들어 줘. 반드시 Active-Arm Toggle 방식(Tab 키로 양팔을 번갈아 제어)을 지원해야 해.
> 3. **데이터셋 (`dataset/`):** 수집된 HDF5 데이터를 PyTorch DataLoader에서 사용할 수 있도록 파싱하고, Observation/Action을 Min-Max로 정규화(Normalization)하며, Horizon(과거 관측치 개수, 미래 예측 액션 개수) 단위로 텐서를 슬라이싱(Chunking)하는 데이터셋 클래스를 구현해 줘.
> 4. **모델 아키텍처 (`models/`):** 다음 3가지 최신 모방 학습 알고리즘을 각각 모듈화하여 구현해 줘.
>    - Behavior Cloning (MLP / RNN 기반 Baseline)
>    - Diffusion Policy (1D Temporal UNet 구조 및 DDPM/DDIM 스케줄러 기반)
>    - ACT (Action Chunking with Transformers, CVAE 기반)
> 5. **통합 학습 스크립트 (`scripts/train.py`):**
>    - CLI의 `--algo` 인자를 통해 위 3가지 알고리즘 중 하나를 유연하게 선택해 학습할 수 있어야 해.
>    - 검증 손실(Validation Loss)이 갱신될 때마다 `best_model.pt`로 저장하는 로직을 포함해 줘.
>    - **[중요] TensorBoard 연동:** `torch.utils.tensorboard.SummaryWriter`를 사용해 Train Loss, Val Loss, Learning Rate의 변화 추이를 `runs/` 폴더에 실시간으로 기록하는 코드를 필수로 넣어 줘.
> 6. **자동 데모 수집 (`scripts/generate_scripted_demos_parallel.py`):** 사람의 키보드 조작 없이 코드(State Machine)로 로봇을 제어하여 완벽한 데모를 대량(예: 50개)으로 자동 수집하는 스크립트를 작성해 줘.
> 7. **검증 및 평가 (`scripts/eval.py`, `replay_demos.py`):** 학습이 완료된 가중치 모델을 Isaac Sim 환경에 띄워 실제 미션 성공률을 Closed-loop로 측정하는 평가 스크립트와, 수집된 데모 파일이 정상적인지 시뮬레이션에서 재현(Replay)하는 스크립트를 구현해 줘.

> 
> **[에이전트 필수 행동 수칙 (CRITICAL)]**
> 1. **설명만 하지 말고 파일 생성하기:** 위에서 언급된 모든 폴더와 Python 파일들을 단순히 설명하는 데 그치지 말고, `write_to_file` 같은 도구를 사용해서 실제 디렉토리에 **모든 파일과 전체 코드를 빠짐없이 생성**해 줘.
> 2. **생략 금지:** 코드 작성 시 `pass`, `TODO`, `...` (구현 생략) 등을 절대 사용하지 마. 나중에 내가 수정할 필요 없이 즉시 실행 가능한(Ready-to-run) 수준의 완전하고 동작하는 코드로 처음부터 끝까지 채워 줘.
> 3. **모든 구성요소 작성:** `requirements.txt` 패키지 목록부터 `configs/` 내부의 YAML 및 Python 환경 설정 파일까지, 프로젝트 구동에 필요한 단 하나의 파일도 누락 없이 전부 직접 작성해 줘.

---

## 5. Antigravity AI 프로젝트 재구성 전체 프롬프트 (파일 통합본)

현재 프로젝트의 **모든 소스 코드(Python, YAML 등) 원본을 포함**하여, 에이전트가 단 하나의 코드도 생략하지 않고 완벽하게 동일한 구조와 코드를 가진 프로젝트를 재구축할 수 있도록 지시하는 마스터 프롬프트를 별도의 파일로 생성해 두었습니다.

👉 **[recreation_prompt.md](recreation_prompt.md)** 

나중에 새로운 환경에서 프로젝트를 복원하실 때, 위 링크된 `recreation_prompt.md` 파일의 전체 내용을 복사해서 에이전트(LLM)에게 전달해 주시면 됩니다. 프롬프트 안에는 다음과 같은 강력한 행동 수칙과 모든 소스 코드가 들어있어 100% 동일한 복원이 보장됩니다.

> **[에이전트 필수 행동 수칙 (CRITICAL)]**
> 1. **설명만 하지 말고 파일 생성하기:** 모든 폴더와 파일들을 실제 디렉토리에 빠짐없이 생성해 줘.
> 2. **코드 생략 절대 금지:** 제공된 텍스트의 파일 내용(코드)을 100% 동일하게 복사해서 작성해 줘. `pass`, `TODO`, `...` 등을 사용해서 코드를 임의로 생략하거나 축약하면 절대 안 돼. 
> 3. **프로젝트 구조 완벽 재현:** 파일 경로에 맞게 폴더 구조를 생성하고 정확히 위치시켜 줘.

---

## 6. (추천) 원클릭 복원 스크립트 (.sh)

LLM 프롬프트를 사용하는 것보다 더 빠르고 확실하게 프로젝트를 동일하게 생성하려면, 포함되어 있는 자동 복원 스크립트를 사용하는 것을 추천합니다.

👉 **[recreate_project.sh](recreate_project.sh)**

위 쉘 스크립트 파일 안에는 현재 프로젝트의 모든 소스 코드가 하드코딩되어 있습니다. 완전히 빈 폴더(또는 새로운 서버 환경)에 이 파일 하나만 덩그러니 복사해두고 아래 명령어만 실행하면 알아서 모든 폴더를 만들고 코드를 찍어냅니다.

```bash
# 실행 권한 부여 후 스크립트 실행
chmod +x recreate_project.sh
./recreate_project.sh
```


## 전체 파이프라인 자동 실행 (Pipeline Execution)
제공되는 쉘 스크립트를 사용하면 **데이터 수집 ➔ 모델 학습 ➔ 순차 평가 ➔ 병렬 평가**로 이어지는 전체 과정을 한 번에 자동으로 실행할 수 있습니다.
You can run the full pipeline (Data Collection ➔ Training ➔ Sequential Eval ➔ Parallel Eval) using the provided bash script:

```bash
./run_pipeline.sh
```


## 7. MacOS (Apple Silicon) 학습 지원 및 투트랙(Two-track) 워크플로우

이 프로젝트는 데이터 수집과 딥러닝 모델 학습이 완벽히 분리되어 있어, **MacOS 환경에서의 순수 모델 학습**을 완벽하게 지원합니다.

### ❌ 데이터 수집 및 시뮬레이션 평가 (MacOS 구동 불가)
Isaac Sim 시뮬레이터를 띄워야 하는 `scripts/generate_scripted_demos_parallel.py` 및 `scripts/eval.py`는 NVIDIA RTX GPU(Linux/Windows)가 필수적입니다.

### ⭕ 인공지능 모델 학습 (`train.py`) (MacOS 구동 가능)
학습 코드(`scripts/train.py`)는 Isaac Sim이나 Omniverse에 전혀 의존하지 않는 순수 PyTorch 스크립트입니다. 데이터 수집이 끝난 뒤, Mac으로 프로젝트 폴더 전체를 복사하면 쾌적한 학습이 가능합니다.

**Mac 환경 학습 방법:**
```bash
# 1. Linux 장비에서 데이터(data/demos.hdf5) 수집 완료 후 전체 폴더 복사

# 2. Mac 환경에서 필수 패키지만 설치
pip install -r requirements.txt

# 3. Apple Silicon (M1/M2/M3) MPS 가속을 이용해 모델 훈련
python scripts/train.py --algo=diffusion --epochs=1500 --device=mps
```

> **Tip:** 학습이 끝나고 생성된 `checkpoints/{algo}/best_model.pt` 가중치 파일만 다시 Linux 장비로 가져가서 `eval.py`로 테스트하시면 가장 효율적인 작업 환경을 구성하실 수 있습니다.
