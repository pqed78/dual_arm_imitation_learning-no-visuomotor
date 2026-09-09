# Antigravity AI 프로젝트 재구성 전체 프롬프트

이 파일은 Antigravity AI(혹은 다른 LLM 에이전트)에게 현재 프로젝트와 **토씨 하나 틀리지 않고 완벽하게 동일한** 프로젝트를 재구축하도록 지시할 때 사용하는 전체 프롬프트입니다.

이 파일 전체 내용을 복사해서 에이전트에게 전달하세요.

---
**[프롬프트 시작]**

너는 Isaac Lab과 PyTorch를 기반으로 로봇 팔 제어 및 모방 학습(Imitation Learning) 환경을 구축하는 전문가야.
기존의 강화학습(RL) 환경 코드를 건드리지 않고, 독립적인 모방 학습 전용 폴더(`dual_arm_il`) 내에 완전한 IL 파이프라인을 아래 제공된 파일 내용과 **정확히 동일하게** 구성해 줘.

**[에이전트 필수 행동 수칙 (CRITICAL)]**
1. **설명만 하지 말고 파일 생성하기:** 아래 명시된 모든 폴더와 파일들을 `write_to_file` 도구를 사용해서 실제 디렉토리에 빠짐없이 생성해 줘.
2. **코드 생략 절대 금지:** 아래 텍스트에 포함된 파일 내용(코드)을 100% 동일하게 복사해서 작성해 줘. `pass`, `TODO`, `...` 등을 사용해서 코드를 임의로 생략하거나 축약하면 절대 안 돼. 제공된 파일 내용 그대로 완전한 코드로 작성해야 해.
3. **프로젝트 구조 완벽 재현:** 파일 경로에 맞게 폴더 구조를 생성하고 그 안에 파일을 정확히 위치시켜 줘.

아래는 네가 정확하게 작성해야 할 파일들의 경로와 전체 소스 코드 내용이야.

---

### [File: `README.md`]
```md
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
│   ├── generate_scripted_demos.py  # [Recommended] Script-based high-quality single demo auto-generator

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

#### [Method A] Script-based Single Demo Auto-generator (`generate_scripted_demos.py`)
Generates perfect, high-quality demos sequentially using a single robot when debugging or visual confirmation is needed. (Wait time optimization patch applied)

```bash
# Auto-collect 50 demos while watching the GUI screen
python scripts/generate_scripted_demos.py --num_demos=50
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
│   ├── generate_scripted_demos.py  # [추천] 스크립트 기반 고품질 단일 데모 자동 생성기

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

#### [방법 A] 스크립트 기반 단일 데모 자동 생성기 (`generate_scripted_demos.py`)
디버깅이나 시각적 확인이 필요할 때 1대의 로봇이 순차적으로 완벽한 고품질 데모를 생성합니다. (대기 시간 최적화 패치 적용 완료)

```bash
# GUI 화면을 보면서 50개 데모 자동 수집
python scripts/generate_scripted_demos.py --num_demos=50
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
> 6. **자동 데모 수집 (`scripts/generate_scripted_demos.py`):** 사람의 키보드 조작 없이 코드(State Machine)로 로봇을 제어하여 완벽한 데모를 대량(예: 50개)으로 자동 수집하는 스크립트를 작성해 줘.
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
```

### [File: `__init__.py`]
```python
# Dual Arm Imitation Learning Package
```

### [File: `configs/act_cfg.yaml`]
```yaml
# Action Chunking with Transformers (ACT) Configuration

algo: "act"

# Chunking parameters
chunk_size: 24         # Action chunk length (num future actions predicted per step)
kl_weight: 10.0        # Weight for CVAE latent KL divergence loss

# Transformer architecture
d_model: 256
nhead: 8
num_encoder_layers: 4
num_decoder_layers: 6
dim_feedforward: 1024
dropout: 0.1
latent_dim: 32         # Dimension of CVAE style latent variable z

# Inference / Rollout
temporal_ensembling: true
temporal_ensemble_coeff: 0.01  # Exponential weighting for overlapping action predictions

# Training hyperparameters
learning_rate: 1.0e-4
weight_decay: 1.0e-4
batch_size: 64
epochs: 150
lr_drop_epoch: 100

# Checkpointing & Logging
save_interval: 10
eval_interval: 5
```

### [File: `configs/bc_cfg.yaml`]
```yaml
# Behavior Cloning (BC) Configuration

algo: "bc"
model_type: "mlp" # "mlp" or "rnn"

# Network architecture
hidden_dims: [512, 512, 256]
activation: "relu"
dropout: 0.1

# RNN specific (if model_type == "rnn")
rnn_hidden_dim: 256
rnn_num_layers: 2
seq_len: 10

# Training hyperparameters
learning_rate: 1.0e-3
weight_decay: 1.0e-5
batch_size: 128
epochs: 100
lr_drop_epoch: 60

# Checkpointing & Logging
save_interval: 10
eval_interval: 5
```

### [File: `configs/diffusion_cfg.yaml`]
```yaml
# Diffusion Policy Configuration (1D Temporal UNet)

algo: "diffusion"

# Horizon parameters
pred_horizon: 16       # Number of future actions predicted by model (Tp)
obs_horizon: 2         # Number of historical observations conditioned on (To)
act_horizon: 8         # Number of action steps executed before replanning (Ta)

# UNet Architecture
down_dims: [256, 512, 1024]
kernel_size: 5
n_groups: 8
cond_predict_scale: true

# Diffusion Scheduler (DDPM / DDIM)
num_train_timesteps: 100
beta_schedule: "squaredcos_cap_v2" # or "linear"
prediction_type: "epsilon"         # predict noise epsilon or sample x_0
num_inference_steps: 15            # DDIM accelerated sampling steps for fast closed-loop control

# Training hyperparameters
learning_rate: 1.0e-4
weight_decay: 1.0e-6
batch_size: 64
epochs: 150
lr_scheduler: "cosine"
lr_warmup_steps: 500

# Checkpointing & Logging
save_interval: 10
eval_interval: 5
```

### [File: `configs/env_cfg.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Dual Arm Imitation Learning.

This configuration inherits directly from the RL environment (dual_arm0)
without modifying any RL code, while customizing parameters for
teleoperation, demonstration collection, and IL policy rollout.
"""

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

# Import the base configuration from dual_arm0 (installed in environment)
from dual_arm0.tasks.dual_arm.dual_arm_env_cfg import DualArmEnvCfg, DualArmSceneCfg
from dual_arm0.tasks.dual_arm import rewards as base_rewards


@configclass
class DualArmILEnvCfg(DualArmEnvCfg):
    """Environment configuration tailored for Imitation Learning."""

    def __post_init__(self):
        super().__post_init__()

        # For teleoperation and evaluation, default to 1 environment
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Episode settings for teleoperation & evaluation
        self.episode_length_s = 25.0  # 25 seconds per episode to give human ample time
        self.decimation = 2           # 60Hz / 2 = 30Hz control frequency

        # Disable automatic time_out termination during teleop if needed (can be toggled)
        # self.terminations.time_out = None

        # Camera viewer position for comfortable human teleoperation view
        self.viewer.eye = (1.2, 0.0, 1.0)
        self.viewer.lookat = (0.35, 0.0, 0.2)

        # -------------------------------------------------------------
        # Gripper Clamping Force & Grasp Stability Enhancements
        # -------------------------------------------------------------
        import copy
        self.scene.robot = copy.deepcopy(self.scene.robot)
        self.scene.object = copy.deepcopy(self.scene.object)

        # 1. Gripper Actuator Clamping Force:
        # Default in DUAL_FRANKA_CFG was: stiffness=2000.0, damping=100.0, effort_limit=200.0.
        # When grasping a 3cm baton, finger position error is only ~0.015m,
        # producing only 30 N of clamping force (2000.0 * 0.015).
        # We increase stiffness to 10,000 N/m (5x force) and effort_limit to 400.0 N:
        # Clamping force becomes 150 N per finger (total 300 N normal force).
        if "hand" in self.scene.robot.actuators:
            self.scene.robot.actuators["hand"].stiffness = 3000.0
            self.scene.robot.actuators["hand"].damping = 100.0
            self.scene.robot.actuators["hand"].effort_limit = 300.0

        # 2. Realistic Baton Mass:
        # In dual_arm0, mass was set to 1.0kg with the note:
        # "# [수정] 1.0kg은 너무 무거워서 들고 이동할 때 놓침. 0.2kg으로 경량화."
        # Reduced to 0.1kg per user request for even lighter handling.
        if hasattr(self.scene.object.spawn, "mass_props") and self.scene.object.spawn.mass_props is not None:
            self.scene.object.spawn.mass_props.mass = 0.1

        # 3. Cuboid Size: 4cm x 4cm x 25cm
        # Enlarged from 3cm to 4cm (0.04m x 0.04m x 0.25m) for larger contact area and stronger grasp.
        # Initial center height is set to half thickness: Z = 0.020m.
        if hasattr(self.scene.object, "spawn") and hasattr(self.scene.object.spawn, "size"):
            self.scene.object.spawn.size = (0.04, 0.04, 0.25)
        if hasattr(self.scene.object, "init_state") and hasattr(self.scene.object.init_state, "pos"):
            self.scene.object.init_state.pos = (0.5, 0.0, 0.020)

        # 4. Anti-Slip High Friction Physics Material:
        # Static and dynamic friction set to 3.0 with friction_combine_mode="max"
        # to ensure rock-solid non-slip contact with Franka finger pads.
        if hasattr(self.scene.object.spawn, "physics_material"):
            self.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=3.0,
                friction_combine_mode="max",
                restitution=0.0,
                restitution_combine_mode="min",
            )

        # 5. Set target base position to (0.5, 0.5) and update random offset range (Final X,Y: 0.3~0.7)
        if hasattr(self.scene, "target") and hasattr(self.scene.target, "init_state"):
            self.scene.target.init_state.pos = (0.5, 0.5, 0.001)
            if hasattr(self.scene.target, "spawn") and hasattr(self.scene.target.spawn, "radius"):
                self.scene.target.spawn.radius = 0.125

        if hasattr(self, "events") and hasattr(self.events, "reset_target"):
            self.events.reset_target.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0)}
```

### [File: `dataset/il_dataset.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch Dataset loader for Dual Arm Imitation Learning.

Supports HDF5 datasets generated by collect_demos.py and handles:
- Normalization (mean/std or min/max)
- Single-step slicing for standard BC
- Sequence / History slicing for RNN BC
- Horizon Chunking for Diffusion Policy (To, Tp) and ACT (chunk_size)
"""

from __future__ import annotations

import os
import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DualArmDataset(Dataset):
    """Dataset class for dual arm demonstrations stored in HDF5."""

    def __init__(
        self,
        dataset_path: str,
        algo: str = "bc",
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        stats: dict | None = None,
        normalize: bool = True,
    ):
        """Initialize the dataset.

        Args:
            dataset_path: Path to the HDF5 file containing 'data/demo_x'.
            algo: "bc", "diffusion", or "act".
            pred_horizon: Future action sequence length (for diffusion / act).
            obs_horizon: Past observation sequence length (for diffusion).
            stats: Precomputed normalization stats (dict with 'obs_mean', 'obs_std', etc.).
            normalize: Whether to normalize obs and actions to zero mean / unit var.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.algo = algo.lower()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.normalize = normalize

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load all trajectories into memory for high-throughput GPU training
        self.demos_obs = []
        self.demos_act = []
        self.indices = []  # Maps flat dataset index to (demo_idx, timestep)

        with h5py.File(dataset_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

            for demo_idx, key in enumerate(demo_keys):
                obs = np.array(data_grp[key]["obs"], dtype=np.float32)
                act = np.array(data_grp[key]["actions"], dtype=np.float32)
                ep_len = len(act)

                self.demos_obs.append(obs)
                self.demos_act.append(act)

                for t in range(ep_len):
                    self.indices.append((demo_idx, t))

        self.num_samples = len(self.indices)
        print(f"[Dataset] Loaded {len(self.demos_obs)} demos with {self.num_samples} total transitions.")

        # Compute or apply normalization statistics
        all_obs = np.concatenate(self.demos_obs, axis=0)
        all_act = np.concatenate(self.demos_act, axis=0)

        self.obs_dim = all_obs.shape[1]
        self.act_dim = all_act.shape[1]

        if stats is None:
            self.stats = {
                "obs_mean": np.mean(all_obs, axis=0),
                "obs_std": np.std(all_obs, axis=0) + 1e-6,
                "act_mean": np.mean(all_act, axis=0),
                "act_std": np.std(all_act, axis=0) + 1e-6,
                "act_min": np.min(all_act, axis=0),
                "act_max": np.max(all_act, axis=0),
            }
        else:
            self.stats = stats

    def save_stats(self, save_path: str):
        """Save normalization statistics to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"[Dataset] Normalization statistics saved to {save_path}")

    @staticmethod
    def load_stats(stats_path: str) -> dict:
        """Load normalization statistics from disk."""
        with open(stats_path, "rb") as f:
            return pickle.load(f)

    def normalize_obs(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Normalize observation."""
        if not self.normalize:
            return obs
        if isinstance(obs, torch.Tensor):
            mean = torch.as_tensor(self.stats["obs_mean"], device=obs.device, dtype=obs.dtype)
            std = torch.as_tensor(self.stats["obs_std"], device=obs.device, dtype=obs.dtype)
            return (obs - mean) / std
        return (obs - self.stats["obs_mean"]) / self.stats["obs_std"]

    def unnormalize_actions(self, act: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Denormalize action back to environment space."""
        if not self.normalize:
            return act
        if isinstance(act, torch.Tensor):
            mean = torch.as_tensor(self.stats["act_mean"], device=act.device, dtype=act.dtype)
            std = torch.as_tensor(self.stats["act_std"], device=act.device, dtype=act.dtype)
            return act * std + mean
        return act * self.stats["act_std"] + self.stats["act_mean"]

    def normalize_actions(self, act: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Normalize action."""
        if not self.normalize:
            return act
        if isinstance(act, torch.Tensor):
            mean = torch.as_tensor(self.stats["act_mean"], device=act.device, dtype=act.dtype)
            std = torch.as_tensor(self.stats["act_std"], device=act.device, dtype=act.dtype)
            return (act - mean) / std
        return (act - self.stats["act_mean"]) / self.stats["act_std"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_idx, t = self.indices[idx]
        demo_obs = self.demos_obs[demo_idx]
        demo_act = self.demos_act[demo_idx]
        ep_len = len(demo_act)

        if self.algo == "bc":
            # Standard single-step Behavior Cloning
            obs = demo_obs[t]
            act = demo_act[t]

            if self.normalize:
                obs = (obs - self.stats["obs_mean"]) / self.stats["obs_std"]
                act = (act - self.stats["act_mean"]) / self.stats["act_std"]

            return {
                "obs": torch.tensor(obs, dtype=torch.float32),
                "action": torch.tensor(act, dtype=torch.float32),
            }

        elif self.algo == "diffusion":
            # Diffusion Policy: To past observations, Tp future actions
            # Slice observations [t - obs_horizon + 1 : t + 1] with left padding
            start_t = max(0, t - self.obs_horizon + 1)
            obs_seq = demo_obs[start_t : t + 1]
            if len(obs_seq) < self.obs_horizon:
                # Pad repeating the earliest observation
                pad_len = self.obs_horizon - len(obs_seq)
                pad = np.repeat(obs_seq[:1], pad_len, axis=0)
                obs_seq = np.concatenate([pad, obs_seq], axis=0)

            # Slice actions [t : t + pred_horizon] with right padding
            end_t = min(ep_len, t + self.pred_horizon)
            act_seq = demo_act[t:end_t]
            if len(act_seq) < self.pred_horizon:
                # Pad repeating the last action
                pad_len = self.pred_horizon - len(act_seq)
                pad = np.repeat(act_seq[-1:], pad_len, axis=0)
                act_seq = np.concatenate([act_seq, pad], axis=0)

            if self.normalize:
                obs_seq = (obs_seq - self.stats["obs_mean"]) / self.stats["obs_std"]
                act_seq = (act_seq - self.stats["act_mean"]) / self.stats["act_std"]

            return {
                "obs": torch.tensor(obs_seq, dtype=torch.float32),      # (To, D_obs)
                "action": torch.tensor(act_seq, dtype=torch.float32),  # (Tp, D_act)
            }

        elif self.algo == "act":
            # ACT: Current observation + Action chunk of size pred_horizon
            obs = demo_obs[t]

            end_t = min(ep_len, t + self.pred_horizon)
            act_seq = demo_act[t:end_t]
            if len(act_seq) < self.pred_horizon:
                pad_len = self.pred_horizon - len(act_seq)
                pad = np.repeat(act_seq[-1:], pad_len, axis=0)
                act_seq = np.concatenate([act_seq, pad], axis=0)

            # Optional action padding mask (1 for valid action, 0 for padded)
            is_pad = np.zeros(self.pred_horizon, dtype=np.float32)
            actual_len = min(self.pred_horizon, ep_len - t)
            is_pad[actual_len:] = 1.0

            if self.normalize:
                obs = (obs - self.stats["obs_mean"]) / self.stats["obs_std"]
                act_seq = (act_seq - self.stats["act_mean"]) / self.stats["act_std"]

            return {
                "obs": torch.tensor(obs, dtype=torch.float32),          # (D_obs,)
                "action": torch.tensor(act_seq, dtype=torch.float32),  # (chunk_size, D_act)
                "is_pad": torch.tensor(is_pad, dtype=torch.float32),   # (chunk_size,)
            }

        else:
            raise ValueError(f"Unsupported algo: {self.algo}")
```

### [File: `models/__init__.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation Learning Model Zoo."""

from .bc.bc_policy import MLPBCPolicy, RNNBCPolicy
from .diffusion.diffusion_policy import DiffusionPolicy
from .act.act_policy import ACTPolicy

__all__ = ["MLPBCPolicy", "RNNBCPolicy", "DiffusionPolicy", "ACTPolicy"]
```

### [File: `models/act/__init__.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

from .act_policy import ACTPolicy

__all__ = ["ACTPolicy"]
```

### [File: `models/act/act_policy.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Action Chunking with Transformers (ACT) Policy (Zhao et al. 2023)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import ACTTransformer


class ACTPolicy(nn.Module):
    """CVAE + Transformer Action Chunking Policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        chunk_size: int = 24,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        dropout: float = 0.1,
        temporal_ensembling: bool = True,
        temporal_ensemble_coeff: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.d_model = d_model
        self.temporal_ensembling = temporal_ensembling
        self.temporal_ensemble_coeff = temporal_ensemble_coeff

        # --- CVAE Encoder (Obs + Action Chunk -> Latent z) ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.obs_proj_enc = nn.Linear(obs_dim, d_model)
        self.act_proj_enc = nn.Linear(act_dim, d_model)

        cvae_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_encoder_layer, num_layers=2)
        self.latent_proj = nn.Linear(d_model, latent_dim * 2)  # mu and logvar

        # --- Decoder (Obs + Latent z -> Action Chunk) ---
        self.obs_proj_dec = nn.Linear(obs_dim, d_model)
        self.latent_proj_dec = nn.Linear(latent_dim, d_model)

        # Learnable action sequence query tokens
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model))

        # Core Transformer
        self.transformer = ACTTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output projection to action space
        self.action_head = nn.Linear(d_model, act_dim)

        # Temporal Ensembling buffer for inference
        self.reset_temporal_ensemble()

    def reset_temporal_ensemble(self):
        """Reset the temporal ensembling action buffer."""
        self.ensemble_buffer = {}  # maps future timestep t -> list of (predicted_action, weight)

    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode demonstration trajectory into latent style distribution.

        Args:
            obs: (B, obs_dim)
            actions: (B, chunk_size, act_dim)
        Returns:
            mu, logvar: (B, latent_dim)
        """
        batch_size = obs.shape[0]

        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        obs_tok = self.obs_proj_enc(obs).unsqueeze(1)    # (B, 1, d_model)
        act_tok = self.act_proj_enc(actions)             # (B, chunk_size, d_model)

        seq = torch.cat([cls, obs_tok, act_tok], dim=1)  # (B, 2 + chunk_size, d_model)
        enc_out = self.cvae_encoder(seq)

        cls_out = enc_out[:, 0, :]                       # (B, d_model)
        latent_params = self.latent_proj(cls_out)
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent z ~ N(mu, exp(logvar))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass.

        Args:
            obs: (B, obs_dim)
            actions: (B, chunk_size, act_dim) when training, None during inference
        Returns:
            pred_actions: (B, chunk_size, act_dim)
            mu, logvar: (B, latent_dim) or (None, None)
        """
        batch_size = obs.shape[0]

        if actions is not None:
            # Training mode: sample z from encoder
            mu, logvar = self.encode(obs, actions)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference mode: set latent z to prior mean (0)
            mu, logvar = None, None
            z = torch.zeros((batch_size, self.latent_dim), device=obs.device, dtype=obs.dtype)

        # Prepare Decoder inputs: Condition tokens [obs, z]
        obs_tok = self.obs_proj_dec(obs).unsqueeze(1)       # (B, 1, d_model)
        z_tok = self.latent_proj_dec(z).unsqueeze(1)        # (B, 1, d_model)
        context = torch.cat([obs_tok, z_tok], dim=1)        # (B, 2, d_model)

        queries = self.action_queries.expand(batch_size, -1, -1)  # (B, chunk_size, d_model)

        dec_out = self.transformer(src=context, tgt=queries)
        pred_actions = self.action_head(dec_out)

        return pred_actions, mu, logvar

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute ACT loss: L1/MSE reconstruction + KL divergence."""
        obs = batch["obs"]
        target_actions = batch["action"]

        pred_actions, mu, logvar = self.forward(obs, target_actions)

        # L1 / L2 action reconstruction loss
        recon_loss = F.l1_loss(pred_actions, target_actions)

        # KL divergence: D_KL(q(z|o,a) || p(z)) where p(z) ~ N(0, I)
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    @torch.no_grad()
    def predict_action_step(
        self,
        obs: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """Closed-loop inference with optional Temporal Ensembling.

        Args:
            obs: (obs_dim,) or (1, obs_dim)
            current_step: Current environment timestep (integer)
        Returns:
            action: (act_dim,) action for the immediate current step
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Predict fresh chunk of actions for the next chunk_size steps
        pred_chunk, _, _ = self.forward(obs, actions=None)  # (1, chunk_size, act_dim)
        chunk_actions = pred_chunk.squeeze(0)                # (chunk_size, act_dim)

        if not self.temporal_ensembling:
            # Without ensembling: just return the first action
            return chunk_actions[0]

        # Temporal Ensembling: Add predicted trajectory to buffer
        k = self.temporal_ensemble_coeff
        for i in range(self.chunk_size):
            t = current_step + i
            # Exponential decay weight: w = exp(-k * i)
            weight = np.exp(-k * i)
            action_i = chunk_actions[i].detach()

            if t not in self.ensemble_buffer:
                self.ensemble_buffer[t] = []
            self.ensemble_buffer[t].append((action_i, weight))

        # Compute weighted average for the current step
        entries = self.ensemble_buffer.pop(current_step, None)
        if entries is None:
            return chunk_actions[0]

        total_weight = sum(w for _, w in entries)
        blended_action = sum(act * (w / total_weight) for act, w in entries)

        # Clean old timesteps from buffer to prevent memory growth
        old_keys = [k for k in self.ensemble_buffer.keys() if k < current_step]
        for k_old in old_keys:
            del self.ensemble_buffer[k_old]

        return blended_action
```

### [File: `models/act/transformer.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Transformer Backbone for ACT (Action Chunking with Transformers)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal 1D Positional Embedding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1)]


class ACTTransformer(nn.Module):
    """Transformer Encoder-Decoder architecture for ACT."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.pos_emb = SinusoidalPositionEmbedding(d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, src_len, d_model) - Context tokens (obs + style latent z)
            tgt: (B, tgt_len, d_model) - Query tokens for action sequence
            src_key_padding_mask: (B, src_len)
        Returns:
            out: (B, tgt_len, d_model)
        """
        src = self.pos_emb(src)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        tgt = self.pos_emb(tgt)
        out = self.decoder(tgt, memory)
        return out
```

### [File: `models/bc/__init__.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

from .bc_policy import MLPBCPolicy, RNNBCPolicy

__all__ = ["MLPBCPolicy", "RNNBCPolicy"]
```

### [File: `models/bc/bc_policy.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior Cloning (BC) Policies.

Includes MLP-based and RNN-based policy networks for state-based imitation learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBCPolicy(nn.Module):
    """Multi-Layer Perceptron Behavior Cloning Policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [512, 512, 256],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        act_fn = nn.ReLU if activation.lower() == "relu" else nn.GELU

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict action given observation.

        Args:
            obs: (B, obs_dim) or (obs_dim,)
        Returns:
            action: (B, act_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            return self.net(obs).squeeze(0)
        return self.net(obs)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute training loss on a batch.

        Args:
            batch: dict with "obs" (B, obs_dim) and "action" (B, act_dim)
        """
        pred_act = self.forward(batch["obs"])
        loss = F.mse_loss(pred_act, batch["action"])
        return {"loss": loss, "mse": loss.detach()}


class RNNBCPolicy(nn.Module):
    """Recurrent (LSTM) Behavior Cloning Policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

    def forward(self, obs_seq: torch.Tensor, hidden=None) -> tuple[torch.Tensor, tuple]:
        """Predict action given observation sequence.

        Args:
            obs_seq: (B, seq_len, obs_dim)
            hidden: Optional (h_0, c_0)
        Returns:
            action: (B, act_dim) for the final timestep
            hidden: updated (h_n, c_n)
        """
        out, hidden = self.lstm(obs_seq, hidden)
        last_out = out[:, -1, :]  # Take output of last timestep
        action = self.head(last_out)
        return action, hidden

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = batch["obs"]
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)
        pred_act, _ = self.forward(obs)
        loss = F.mse_loss(pred_act, batch["action"])
        return {"loss": loss, "mse": loss.detach()}
```

### [File: `models/diffusion/__init__.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

from .diffusion_policy import DiffusionPolicy
from .unet1d import ConditionalUnet1D

__all__ = ["DiffusionPolicy", "ConditionalUnet1D"]
```

### [File: `models/diffusion/diffusion_policy.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion Policy (Chi et al. 2023) implementation for State-based Dual Arm Manipulation."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet1d import ConditionalUnet1D


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in Nichol and Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionPolicy(nn.Module):
    """Diffusion Policy model supporting training and fast closed-loop inference."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.num_train_timesteps = num_train_timesteps

        # Global conditioning dimension = obs_horizon * obs_dim
        self.cond_dim = obs_horizon * obs_dim

        # 1D Temporal UNet backbone
        self.model = ConditionalUnet1D(
            act_dim=act_dim,
            cond_dim=self.cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        # Setup diffusion noise scheduler constants
        if beta_schedule == "squaredcos_cap_v2":
            betas = cosine_beta_schedule(num_train_timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers so they are automatically moved with model.to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8),
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward diffusion training step.

        Args:
            batch: dict with:
                "obs": (B, obs_horizon, obs_dim)
                "action": (B, pred_horizon, act_dim)
        """
        obs_seq = batch["obs"]
        act_seq = batch["action"]
        batch_size = act_seq.shape[0]

        # Flatten observation history into single conditioning vector
        global_cond = obs_seq.reshape(batch_size, -1)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=act_seq.device, dtype=torch.long
        )

        # Sample noise
        noise = torch.randn_like(act_seq)

        # Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        noisy_actions = sqrt_alpha * act_seq + sqrt_one_minus_alpha * noise

        # Predict noise
        pred_noise = self.model(noisy_actions, timesteps, global_cond)

        # Loss: MSE between true noise and predicted noise
        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss, "mse": loss.detach()}

    @torch.no_grad()
    def predict_action(
        self,
        obs_seq: torch.Tensor,
        num_inference_steps: int = 15,
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """Sample actions using reverse diffusion process.

        Args:
            obs_seq: (B, obs_horizon, obs_dim) or (obs_horizon, obs_dim)
            num_inference_steps: Number of sampling steps (fast DDIM acceleration)
            use_ddim: Whether to use DDIM (fast) or standard DDPM
        Returns:
            actions: (B, pred_horizon, act_dim)
        """
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        batch_size = obs_seq.shape[0]
        device = obs_seq.device

        global_cond = obs_seq.reshape(batch_size, -1)

        # Start from pure Gaussian noise
        x = torch.randn((batch_size, self.pred_horizon, self.act_dim), device=device)

        if use_ddim and num_inference_steps < self.num_train_timesteps:
            # DDIM accelerated sampling
            timesteps = torch.linspace(
                self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
            )

            for i in range(len(timesteps)):
                t = timesteps[i].repeat(batch_size)
                pred_noise = self.model(x, t, global_cond)

                alpha_bar = self.alphas_cumprod[t].view(batch_size, 1, 1)
                t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=device)
                alpha_bar_prev = self.alphas_cumprod[t_prev.repeat(batch_size)].view(batch_size, 1, 1)

                # Predict x_0 from current noise prediction
                pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

                if i < len(timesteps) - 1:
                    dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                else:
                    x = pred_x0
        else:
            # Standard DDPM reverse process
            for t_idx in reversed(range(self.num_train_timesteps)):
                t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
                pred_noise = self.model(x, t, global_cond)

                beta = self.betas[t].view(batch_size, 1, 1)
                alpha = self.alphas[t].view(batch_size, 1, 1)
                alpha_bar = self.alphas_cumprod[t].view(batch_size, 1, 1)

                # DDPM mean
                mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise)

                if t_idx > 0:
                    var = self.posterior_variance[t].view(batch_size, 1, 1)
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(var) * noise
                else:
                    x = mean

        return x
```

### [File: `models/diffusion/unet1d.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""1D Temporal Conditional UNet for Diffusion Policy (Chi et al. 2023)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning on timesteps and global observations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)

        # FiLM generator: predicts scale and bias from condition embedding
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        """
        out = self.conv1(x)

        # FiLM scale & bias
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, 2 * out_channels, 1)
        scale, bias = torch.chunk(cond_embed, 2, dim=1)

        out = out * (1.0 + scale) + bias
        out = self.conv2(out)

        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D Temporal UNet conditional on observation history and diffusion step."""

    def __init__(
        self,
        act_dim: int,
        cond_dim: int,
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.cond_dim = cond_dim

        # Timestep embedding MLP
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Combined condition dimension (timestep + observation condition)
        total_cond_dim = diffusion_step_embed_dim + cond_dim

        # Encoder (Downsampling)
        all_dims = [act_dim] + list(down_dims)
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims)):
            in_d = all_dims[i]
            out_d = all_dims[i + 1]
            self.down_blocks.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(in_d, out_d, total_cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1D(out_d, out_d, total_cond_dim, kernel_size, n_groups),
                    nn.Conv1d(out_d, out_d, kernel_size=3, stride=2, padding=1),  # Downsample
                ])
            )

        # Mid block
        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(mid_dim, mid_dim, total_cond_dim, kernel_size, n_groups)
        self.mid_block2 = ConditionalResidualBlock1D(mid_dim, mid_dim, total_cond_dim, kernel_size, n_groups)

        # Decoder (Upsampling)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(down_dims))):
            dim_in = down_dims[i]
            dim_out = down_dims[i - 1] if i > 0 else down_dims[0]
            self.up_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose1d(dim_in, dim_in, kernel_size=4, stride=2, padding=1),  # Upsample
                    ConditionalResidualBlock1D(dim_in * 2, dim_out, total_cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1D(dim_out, dim_out, total_cond_dim, kernel_size, n_groups),
                ])
            )

        # Final projection to action dimension
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(down_dims[0], act_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, T, act_dim) noisy actions
            timestep: (B,) diffusion timesteps
            global_cond: (B, cond_dim) observation embedding
        Returns:
            predicted noise or action: (B, T, act_dim)
        """
        # Permute for 1D convolution: (B, act_dim, T)
        x = sample.transpose(1, 2)
        orig_len = x.shape[-1]

        # Compute combined condition vector
        time_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([time_emb, global_cond], dim=-1)

        skips = []
        # Downward pass
        for res1, res2, downsample in self.down_blocks:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        # Mid pass
        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        # Upward pass
        for upsample, res1, res2 in self.up_blocks:
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        # Final projection
        x = self.final_conv(x)

        if x.shape[-1] != orig_len:
            x = F.interpolate(x, size=orig_len, mode="linear", align_corners=False)

        # Permute back: (B, T, act_dim)
        return x.transpose(1, 2)

```

### [File: `requirements.txt`]
```txt
# Dependencies for Imitation Learning in Isaac Lab
h5py>=3.8.0
pyyaml>=6.0
torch>=2.0.0
numpy>=1.22.0
scipy>=1.9.0
tqdm>=4.65.0
einops>=0.7.0
```

### [File: `scripts/eval.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop Evaluation Script for Dual Arm Imitation Learning Policies.

Tests trained BC, Diffusion Policy, or ACT models in the Isaac Sim simulation
and logs task success metrics.

Usage:
    python scripts/eval.py --algo=diffusion --num_episodes=10
    python scripts/eval.py --algo=act --num_episodes=10
    python scripts/eval.py --algo=bc --num_episodes=10
"""

import argparse
import collections
import os
import sys
import torch

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Evaluate trained imitation learning policies.")
parser.add_argument("--algo", type=str, default="diffusion", choices=["bc", "diffusion", "act"])
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt).")
parser.add_argument("--stats", type=str, default=None, help="Path to normalization stats.pkl.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps per episode (~23 seconds at 30Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg
from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy


def main():
    algo = args_cli.algo.lower()
    ckpt_path = args_cli.checkpoint or os.path.join(PROJECT_ROOT, "checkpoints", algo, "best_model.pt")
    stats_path = args_cli.stats or os.path.join(PROJECT_ROOT, "checkpoints", algo, "stats.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    print("=" * 60)
    print(f" Evaluating Policy: {algo.upper()}")
    print(f" Checkpoint : {ckpt_path}")
    print(f" Stats      : {stats_path}")
    print("=" * 60)

    # Load Normalization Stats
    stats = DualArmDataset.load_stats(stats_path)
    obs_mean = torch.tensor(stats["obs_mean"], device=args_cli.device, dtype=torch.float32)
    obs_std = torch.tensor(stats["obs_std"], device=args_cli.device, dtype=torch.float32)
    act_mean = torch.tensor(stats["act_mean"], device=args_cli.device, dtype=torch.float32)
    act_std = torch.tensor(stats["act_std"], device=args_cli.device, dtype=torch.float32)

    # Load Checkpoint & Instantiate Model
    checkpoint = torch.load(ckpt_path, map_location=args_cli.device)
    cfg = checkpoint.get("config", {})
    obs_dim = checkpoint.get("obs_dim", len(obs_mean))
    act_dim = checkpoint.get("act_dim", len(act_mean))

    if algo == "bc":
        model = MLPBCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=cfg.get("hidden_dims", [512, 512, 256]))
    elif algo == "diffusion":
        model = DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
        )
    elif algo == "act":
        model = ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            latent_dim=cfg.get("latent_dim", 32),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args_cli.device)
    model.eval()

    # Create Environment
    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

    obs_horizon = cfg.get("obs_horizon", 2)
    act_horizon = cfg.get("act_horizon", 8)

    num_success = 0
    total_episodes = args_cli.num_episodes

    print(f"\nStarting {total_episodes} evaluation episodes...\n")

    for ep_idx in range(1, total_episodes + 1):
        obs, _ = env.reset()
        if hasattr(model, "reset_temporal_ensemble"):
            model.reset_temporal_ensemble()

        # Observation queue for Diffusion Policy
        obs_queue = collections.deque(maxlen=obs_horizon)
        # Action queue for multi-step receding horizon execution
        action_queue = collections.deque()

        ep_success = False

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            # Extract and normalize observation
            raw_obs = obs["policy"].squeeze(0)  # (obs_dim,)
            norm_obs = (raw_obs - obs_mean) / obs_std

            with torch.no_grad():
                if algo == "bc":
                    norm_action = model(norm_obs)
                    action = norm_action * act_std + act_mean

                elif algo == "diffusion":
                    obs_queue.append(norm_obs)
                    while len(obs_queue) < obs_horizon:
                        obs_queue.append(norm_obs)

                    if len(action_queue) == 0:
                        obs_tensor = torch.stack(list(obs_queue), dim=0).unsqueeze(0)  # (1, To, obs_dim)
                        pred_act_chunk = model.predict_action(obs_tensor, num_inference_steps=15)
                        pred_act_chunk = pred_act_chunk.squeeze(0)  # (Tp, act_dim)

                        # Enqueue first act_horizon actions
                        for a_idx in range(min(act_horizon, len(pred_act_chunk))):
                            act_unnorm = pred_act_chunk[a_idx] * act_std + act_mean
                            action_queue.append(act_unnorm)

                    action = action_queue.popleft()

                elif algo == "act":
                    norm_action = model.predict_action_step(norm_obs, current_step=step)
                    action = norm_action * act_std + act_mean

            # Step environment: action shape (1, act_dim)
            action_step = action.unsqueeze(0)
            obs, reward, terminated, truncated, _ = env.step(action_step)

            # Check task success: baton placed near red target
            obj_pos = env.scene["object"].data.root_pos_w.squeeze(0)
            target_pos = env.scene["target"].data.root_pos_w.squeeze(0)
            dist_to_target = torch.norm(obj_pos[:2] - target_pos[:2]).item()

            if dist_to_target < 0.12 and obj_pos[2].item() < 0.05:
                ep_success = True
                print(f"[Episode {ep_idx}] SUCCESS! Placed at target (dist: {dist_to_target:.3f}m, step: {step})")
                break

            if terminated.item() or truncated.item():
                break

        if ep_success:
            num_success += 1
        else:
            print(f"[Episode {ep_idx}] Failed or timed out.")

    success_rate = (num_success / total_episodes) * 100.0
    print("\n" + "=" * 60)
    print(f" Evaluation Completed: {algo.upper()}")
    print(f" Total Episodes : {total_episodes}")
    print(f" Successes      : {num_success}")
    print(f" Success Rate   : {success_rate:.1f}%")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
```

### [File: `scripts/generate_scripted_demos.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Automatic Demonstration Generator for Dual Arm Handover Task.

Generates high-quality, optimal demonstration trajectories automatically using
Waypoint State Machine and Differential Inverse Kinematics.

Features:
- 100% automated: No human teleoperation or manual effort required.
- Robust Closed-Loop IK: DLS position control with joint limit protection.
- Exact Action Inversion: Automatically accounts for Isaac Lab's default joint offsets and action scales.
- Direct export to HDF5: Ready for immediate training with BC, Diffusion Policy, or ACT.

Usage:
    # Fast Headless generation (recommended):
    python scripts/generate_scripted_demos.py --num_demos=500 --headless

    # GUI mode (visualize robot motion):
    python scripts/generate_scripted_demos.py --num_demos=10
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate scripted demonstrations for Dual Arm Handover.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demonstrations to generate.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to output HDF5 dataset.",
)
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps before episode timeout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg


def solve_pose_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: float = 0.05,
    q_current: torch.Tensor | None = None,
    q_nominal: torch.Tensor | None = None,
    k_null: float = 0.5,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics with 7-DOF nullspace elbow flare projection.

    Args:
        jacobian: Full Jacobian of shape (1, 6, 7).
        delta_pose: Desired 6D delta [dx, dy, dz, wx, wy, wz] of shape (1, 6).
        damping: Damping factor lambda.
        q_current: Current joint positions of shape (1, 7).
        q_nominal: Desired nominal joint posture for nullspace projection (1, 7).
        k_null: Nullspace projection gain.

    Returns:
        delta_q: Change in joint angles of shape (1, 7).
    """
    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)
    j_pinv = torch.matmul(j.transpose(0, 1), inv_term)  # (7, 6)
    delta_q = torch.matmul(j_pinv, e)  # (7,)

    if q_current is not None and q_nominal is not None:
        eye_n = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
        null_proj = eye_n - torch.matmul(j_pinv, j)  # (7, 7)
        q_err = (q_nominal - q_current).squeeze(0)  # (7,)
        delta_q_null = torch.matmul(null_proj, k_null * q_err)  # (7,)
        delta_q = delta_q + delta_q_null

    return delta_q.unsqueeze(0)



def get_tcp_jacobian(jacobian: torch.Tensor, wrist_pos: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
    r = tcp_pos - wrist_pos  # (N, 3)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zero = torch.zeros_like(rx)
    r_skew = torch.stack([
        torch.stack([zero, -rz, ry], dim=-1),
        torch.stack([rz, zero, -rx], dim=-1),
        torch.stack([-ry, rx, zero], dim=-1)
    ], dim=1)  # (N, 3, 3)
    j_v = jacobian[:, :3, :]   # (N, 3, 7)
    j_w = jacobian[:, 3:6, :]  # (N, 3, 7)
    j_v_tcp = j_v - torch.bmm(r_skew, j_w)
    return torch.cat([j_v_tcp, j_w], dim=1)  # (N, 6, 7)


def compute_desired_grasp_rot(
    wrist_quat: torch.Tensor,
    cube_z_dir: torch.Tensor,
    gain: float = 0.5,
    max_rot_step: float = 0.04,
    directed: bool = False,
) -> torch.Tensor:
    """Compute angular error vector to align Franka gripper perpendicular to baton.

    Ensures:
    1. Local Z axis (palm normal) points vertically downwards [0, 0, -1].
    2. Local X axis (palm lateral) is parallel to the baton longitudinal axis.
    3. Local Y axis (finger opening/closing) is perpendicular to the baton,
       pinching the 3cm thickness cleanly.

    Args:
        wrist_quat: Hand orientation quaternion (1, 4) in (w, x, y, z).
        cube_z_dir: Baton length direction unit vector (1, 3).
        gain: Proportional convergence gain.
        max_rot_step: Maximum angular velocity step in radians.
        directed: If True, do not use 180-degree symmetry. Align exactly to cube_z_dir.

    Returns:
        e_rot_step: (1, 3) angular delta vector [wx, wy, wz].
    """
    w = wrist_quat[:, 0]
    x = wrist_quat[:, 1]
    y = wrist_quat[:, 2]
    z = wrist_quat[:, 3]

    # Current rotation matrix columns in world coordinates
    x_curr = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)

    y_curr = torch.stack([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dim=-1)

    z_curr = torch.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)

    # Desired Z axis: straight down
    z_des = torch.tensor([[0.0, 0.0, -1.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    # Baton horizontal projection
    baton_xy = cube_z_dir.clone()
    baton_xy[:, 2] = 0.0
    norm_xy = torch.norm(baton_xy, dim=-1, keepdim=True)
    if norm_xy.item() > 1e-4:
        x_cand = baton_xy / norm_xy
    else:
        x_cand = torch.tensor([[1.0, 0.0, 0.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    if directed:
        x_des = x_cand
    else:
        # Align with whichever 180-degree symmetric direction is closer to current x_curr
        dot = torch.sum(x_curr * x_cand, dim=-1, keepdim=True)
        sign = torch.sign(dot + 1e-6)
        x_des = sign * x_cand

    # Desired Y axis = Z_des x X_des (finger pinching axis perpendicular to baton)
    y_des = torch.cross(z_des, x_des, dim=-1)

    # Orientation error via Lie algebra so(3) cross-product error
    e_rot = 0.5 * (
        torch.cross(x_curr, x_des, dim=-1) +
        torch.cross(y_curr, y_des, dim=-1) +
        torch.cross(z_curr, z_des, dim=-1)
    )

    # Apply gain and clamp
    e_rot_step = e_rot * gain
    err_norm = torch.norm(e_rot_step)
    if err_norm > max_rot_step:
        e_rot_step = e_rot_step * (max_rot_step / err_norm)

    return e_rot_step


def compute_tcp(wrist_pos: torch.Tensor, wrist_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Franka TCP position and Z-axis direction from wrist pose."""
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    return tcp_pos, z_dir


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list, init_states: dict = None, obj_traj: list = None, joint_traj: list = None):
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)
    mode = "a" if os.path.exists(hdf5_path) else "w"
    with h5py.File(hdf5_path, mode) as f:
        data_group = f.require_group("data")
        demo_group = data_group.create_group(f"demo_{ep_idx}")

        obs_array = np.array(observations, dtype=np.float32)
        act_array = np.array(actions, dtype=np.float32)
        rew_array = np.array(rewards, dtype=np.float32)

        demo_group.create_dataset("obs", data=obs_array, compression="gzip")
        demo_group.create_dataset("actions", data=act_array, compression="gzip")
        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.attrs["num_samples"] = len(act_array)
        if init_states:
            for k, v in init_states.items():
                demo_group.create_dataset(k, data=v)
                
        if obj_traj is not None:
            demo_group.create_dataset("object_poses", data=np.array(obj_traj, dtype=np.float32), compression="gzip")
        if joint_traj is not None:
            demo_group.create_dataset("robot_joint_poses", data=np.array(joint_traj, dtype=np.float32), compression="gzip")

        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")


# State Machine Phase Constants
PHASE_NAMES = [
    "INIT",
    "RIGHT_HOVER",
    "RIGHT_DESCEND",
    "RIGHT_GRASP",
    "RIGHT_LIFT",
    "RIGHT_HANDOVER",
    "LEFT_APPROACH",
    "LEFT_GRASP",
    "RIGHT_RELEASE",
    "RIGHT_LIFT_CLEAR",
    "RIGHT_RETREAT",
    "LEFT_HOVER_TARGET",
    "LEFT_LOWER_TARGET",
    "LEFT_RELEASE",
    "LEFT_RETREAT",
    "SUCCESS",
]
PHASE_INIT = 0
PHASE_RIGHT_HOVER = 1
PHASE_RIGHT_DESCEND = 2
PHASE_RIGHT_GRASP = 3
PHASE_RIGHT_LIFT = 4
PHASE_RIGHT_HANDOVER = 5
PHASE_LEFT_APPROACH = 6
PHASE_LEFT_GRASP = 7
PHASE_RIGHT_RELEASE = 8
PHASE_RIGHT_LIFT_CLEAR = 9
PHASE_RIGHT_RETREAT = 10
PHASE_LEFT_HOVER_TARGET = 11
PHASE_LEFT_LOWER_TARGET = 12
PHASE_LEFT_RELEASE = 13
PHASE_LEFT_RETREAT = 14
PHASE_SUCCESS = 15


def main():
    cfg = DualArmILEnvCfg()
    cfg.sim.device = args_cli.device

    print("[Scripted Demo] Initializing Isaac Lab Environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]
    target = env.scene["target"]

    # Body Indices
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Joint Indices in Articulation
    left_arm_joint_ids, left_arm_names = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, right_arm_names = robot.find_joints("panda_joint[1-7]_0")

    # Gripper finger joint indices (for real-time grip width and fake grasp verification)
    left_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]$")
    right_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]_0")

    # Safe standby postures (arm base and elbow rotated away from center workspace)
    left_standby_joints = robot.data.default_joint_pos[:, left_arm_joint_ids].clone()
    left_standby_joints[:, 0] = 0.60  # +34 deg, outward to +Y quadrant

    right_standby_joints = robot.data.default_joint_pos[:, right_arm_joint_ids].clone()
    right_standby_joints[:, 0] = -0.60  # -34 deg, outward to -Y quadrant

    right_standby_joints[:, 4] += 0.5
    left_standby_joints[:, 4] -= 0.5

    # Nominal joint postures for 7-DOF nullspace elbow flaring
    q_nominal_left = left_standby_joints.clone()
    q_nominal_right = right_standby_joints.clone()

    # Fixed world orientations for stable, jitter-free transport
    fixed_z_down = torch.tensor([[0.0, 0.0, -1.0]], device=args_cli.device)
    fixed_x_along = torch.tensor([[1.0, 0.0, 0.0]], device=args_cli.device)

    # Shift joint IDs for Jacobian lookup (0 for fixed base robot)
    num_base_dofs = getattr(robot, "num_base_dofs", 0)
    jacobi_right_joint_ids = [j + num_base_dofs for j in right_arm_joint_ids]
    jacobi_left_joint_ids = [j + num_base_dofs for j in left_arm_joint_ids]

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]
    print(f"[Scripted Demo] Action Manager 'arm_action' controls {len(arm_term._joint_ids)} joints.")
    print(f"  Scale : {arm_term._scale}")

    # Joint limits for safety clamping
    soft_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    soft_upper = robot.data.soft_joint_pos_limits[:, :, 1]

    # Check existing dataset
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

    print(f"[Scripted Demo] Existing demonstrations: {existing_demos}")
    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    start_time = time.time()
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()

        init_states = {
            "init_object_pos": (env.scene["object"].data.root_pos_w - env.scene.env_origins).clone().squeeze(0).cpu().numpy(),
            "init_object_quat": env.scene["object"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_target_pos": (env.scene["target"].data.root_pos_w - env.scene.env_origins).clone().squeeze(0).cpu().numpy(),
            "init_target_quat": env.scene["target"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_pos": (env.scene["robot"].data.root_pos_w - env.scene.env_origins).clone().squeeze(0).cpu().numpy(),
            "init_robot_quat": env.scene["robot"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_joint_pos": env.scene["robot"].data.joint_pos.clone().squeeze(0).cpu().numpy(),
            "init_robot_joint_vel": env.scene["robot"].data.joint_vel.clone().squeeze(0).cpu().numpy(),
        }

        # Initialize commanded joint targets from the robot's safe standby positions
        commanded_left = left_standby_joints.clone()
        commanded_right = right_standby_joints.clone()

        left_gripper_cmd = 1.0   # Open (+1.0)
        right_gripper_cmd = 1.0  # Open (+1.0)

        phase = PHASE_INIT
        phase_timer = 0
        latched_align_sign = None
        latched_cube_z = None
        target_cube_z = None
        target_cube_z_for_wrist = None
        latched_left_grasp_offset = None

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_obj_traj = []
        ep_joint_traj = []

        print(f"\n>>> Generating Scripted Demo #{collected_count} (Target: {target_count}) <<<")

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            phase_timer += 1

            # Object and Target Poses
            obj_pos = obj.data.root_pos_w.clone()        # (1, 3)
            obj_quat = obj.data.root_quat_w.clone()      # (1, 4)
            target_pos = target.data.root_pos_w.clone()  # (1, 3)

            # Fail-fast: If baton drops during mid-air phases (Handover to Hover Target), abort episode
            if 5 <= phase <= 11 and obj_pos[0, 2].item() < 0.05:
                print(f"  [X] Dropped baton in mid-air at phase {PHASE_NAMES[phase]}! Discarding episode...")
                break

            # Franka Wrist and TCP Poses
            wrist_pos_r = robot.data.body_pos_w[:, right_hand_idx]
            wrist_quat_r = robot.data.body_quat_w[:, right_hand_idx]
            tcp_pos_r, z_dir_r = compute_tcp(wrist_pos_r, wrist_quat_r)

            wrist_pos_l = robot.data.body_pos_w[:, left_hand_idx]
            wrist_quat_l = robot.data.body_quat_w[:, left_hand_idx]
            tcp_pos_l, z_dir_l = compute_tcp(wrist_pos_l, wrist_quat_l)

            # Real-time gripper widths (for fake grasp detection)
            right_gripper_width = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1).item()
            left_gripper_width = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1).item()

            # Baton Geometry: Long axis vector (Cuboid Z-axis in world frame)
            ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
            cube_z = torch.stack([
                2.0 * (ox * oz + ow * oy),
                2.0 * (oy * oz - ow * ox),
                1.0 - 2.0 * (ox * ox + oy * oy),
            ], dim=-1)

            # Latch grasp alignment sign at episode start to prevent sign-flipping glitch
            if latched_align_sign is None:
                latched_align_sign = torch.sign(cube_z[:, 1:2] + 1e-6).clone()
                latched_cube_z = cube_z.clone()
                target_cube_z = torch.tensor([[0.0, latched_align_sign.item(), 0.0]], device=args_cli.device)

            # Optimal Grasp Targets:
            # Distance 0.055m (5.5cm from center) reduces cantilever droop torque by 35%
            # while leaving 11cm of clearance between right and left grippers!
            pick_target = obj_pos - latched_align_sign * 0.055 * latched_cube_z
            if latched_align_sign is not None:
                place_target = obj_pos + latched_align_sign * 0.055 * target_cube_z
            else:
                place_target = obj_pos + latched_align_sign * 0.055 * latched_cube_z

            # Handover zone (between both arms)
            handover_pos = env.scene.env_origins + torch.tensor([[0.30, 0.0, 0.20]], device=args_cli.device)
            # Left arm pre-handover hover position: 20cm in +Y, 12cm above handover zone
            left_wait_pos = handover_pos.clone()
            left_wait_pos[:, 1] += 0.20
            left_wait_pos[:, 2] += 0.12

            delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
            delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
            delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
            delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
            
            k_null_r = 0.05
            k_null_l = 0.05
            damping_r = 0.05
            damping_l = 0.05

            # --- State Machine Logic ---
            if phase == PHASE_INIT:
                right_gripper_cmd = 1.0
                left_gripper_cmd = 1.0
                # Smoothly swing left arm to safe standby posture (+Y quadrant)
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                if phase_timer > 15:
                    phase = PHASE_RIGHT_HOVER
                    phase_timer = 0

            elif phase == PHASE_RIGHT_HOVER:
                # Left arm holds safe standby posture away from center
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                hover_tgt = pick_target.clone()
                hover_tgt[:, 2] = pick_target[:, 2] + 0.08
                err = hover_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(hover_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(hover_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, directed=True)
                if (dist_xy < 0.025 and dist_z < 0.03) or phase_timer > 90:
                    phase = PHASE_RIGHT_DESCEND
                    phase_timer = 0

            elif phase == PHASE_RIGHT_DESCEND:
                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                # Descend to center of cuboid (TCP Z=0.025, to avoid fingertips crashing into table)
                descend_tgt = pick_target.clone()
                descend_tgt[:, 2] = torch.clamp(pick_target[:, 2], min=0.025)
                err = descend_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(descend_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(descend_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                reached = (dist_xy < 0.010 and dist_z < 0.005)
                if reached or phase_timer > 160:
                    if reached:
                        print(f"  [Right Grasp Reached!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                    else:
                        print(f"  [Right Grasp Timeout!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                    phase = PHASE_RIGHT_GRASP
                    phase_timer = 0

            elif phase == PHASE_RIGHT_GRASP:
                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                # Close right gripper tightly around baton
                right_gripper_cmd = -1.0
                if phase_timer > 25:
                    if right_gripper_width > 0.025:
                        print(f"  [Right Grasp Verified!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm")
                        
                        w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                        x_curr = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                        latched_wrist_sign = latched_align_sign.clone()
                        target_cube_z_for_wrist = latched_wrist_sign * target_cube_z
                        
                        phase = PHASE_RIGHT_LIFT
                        phase_timer = 0
                    else:
                        print(f"  [Right Grasp Missed!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm < 25mm. Retrying descend...")
                        right_gripper_cmd = 1.0
                        if phase_timer > 40:
                            phase = PHASE_RIGHT_DESCEND
                            phase_timer = 0

            elif phase == PHASE_RIGHT_LIFT:

                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                # Lift to stable handover height Z = 0.18m with locked horizontal orientation
                lift_tgt = pick_target.clone()
                lift_tgt[:, 2] = 0.18
                err = lift_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.025 or phase_timer > 70:
                    phase = PHASE_RIGHT_HANDOVER
                    phase_timer = 0

            elif phase == PHASE_RIGHT_HANDOVER:
                # Right arm transports baton to handover zone and slowly twists 90 degrees
                # Completely eliminate nullspace and massively increase damping to safely pass through wrist singularity
                k_null_r = 0.0
                damping_r = 0.25
                err = handover_pos - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                # Extremely slow and smooth rotation to prevent inertial shaking of the 25cm baton
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist, gain=0.1, max_rot_step=0.020, directed=True)

                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)

                # Wait until BOTH position and orientation (90-degree twist) are fully reached
                # Orientation is aligned when the right wrist's local X-axis is strictly parallel to target_cube_z_for_wrist
                w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                x_curr_r = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                rot_aligned = (1.0 - torch.sum(x_curr_r * target_cube_z_for_wrist, dim=-1)) < 0.005
                
                if (dist < 0.03 and rot_aligned.item()) or phase_timer > 400:
                    phase = PHASE_LEFT_APPROACH
                    phase_timer = 0

            elif phase == PHASE_LEFT_APPROACH:
                # Right arm holds baton rock-solid at handover zone
                k_null_r = 0.05
                right_gripper_cmd = -1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)

                # Left arm descends: first move horizontally to align above, then straight down
                left_grasp_tgt = place_target.clone()
                left_grasp_tgt[:, 2] = place_target[:, 2] - 0.006
                dist_xy = torch.norm(left_grasp_tgt[:, :2] - tcp_pos_l[:, :2])
                
                if dist_xy > 0.03:
                    curr_tgt = left_grasp_tgt.clone()
                    curr_tgt[:, 2] = left_wait_pos[:, 2]
                else:
                    curr_tgt = left_grasp_tgt.clone()
                    
                err = curr_tgt - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(left_grasp_tgt[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                reached_l = (dist_xy < 0.025 and dist_z < 0.015)
                if reached_l or phase_timer > 160:
                    if reached_l:
                        print(f"  [Left Handover Reached!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                    else:
                        print(f"  [Left Handover Timeout!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                    phase = PHASE_LEFT_GRASP
                    phase_timer = 0

            elif phase == PHASE_LEFT_GRASP:
                k_null_r = 0.05
                # Right arm continues holding
                right_gripper_cmd = -1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
                # Left arm closes gripper
                left_gripper_cmd = -1.0
                if phase_timer > 35:
                    if left_gripper_width > 0.025:
                        print(f"  [Left Grasp Verified!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm")
                        latched_left_grasp_offset = tcp_pos_l - obj_pos
                        phase = PHASE_RIGHT_RELEASE
                        phase_timer = 0
                    else:
                        print(f"  [Left Grasp Missed!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm < 25mm. Retrying...")
                        left_gripper_cmd = 1.0
                        if phase_timer > 50:
                            phase = PHASE_LEFT_APPROACH
                            phase_timer = 0

            elif phase == PHASE_RIGHT_RELEASE:
                k_null_r = 0.05
                # Left arm holds tightly
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                # Right arm opens gripper
                right_gripper_cmd = 1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
                if phase_timer > 15:
                    phase = PHASE_RIGHT_LIFT_CLEAR
                    phase_timer = 0

            elif phase == PHASE_RIGHT_LIFT_CLEAR:
                k_null_r = 0.05
                # Left arm firmly holds baton
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                # Right arm moves -Y (away from left arm) AND UP to avoid collision
                right_gripper_cmd = 1.0
                lift_clear_tgt = handover_pos.clone()
                lift_clear_tgt[:, 1] -= 0.15  # Move -15cm in Y (away from left arm)
                lift_clear_tgt[:, 2] = 0.35   # Lift up to Z=0.35m
                err = lift_clear_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.03 or phase_timer > 60:
                    phase = PHASE_RIGHT_RETREAT
                    phase_timer = 0

            elif phase == PHASE_RIGHT_RETREAT:
                # Right arm retreats backwards towards base to clear the table
                retreat_tgt = lift_clear_tgt.clone()
                retreat_tgt[:, 0] -= 0.15  # Move back 15cm
                err = retreat_tgt - tcp_pos_r
                step_size = min(0.012, torch.norm(err).item())
                delta_r_pos = (err / (torch.norm(err) + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z)
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                right_gripper_cmd = 1.0
                # Right arm smoothly moves to safe right standby posture (-Y quadrant)
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                if phase_timer > 15:
                    phase = PHASE_LEFT_HOVER_TARGET
                    phase_timer = 0

            elif phase == PHASE_LEFT_HOVER_TARGET:
                # Right arm stays safely parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Manual tuning to compensate for physical drop/slip offset (pull back from +X, +Y)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset + manual_offset
                tgt_hover = final_place_target.clone()
                tgt_hover[:, 2] = target_pos[:, 2] + 0.12
                err = tgt_hover - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.016, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                # Locked fixed horizontal orientation to eliminate rotational jitter/slipping
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2])
                dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])
                if (dist_xy < 0.01 and dist_z < 0.015) or phase_timer > 70:
                    phase = PHASE_LEFT_LOWER_TARGET
                    phase_timer = 0

            elif phase == PHASE_LEFT_LOWER_TARGET:
                # Right arm stays safely parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Gently lower baton onto target (Z=0.022m, soft landing 2mm above table surface)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset + manual_offset
                tgt_lower = final_place_target.clone()
                tgt_lower[:, 2] = 0.022
                err = tgt_lower - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(tgt_lower[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.008, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                if (dist < 0.015 or dist_z < 0.006) or phase_timer > 60:
                    phase = PHASE_LEFT_RELEASE
                    phase_timer = 0

            elif phase == PHASE_LEFT_RELEASE:
                # Right arm stays parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Open left gripper to release baton on target
                left_gripper_cmd = 1.0
                if phase_timer > 10:
                    phase = PHASE_LEFT_RETREAT
                    phase_timer = 0

            elif phase == PHASE_LEFT_RETREAT:
                # Right arm stays parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                left_gripper_cmd = 1.0
                # Move left arm up and retreat
                tgt_up = target_pos.clone()
                tgt_up[:, 2] += 0.20
                err = tgt_up - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                if dist < 0.03 or phase_timer > 40:
                    phase = PHASE_SUCCESS
                    phase_timer = 0

            # Real-time 3D state and tracking error monitoring
            if step % 25 == 0:
                dist_pick = torch.norm(pick_target - tcp_pos_r).item() * 1000
                dist_place = torch.norm(place_target - tcp_pos_l).item() * 1000
                print(
                    f"  Step {step:03d} | {PHASE_NAMES[phase]:<18} | "
                    f"Pick:[{pick_target[0,0]:.2f},{pick_target[0,1]:.2f},{pick_target[0,2]:.2f}] | "
                    f"R_TCP:[{tcp_pos_r[0,0]:.2f},{tcp_pos_r[0,1]:.2f},{tcp_pos_r[0,2]:.2f}]({dist_pick:.1f}mm) | "
                    f"Place:[{place_target[0,0]:.2f},{place_target[0,1]:.2f},{place_target[0,2]:.2f}] | "
                    f"L_TCP:[{tcp_pos_l[0,0]:.2f},{tcp_pos_l[0,1]:.2f},{tcp_pos_l[0,2]:.2f}]({dist_place:.1f}mm)"
                )

            # --- 6D Inverse Kinematics with 7-DOF Nullspace Elbow Flare Projection ---
            jacobians = robot.root_physx_view.get_jacobians()

            delta_r_pose = torch.cat([delta_r_pos, delta_r_rot], dim=-1)
            if torch.norm(delta_r_pose) > 1e-5:
                j_r_wrist = jacobians[:, jacobi_right_hand_idx, :6, :][:, :, jacobi_right_joint_ids]
                j_r_tcp = get_tcp_jacobian(j_r_wrist, wrist_pos_r, tcp_pos_r)
                dq_r = solve_pose_dls_ik(
                    j_r_tcp,
                    delta_r_pose,
                    damping=damping_r,
                    q_current=commanded_right,
                    q_nominal=q_nominal_right,
                    k_null=k_null_r,
                )
                dq_r = torch.clamp(dq_r, min=-0.08, max=0.08)
                commanded_right += dq_r
                # Clamp within safe joint limits
                commanded_right = torch.clamp(
                    commanded_right,
                    min=soft_lower[:, right_arm_joint_ids] + 0.02,
                    max=soft_upper[:, right_arm_joint_ids] - 0.02,
                )

            delta_l_pose = torch.cat([delta_l_pos, delta_l_rot], dim=-1)
            if torch.norm(delta_l_pose) > 1e-5:
                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l, tcp_pos_l)
                
                # Dynamic task-space relaxation: Free the wrist orientation during flight to maximize reach
                if phase == PHASE_LEFT_HOVER_TARGET:
                    j_l_tcp = j_l_tcp.clone()
                    j_l_tcp[:, 3:6, :] = 0.0
                    
                dq_l = solve_pose_dls_ik(
                    j_l_tcp,
                    delta_l_pose,
                    damping=damping_l,
                    q_current=commanded_left,
                    q_nominal=q_nominal_left,
                    k_null=k_null_l,
                )
                dq_l = torch.clamp(dq_l, min=-0.08, max=0.08)
                commanded_left += dq_l
                commanded_left = torch.clamp(
                    commanded_left,
                    min=soft_lower[:, left_arm_joint_ids] + 0.02,
                    max=soft_upper[:, left_arm_joint_ids] - 0.02,
                )

            # --- Action Inversion & Mapping ---
            # 1. Update full articulation joint targets
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left
            target_all_joints[:, right_arm_joint_ids] = commanded_right

            # 2. Extract joints in the exact order arm_action expects
            arm_targets = target_all_joints[:, arm_term._joint_ids]

            # 3. Invert default offset and scale so applied == arm_targets
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            # 4. Grippers
            l_grip_t = torch.tensor([[left_gripper_cmd]], dtype=torch.float32, device=args_cli.device)
            r_grip_t = torch.tensor([[right_gripper_cmd]], dtype=torch.float32, device=args_cli.device)
            action = torch.cat([raw_arm_action, l_grip_t, r_grip_t], dim=-1)

            # Record step transition
            policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()
            action_np = action.squeeze(0).detach().cpu().numpy()
            ep_obs.append(policy_obs)
            ep_actions.append(action_np)
            

            obj_pose = torch.cat([env.scene["object"].data.root_pos_w - env.scene.env_origins, env.scene["object"].data.root_quat_w], dim=-1).squeeze(0).cpu().numpy()
            ep_obj_traj.append(obj_pose)
            ep_joint_traj.append(robot.data.joint_pos.squeeze(0).cpu().numpy())

            # Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(float(reward.mean().item()))

            # Check Termination / Success
            if phase == PHASE_SUCCESS:
                dist_to_target = torch.norm(obj_pos[0, :2] - target_pos[0, :2]).item()
                if dist_to_target < 0.15 and obj_pos[0, 2].item() < 0.06:
                    save_episode_to_hdf5(
                        args_cli.dataset_file,
                        collected_count,
                        ep_obs,
                        ep_actions,
                        ep_rewards,
                    )
                    collected_count += 1
                    print(f"  [✓] Successfully collected Demo #{collected_count-1} in {step} steps! (dist={dist_to_target:.3f}m)")
                    break
                else:
                    print(f"  [X] Object not on target (dist={dist_to_target:.3f}m, z={obj_pos[0, 2]:.3f}m). Discarding...")
                    break

            if terminated.item() or truncated.item():
                print(f"  [X] Episode terminated early at step {step} (phase={PHASE_NAMES[phase]}). Discarding...")
                break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Scripted Demo Generation Finished!")
    print(f" Total Demos in Dataset : {collected_count}")
    print(f" Time Elapsed           : {elapsed:.1f} seconds ({elapsed/max(1, collected_count-existing_demos):.1f} s/demo)")
    print(f" Saved Dataset          : {args_cli.dataset_file}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
```

### [File: `scripts/replay_demos.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Demonstrations Script.

Replays recorded demonstration trajectories in Isaac Sim to visually
inspect and verify motion quality and task success.

Usage:
    python scripts/replay_demos.py --dataset=data/demos.hdf5 --demo_idx=0
"""

import argparse
import os
import sys
import time
import h5py
import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Replay recorded demonstrations.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--demo_idx", type=int, default=0, help="Demo index to replay (or -1 for all).")
parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg


def replay_single_demo(env: ManagerBasedRLEnv, actions: list, demo_name: str, delay: float, init_states: dict = None):
    print(f"\n--- Replaying {demo_name} ({len(actions)} steps) ---")
    env.reset()

    if init_states:
        # Override object and target initial states
        if "init_object_pos" in init_states:
            obj_state = env.scene["object"].data.default_root_state.clone()
            obj_state[:, :3] = torch.tensor(init_states["init_object_pos"], device=env.device) + env.scene.env_origins
            obj_state[:, 3:7] = torch.tensor(init_states["init_object_quat"], device=env.device)
            env.scene["object"].write_root_state_to_sim(obj_state)
            
        if "init_target_pos" in init_states:
            tgt_state = env.scene["target"].data.default_root_state.clone()
            tgt_state[:, :3] = torch.tensor(init_states["init_target_pos"], device=env.device) + env.scene.env_origins
            tgt_state[:, 3:7] = torch.tensor(init_states["init_target_quat"], device=env.device)
            env.scene["target"].write_root_state_to_sim(tgt_state)
            
        if "init_robot_pos" in init_states:
            rob_state = env.scene["robot"].data.default_root_state.clone()
            rob_state[:, :3] = torch.tensor(init_states["init_robot_pos"], device=env.device) + env.scene.env_origins
            rob_state[:, 3:7] = torch.tensor(init_states["init_robot_quat"], device=env.device)
            env.scene["robot"].write_root_state_to_sim(rob_state)
            
            j_pos = torch.tensor(init_states["init_robot_joint_pos"], device=env.device).unsqueeze(0)
            j_vel = torch.tensor(init_states["init_robot_joint_vel"], device=env.device).unsqueeze(0)
            env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)

    for step_idx, act in enumerate(actions):
        if not simulation_app.is_running():
            break

        act_t = torch.tensor(act, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
        env.step(act_t)

        if delay > 0:
            time.sleep(delay)

    print(f"Finished {demo_name}.")


def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            env.close()
            simulation_app.close()
            return

        if args_cli.demo_idx >= 0:
            target_key = f"demo_{args_cli.demo_idx}"
            if target_key not in demo_keys:
                print(f"[Replay] Demo index {args_cli.demo_idx} not found. Available: {demo_keys}")
                env.close()
                simulation_app.close()
                return
            target_demos = [target_key]
        else:
            target_demos = demo_keys

        for key in target_demos:
            actions = data_grp[key]["actions"][:]
            init_states = None
            if "init_object_pos" in data_grp[key]:
                init_states = {
                    "init_object_pos": data_grp[key]["init_object_pos"][:],
                    "init_object_quat": data_grp[key]["init_object_quat"][:],
                    "init_target_pos": data_grp[key]["init_target_pos"][:],
                    "init_target_quat": data_grp[key]["init_target_quat"][:],
                }
            if "init_robot_pos" in data_grp[key]:
                init_states.update({
                    "init_robot_pos": data_grp[key]["init_robot_pos"][:],
                    "init_robot_quat": data_grp[key]["init_robot_quat"][:],
                    "init_robot_joint_pos": data_grp[key]["init_robot_joint_pos"][:],
                    "init_robot_joint_vel": data_grp[key]["init_robot_joint_vel"][:],
                })
                # Just dummy to not break old code
                }
            replay_single_demo(env, actions, key, args_cli.delay, init_states)
            time.sleep(1.0)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
```

### [File: `scripts/replay_demos_kinematic.py`]
```python
import argparse
import os
import sys
import time
import h5py
import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Kinematic Replay of multiple demonstrations (100% Visual Accuracy).")
parser.add_argument("--dataset", type=str, default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"))
parser.add_argument("--num_parallel", type=int, default=4, help="Number of demos to play simultaneously.")
parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg

def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            simulation_app.close()
            return
            
        num_parallel = min(args_cli.num_parallel, len(demo_keys))
        target_demos = demo_keys[:num_parallel]
        print(f"[Kinematic Replay] Preparing to play {num_parallel} demos in parallel: {target_demos}")

        # Check if kinematic data is available
        if "object_poses" not in data_grp[target_demos[0]]:
            print("ERROR: Dataset does not contain 'object_poses' and 'robot_joint_poses'.")
            print("Please regenerate the dataset using the updated collect/generate scripts!")
            simulation_app.close()
            return

        all_obj_traj = []
        all_joint_traj = []
        max_length = 0
        
        for key in target_demos:
            obj_traj = data_grp[key]["object_poses"][:]
            joint_traj = data_grp[key]["robot_joint_poses"][:]
            all_obj_traj.append(obj_traj)
            all_joint_traj.append(joint_traj)
            if len(obj_traj) > max_length:
                max_length = len(obj_traj)

    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = num_parallel
    gym.register(
        id="Isaac-Dual-Arm-IL-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DualArmILEnvCfg},
        disable_env_checker=True,
    )
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped

    env.reset()
    
    print(f"\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")
    
    obj_state = env.scene["object"].data.default_root_state.clone()
    j_pos = env.scene["robot"].data.default_joint_pos.clone()
    j_vel = env.scene["robot"].data.default_joint_vel.clone() * 0.0
    
    # Run loop
    for step_idx in range(max_length):
        if not simulation_app.is_running():
            break

        for i in range(num_parallel):
            o_traj = all_obj_traj[i]
            j_traj = all_joint_traj[i]
            idx = min(step_idx, len(o_traj) - 1)
            
            obj_state[i, :3] = torch.tensor(o_traj[idx, :3], device=env.device) + env.scene.env_origins[i]
            obj_state[i, 3:7] = torch.tensor(o_traj[idx, 3:7], device=env.device)
            j_pos[i] = torch.tensor(j_traj[idx], device=env.device)
            
        env.scene["object"].write_root_state_to_sim(obj_state)
        env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)
        
        # Step physics to render
        env.sim.step()
        
        if args_cli.delay > 0:
            time.sleep(args_cli.delay)

    print("Finished kinematic replay.")
    time.sleep(2.0)
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
```

### [File: `scripts/replay_demos_parallel.py`]
```python
import argparse
import os
import sys
import time
import h5py
import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Replay multiple demonstrations in parallel.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--num_parallel", type=int, default=4, help="Number of demos to play simultaneously.")
parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg

def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            simulation_app.close()
            return
            
        num_parallel = min(args_cli.num_parallel, len(demo_keys))
        target_demos = demo_keys[:num_parallel]
        print(f"[Replay] Preparing to play {num_parallel} demos in parallel: {target_demos}")

        # Load all actions for the selected demos
        all_actions = []
        all_init_states = []
        max_length = 0
        for key in target_demos:
            acts = data_grp[key]["actions"][:]
            all_actions.append(acts)
            if len(acts) > max_length:
                max_length = len(acts)
            if "init_object_pos" in data_grp[key]:
                state_dict = {
                    "init_object_pos": data_grp[key]["init_object_pos"][:],
                    "init_object_quat": data_grp[key]["init_object_quat"][:],
                    "init_target_pos": data_grp[key]["init_target_pos"][:],
                    "init_target_quat": data_grp[key]["init_target_quat"][:],
                }
                if "init_robot_pos" in data_grp[key]:
                    state_dict.update({
                        "init_robot_pos": data_grp[key]["init_robot_pos"][:],
                        "init_robot_quat": data_grp[key]["init_robot_quat"][:],
                        "init_robot_joint_pos": data_grp[key]["init_robot_joint_pos"][:],
                        "init_robot_joint_vel": data_grp[key]["init_robot_joint_vel"][:],
                    })
                all_init_states.append(state_dict)
            else:
                all_init_states.append(None)

    # Setup environment with num_envs = num_parallel
    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = num_parallel
    gym.register(
        id="Isaac-Dual-Arm-IL-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DualArmILEnvCfg},
        disable_env_checker=True,
    )
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped

    env.reset()
    
    # Override initial states for each parallel environment
    if any(s is not None for s in all_init_states):
        obj_state = env.scene["object"].data.default_root_state.clone()
        tgt_state = env.scene["target"].data.default_root_state.clone()
        
        for i, s in enumerate(all_init_states):
            if s is not None:
                obj_state[i, :3] = torch.tensor(s["init_object_pos"], device=env.device) + env.scene.env_origins[i]
                obj_state[i, 3:7] = torch.tensor(s["init_object_quat"], device=env.device)
                tgt_state[i, :3] = torch.tensor(s["init_target_pos"], device=env.device) + env.scene.env_origins[i]
                tgt_state[i, 3:7] = torch.tensor(s["init_target_quat"], device=env.device)
                
        env.scene["object"].write_root_state_to_sim(obj_state)
        env.scene["target"].write_root_state_to_sim(tgt_state)
        
        # Override robot states
        if "init_robot_pos" in all_init_states[0]:
            rob_state = env.scene["robot"].data.default_root_state.clone()
            j_pos = env.scene["robot"].data.default_joint_pos.clone()
            j_vel = env.scene["robot"].data.default_joint_vel.clone()
            
            for i, s in enumerate(all_init_states):
                if s is not None and "init_robot_pos" in s:
                    rob_state[i, :3] = torch.tensor(s["init_robot_pos"], device=env.device) + env.scene.env_origins[i]
                    rob_state[i, 3:7] = torch.tensor(s["init_robot_quat"], device=env.device)
                    j_pos[i] = torch.tensor(s["init_robot_joint_pos"], device=env.device)
                    j_vel[i] = torch.tensor(s["init_robot_joint_vel"], device=env.device)
                    
            env.scene["robot"].write_root_state_to_sim(rob_state)
            env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)

    print(f"\n--- Starting parallel replay (Max steps: {max_length}) ---")
    
    # Run loop
    for step_idx in range(max_length):
        if not simulation_app.is_running():
            break

        # Construct batched action [num_envs, act_dim]
        # If a demo has finished, repeat its last action
        batched_actions = []
        for i in range(num_parallel):
            acts = all_actions[i]
            idx = min(step_idx, len(acts) - 1)
            batched_actions.append(acts[idx])
            
        act_t = torch.tensor(batched_actions, dtype=torch.float32, device=args_cli.device)
        env.step(act_t)

        if args_cli.delay > 0:
            time.sleep(args_cli.delay)

    print("Finished parallel replay.")
    time.sleep(2.0)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
```

### [File: `scripts/train.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Training Script for Dual Arm Imitation Learning.

Supports:
    - Behavior Cloning (BC)
    - Diffusion Policy
    - Action Chunking with Transformers (ACT)

Usage:
    python scripts/train.py --algo=bc --epochs=100
    python scripts/train.py --algo=diffusion --epochs=150
    python scripts/train.py --algo=act --epochs=150
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train imitation learning policies.")
    parser.add_argument(
        "--algo",
        type=str,
        default="diffusion",
        choices=["bc", "diffusion", "act"],
        help="Algorithm to train: bc, diffusion, or act.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
        help="Path to HDF5 demonstration dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (defaults to configs/{algo}_cfg.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    return parser.parse_args()


def load_config(algo: str, custom_cfg_path: str | None) -> dict:
    if custom_cfg_path is None:
        cfg_path = os.path.join(PROJECT_ROOT, "configs", f"{algo}_cfg.yaml")
    else:
        cfg_path = custom_cfg_path

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(algo: str, cfg: dict, obs_dim: int, act_dim: int) -> torch.nn.Module:
    if algo == "bc":
        model_type = cfg.get("model_type", "mlp")
        if model_type == "rnn":
            return RNNBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=cfg.get("rnn_hidden_dim", 256),
                num_layers=cfg.get("rnn_num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            return MLPBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dims=cfg.get("hidden_dims", [512, 512, 256]),
                dropout=cfg.get("dropout", 0.1),
                activation=cfg.get("activation", "relu"),
            )

    elif algo == "diffusion":
        return DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
            kernel_size=cfg.get("kernel_size", 5),
            n_groups=cfg.get("n_groups", 8),
        )

    elif algo == "act":
        return ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_encoder_layers=cfg.get("num_encoder_layers", 4),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            latent_dim=cfg.get("latent_dim", 32),
            kl_weight=cfg.get("kl_weight", 10.0),
            dropout=cfg.get("dropout", 0.1),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
            temporal_ensemble_coeff=cfg.get("temporal_ensemble_coeff", 0.01),
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.algo, args.config)

    # Overrides
    epochs = args.epochs or cfg.get("epochs", 100)
    batch_size = args.batch_size or cfg.get("batch_size", 64)
    lr = args.lr or cfg.get("learning_rate", 1.0e-4)
    weight_decay = cfg.get("weight_decay", 1.0e-5)

    print("=" * 60)
    print(f" Training Imitation Learning Policy: {args.algo.upper()}")
    print("=" * 60)
    print(f" Dataset    : {args.dataset}")
    print(f" Batch Size : {batch_size}")
    print(f" Epochs     : {epochs}")
    print(f" Device     : {args.device}")
    print(f" LR         : {lr}")
    print("=" * 60)

    # Setup directories
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.algo)
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard setup
    run_name = f"{args.algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir = os.path.join(PROJECT_ROOT, "runs", run_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f" TensorBoard Log Dir: {tb_log_dir}")
    print(" (Run: tensorboard --logdir=runs  to monitor)\n" + "=" * 60)

    # Load Dataset
    pred_h = cfg.get("pred_horizon", 16) if args.algo == "diffusion" else cfg.get("chunk_size", 24)
    obs_h = cfg.get("obs_horizon", 2)

    full_dataset = DualArmDataset(
        dataset_path=args.dataset,
        algo=args.algo,
        pred_horizon=pred_h,
        obs_horizon=obs_h,
    )

    # Save normalization stats for evaluation
    stats_path = os.path.join(save_dir, "stats.pkl")
    full_dataset.save_stats(stats_path)

    # Train / Val Split (90% train, 10% val)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Model
    model = build_model(args.algo, cfg, full_dataset.obs_dim, full_dataset.act_dim)
    model.to(args.device)
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[Model] Resuming training from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=args.device, weights_only=True))
        else:
            print(f"[Warning] Checkpoint not found: {args.resume}. Starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.algo.upper()} created with {total_params:,} trainable parameters.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            num_train_batches += 1
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                loss_dict = model.compute_loss(batch)
                val_loss_sum += loss_dict["loss"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "algo": args.algo,
                    "config": cfg,
                    "obs_dim": full_dataset.obs_dim,
                    "act_dim": full_dataset.act_dim,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best model to {best_ckpt_path} (Val Loss: {best_val_loss:.5f})")

        # Periodic checkpoint
        if epoch % cfg.get("save_interval", 20) == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print("\n[Training Complete]")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_model.pt')}")
    writer.close()


if __name__ == "__main__":
    main()
```

### [File: `teleop/collect_demos.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstration Collection Script for Dual Arm via Teleoperation.

Collects expert demonstrations in Isaac Sim using the DualArmTeleopController,
and exports successful trajectories into an HDF5 dataset compatible with
BC, Diffusion Policy, and ACT.

Usage:
    # Run from the isaac_lab root or dual_arm_il directory:
    python /home/optimus/isaac_lab/dual_arm_il/teleop/collect_demos.py --num_demos=20
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Isaac Lab AppLauncher (Must be called before importing omni or isaaclab modules)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect dual arm demonstrations via teleoperation.")
parser.add_argument("--num_demos", type=int, default=20, help="Target number of demonstrations to collect.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to the output HDF5 dataset file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after simulation app launch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
    from dual_arm_il.teleop.dual_arm_teleop import DualArmTeleopController
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg
    from teleop.dual_arm_teleop import DualArmTeleopController


def solve_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics.

    Args:
        jacobian: Jacobian matrix of shape (1, 6, 7).
        delta_pose: Desired task-space delta [dx, dy, dz, droll, dpitch, dyaw] of shape (1, 6).
        damping: Damping factor lambda.

    Returns:
        delta_q: Change in joint angles of shape (1, 7).
    """
    # J: (1, 6, 7)
    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)

    # J * J^T + lambda^2 * I
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)  # (6, 6)

    # J^T * inv * e
    delta_q = torch.matmul(j.transpose(0, 1), torch.matmul(inv_term, e))  # (7,)
    return delta_q.unsqueeze(0)


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list, init_states: dict = None, obj_traj: list = None, joint_traj: list = None):
    """Save a single successful demonstration to the HDF5 file."""
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)

    mode = "a" if os.path.exists(hdf5_path) else "w"
    with h5py.File(hdf5_path, mode) as f:
        data_group = f.require_group("data")
        demo_group = data_group.create_group(f"demo_{ep_idx}")

        obs_array = np.array(observations, dtype=np.float32)
        act_array = np.array(actions, dtype=np.float32)
        rew_array = np.array(rewards, dtype=np.float32)

        demo_group.create_dataset("obs", data=obs_array, compression="gzip")
        demo_group.create_dataset("actions", data=act_array, compression="gzip")
        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.attrs["num_samples"] = len(act_array)
        if init_states:
            for k, v in init_states.items():
                demo_group.create_dataset(k, data=v)
                
        if obj_traj is not None:
            demo_group.create_dataset("object_poses", data=np.array(obj_traj, dtype=np.float32), compression="gzip")
        if joint_traj is not None:
            demo_group.create_dataset("robot_joint_poses", data=np.array(joint_traj, dtype=np.float32), compression="gzip")

        # Update total samples attribute in data group
        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")


def main():
    cfg = DualArmILEnvCfg()
    cfg.sim.device = args_cli.device

    print("[Collect] Initializing Isaac Lab environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]

    # Locate body indices and joint indices
    # Right arm: "panda_hand_0", joints "panda_joint[1-7]_0"
    # Left arm:  "panda_hand",   joints "panda_joint[1-7]"
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # Find 14 arm joint indices in articulation
    left_arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]_0")

    print(f"[Collect] Right hand body index: {right_hand_idx}, Left hand body index: {left_hand_idx}")
    print(f"[Collect] Left arm joints: {left_arm_joint_ids}")
    print(f"[Collect] Right arm joints: {right_arm_joint_ids}")

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]

    # Initialize Teleop Controller
    teleop = DualArmTeleopController(device=args_cli.device)

    # Inspect existing dataset to get starting demo index
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
    print(f"[Collect] Existing demonstrations found: {existing_demos}")

    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    # Main collection loop
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()
        teleop.reset()
        
        init_states = {
            "init_object_pos": (env.scene["object"].data.root_pos_w - env.scene.env_origins).clone().squeeze(0).cpu().numpy(),
            "init_object_quat": env.scene["object"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_target_pos": (env.scene["target"].data.root_pos_w - env.scene.env_origins).clone().squeeze(0).cpu().numpy(),
            "init_target_quat": env.scene["target"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_pos": (env.scene["robot"].data.root_pos_w - env.scene.env_origins).clone().squeeze(0).cpu().numpy(),
            "init_robot_quat": env.scene["robot"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_joint_pos": env.scene["robot"].data.joint_pos.clone().squeeze(0).cpu().numpy(),
            "init_robot_joint_vel": env.scene["robot"].data.joint_vel.clone().squeeze(0).cpu().numpy(),
        }

        # Commanded joint targets buffer (starts at current joint positions)
        current_joint_pos = robot.data.joint_pos.clone()
        commanded_left_joints = current_joint_pos[:, left_arm_joint_ids].clone()
        commanded_right_joints = current_joint_pos[:, right_arm_joint_ids].clone()

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_obj_traj = []
        ep_joint_traj = []

        print(f"\n>>> Starting Episode for Demo #{collected_count} (Target: {target_count}) <<<")

        step_idx = 0
        while simulation_app.is_running():
            step_idx += 1

            # 1. Read teleoperation command
            active_arm, dpos, drot, r_grip, l_grip = teleop.get_delta_command()

            # 2. Compute IK if there is a delta movement
            delta_pose = np.concatenate([dpos, drot])  # 6D: [dx, dy, dz, drx, dry, drz]
            has_motion = np.linalg.norm(dpos) > 1e-5 or np.linalg.norm(drot) > 1e-5

            if has_motion:
                delta_pose_t = torch.tensor(delta_pose, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
                jacobians = robot.root_physx_view.get_jacobians()  # (1, num_bodies, 6, num_dofs)

                if active_arm == "right":
                    # Jacobians for right hand w.r.t right arm joints
                    j_right = jacobians[:, jacobi_right_hand_idx, :, :][:, :, right_arm_joint_ids]
                    dq_right = solve_dls_ik(j_right, delta_pose_t, damping=0.08)
                    commanded_right_joints += dq_right
                else:
                    # Jacobians for left hand w.r.t left arm joints
                    j_left = jacobians[:, jacobi_left_hand_idx, :, :][:, :, left_arm_joint_ids]
                    dq_left = solve_dls_ik(j_left, delta_pose_t, damping=0.08)
                    commanded_left_joints += dq_left

            # 3. Assemble 16D action with exact action inversion
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left_joints
            target_all_joints[:, right_arm_joint_ids] = commanded_right_joints

            arm_targets = target_all_joints[:, arm_term._joint_ids]
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            left_grip_t = torch.tensor([[l_grip]], dtype=torch.float32, device=args_cli.device)
            right_grip_t = torch.tensor([[r_grip]], dtype=torch.float32, device=args_cli.device)

            action = torch.cat([raw_arm_action, left_grip_t, right_grip_t], dim=-1)

            # 4. Record step transition
            policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()
            action_np = action.squeeze(0).detach().cpu().numpy()

            ep_obs.append(policy_obs)
            ep_actions.append(action_np)
            

            obj_pose = torch.cat([env.scene["object"].data.root_pos_w - env.scene.env_origins, env.scene["object"].data.root_quat_w], dim=-1).squeeze(0).cpu().numpy()
            ep_obj_traj.append(obj_pose)
            ep_joint_traj.append(robot.data.joint_pos.squeeze(0).cpu().numpy())

            # 5. Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(float(reward.mean().item()))

            # 6. Check workflow flags
            if teleop.flag_save_episode:
                save_episode_to_hdf5(
                    args_cli.dataset_file,
                    collected_count,
                    ep_obs,
                    ep_actions,
                    ep_rewards,
                    init_states,
                    ep_obj_traj,
                    ep_joint_traj,
                )
                collected_count += 1
                teleop.reset_episode_flags()
                break

            if teleop.flag_discard_episode:
                print(f"[Collect] Episode discarded by user. Total retained: {collected_count}")
                teleop.reset_episode_flags()
                break

            if teleop.flag_reset_env or terminated.item() or truncated.item():
                if terminated.item():
                    print("[Collect] Episode terminated (e.g. object dropped). Discarding...")
                teleop.reset_episode_flags()
                break

    print(f"\n[Collect] Finished! Total demonstrations collected: {collected_count}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
```

### [File: `teleop/dual_arm_teleop.py`]
```python
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual Arm Teleoperation Controller for Isaac Sim.

Provides SE(3) task-space control for both Franka arms with an Active-Arm Toggle (Tab key),
Differential IK calculation, and episode recording controls (Confirm/Discard/Reset).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import carb
import omni


class DualArmTeleopController:
    """Keyboard controller for Dual Franka arms in Isaac Lab.

    Key bindings:
        -------------------------------------------------------------
        [Arm Selection]
        TAB              : Toggle active arm (Right Arm <-> Left Arm)
        1                : Force select Right Arm (Pick Arm)
        2                : Force select Left Arm (Place Arm)

        [Task-Space Movement (Active Arm)]
        W / S            : Move Forward / Backward (+X / -X)
        A / D            : Move Left / Right (+Y / -Y)
        Q / E            : Move Up / Down (+Z / -Z)
        Z / X            : Roll (+ / -)
        T / G            : Pitch (+ / -)
        C / V            : Yaw (+ / -)

        [Gripper Control]
        SPACE or K       : Toggle Gripper of active arm (Open <-> Close)
        O                : Open active gripper
        P                : Close active gripper

        [Data Collection Workflow]
        ENTER or Y       : Confirm & Save current episode to HDF5
        BACKSPACE or N   : Discard current episode and reset
        R                : Reset environment

        [Sensitivity]
        UP / DOWN        : Increase / Decrease translation sensitivity
        -------------------------------------------------------------
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device

        # Sensitivities
        self.pos_step = 0.015   # 1.5 cm per press / tick
        self.rot_step = 0.03    # ~1.7 degrees per press / tick

        # Active arm tracking: "right" or "left"
        self.active_arm = "right"

        # Gripper states: True = closed (-1.0), False = open (+1.0)
        self.right_gripper_closed = False
        self.left_gripper_closed = False

        # Delta accumulators
        self.delta_pos = np.zeros(3, dtype=np.float32)
        self.delta_rot = np.zeros(3, dtype=np.float32)

        # Status flags for demonstration workflow
        self.flag_save_episode = False
        self.flag_discard_episode = False
        self.flag_reset_env = False

        # Acquire Omniverse input interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to keyboard events with weakref
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Key state tracking for smooth continuous holding
        self._keys_down = set()

        print("[DualArmTeleopController] Initialized successfully.")
        self.print_instructions()

    def __del__(self):
        if hasattr(self, "_sub") and self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def print_instructions(self):
        print("=" * 65)
        print(" Dual Arm Teleoperation - Key Bindings")
        print("=" * 65)
        print("  TAB          : Toggle Active Arm (Current: " + self.active_arm.upper() + ")")
        print("  W / S        : Move X (+ / -)")
        print("  A / D        : Move Y (+ / -)")
        print("  Q / E        : Move Z (+ / -)")
        print("  Z / X        : Roll  (+ / -)")
        print("  T / G        : Pitch (+ / -)")
        print("  C / V        : Yaw   (+ / -)")
        print("  SPACE / K    : Toggle Active Gripper (Open <-> Close)")
        print("  ENTER / Y    : [SAVE] Confirm & Save Demo")
        print("  BACKSPACE / N: [DISCARD] Discard Demo & Reset")
        print("  R            : [RESET] Reset Environment")
        print("=" * 65)

    def reset_episode_flags(self):
        """Reset workflow flags at the beginning of each episode."""
        self.flag_save_episode = False
        self.flag_discard_episode = False
        self.flag_reset_env = False
        self.delta_pos[:] = 0.0
        self.delta_rot[:] = 0.0

    def reset(self):
        """Full reset of controller state."""
        self.reset_episode_flags()
        self.right_gripper_closed = False
        self.left_gripper_closed = False
        self.active_arm = "right"

    def _on_keyboard_event(self, event, *args, **kwargs):
        name = event.input.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys_down.add(name)

            # --- Arm Switching ---
            if name == "TAB":
                self.active_arm = "left" if self.active_arm == "right" else "right"
                print(f"[Teleop] Switched active arm to: >> {self.active_arm.upper()} <<")
            elif name in ("NUM_1", "KEY_1"):
                self.active_arm = "right"
                print("[Teleop] Active arm: >> RIGHT (Pick) <<")
            elif name in ("NUM_2", "KEY_2"):
                self.active_arm = "left"
                print("[Teleop] Active arm: >> LEFT (Place) <<")

            # --- Gripper Toggles ---
            elif name in ("SPACE", "K"):
                if self.active_arm == "right":
                    self.right_gripper_closed = not self.right_gripper_closed
                    status = "CLOSED" if self.right_gripper_closed else "OPEN"
                    print(f"[Teleop] Right Gripper: {status}")
                else:
                    self.left_gripper_closed = not self.left_gripper_closed
                    status = "CLOSED" if self.left_gripper_closed else "OPEN"
                    print(f"[Teleop] Left Gripper: {status}")
            elif name == "O":
                if self.active_arm == "right":
                    self.right_gripper_closed = False
                else:
                    self.left_gripper_closed = False
                print(f"[Teleop] {self.active_arm.capitalize()} Gripper OPEN")
            elif name == "P":
                if self.active_arm == "right":
                    self.right_gripper_closed = True
                else:
                    self.left_gripper_closed = True
                print(f"[Teleop] {self.active_arm.capitalize()} Gripper CLOSED")

            # --- Workflow Flags ---
            elif name in ("ENTER", "Y"):
                self.flag_save_episode = True
                print("\n>>> [DEMO CONFIRMED] Saving episode to dataset! <<<")
            elif name in ("BACKSPACE", "N"):
                self.flag_discard_episode = True
                print("\n>>> [DEMO DISCARDED] Episode dropped! Resetting... <<<")
            elif name == "R":
                self.flag_reset_env = True
                print("[Teleop] Reset requested.")

            # --- Sensitivity Adjust ---
            elif name == "UP":
                self.pos_step = min(0.05, self.pos_step + 0.005)
                print(f"[Teleop] Sensitivity increased: {self.pos_step*100:.1f} cm/step")
            elif name == "DOWN":
                self.pos_step = max(0.002, self.pos_step - 0.005)
                print(f"[Teleop] Sensitivity decreased: {self.pos_step*100:.1f} cm/step")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_down.discard(name)

    def get_delta_command(self) -> tuple[str, np.ndarray, np.ndarray, float, float]:
        """Poll currently held keys and compute delta position and rotation for the active arm.

        Returns:
            active_arm: "right" or "left"
            dpos: np.ndarray (3,) [dx, dy, dz] in meters
            drot: np.ndarray (3,) [droll, dpitch, dyaw] in radians
            right_gripper_val: float (-1.0 for close, +1.0 for open)
            left_gripper_val: float (-1.0 for close, +1.0 for open)
        """
        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)

        # X axis (Forward / Backward)
        if "W" in self._keys_down:
            dpos[0] += self.pos_step
        if "S" in self._keys_down:
            dpos[0] -= self.pos_step

        # Y axis (Left / Right)
        if "A" in self._keys_down:
            dpos[1] += self.pos_step
        if "D" in self._keys_down:
            dpos[1] -= self.pos_step

        # Z axis (Up / Down)
        if "Q" in self._keys_down:
            dpos[2] += self.pos_step
        if "E" in self._keys_down:
            dpos[2] -= self.pos_step

        # Roll
        if "Z" in self._keys_down:
            drot[0] += self.rot_step
        if "X" in self._keys_down:
            drot[0] -= self.rot_step

        # Pitch
        if "T" in self._keys_down:
            drot[1] += self.rot_step
        if "G" in self._keys_down:
            drot[1] -= self.rot_step

        # Yaw
        if "C" in self._keys_down:
            drot[2] += self.rot_step
        if "V" in self._keys_down:
            drot[2] -= self.rot_step

        right_grip = -1.0 if self.right_gripper_closed else 1.0
        left_grip = -1.0 if self.left_gripper_closed else 1.0

        return self.active_arm, dpos, drot, right_grip, left_grip
```

