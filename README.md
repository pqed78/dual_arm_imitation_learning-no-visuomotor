# Dual Arm Visuomotor Imitation Learning Project

This is an **independent Visuomotor Imitation Learning (IL)** project that is completely isolated from the existing Reinforcement Learning (RL, `dual_arm0`) environment.
It supports the entire pipeline, from teleoperation demonstration data collection in the Isaac Sim environment to the training and simulation evaluation of three state-of-the-art robot manipulation imitation learning algorithms (**BC, Diffusion Policy, ACT**).

**Key Feature:** This project heavily utilizes **Visuomotor policies**. It introduces an RGB Front Camera (320x240) to the simulation. Ground truth object states are explicitly removed from the policy observation space; instead, the agents must infer the object state purely from the visual input (via a versatile Vision Encoder supporting ResNet, MobileNet, EfficientNet, and ViT) combined with proprioception (joint states & TCP poses).

---

## 1. Project Structure

```text
/home/optimus/isaac_lab/dual_arm_il_visuo/
├── README.md                  # Project usage guide (This document)
├── requirements.txt           # Required Python libraries list
├── configs/                   # Environment and model hyperparameters
│   ├── env_cfg.py             # Teleop/IL custom Isaac Lab environment settings (Camera added, grasping tuned)
│   ├── bc_cfg.yaml            # Behavior Cloning settings
│   ├── diffusion_cfg.yaml     # Diffusion Policy settings
│   └── act_cfg.yaml           # ACT (Action Chunking with Transformers) settings
├── teleop/                    # Teleoperation & Data collection
│   ├── dual_arm_teleop.py     # Active-Arm Toggle SE(3) keyboard controller
│   └── collect_demos.py       # Isaac Sim teleop and HDF5 episode saving script (Saves images)
├── dataset/                   # Dataset pipeline
│   └── il_dataset.py          # PyTorch HDF5 Dataset (Image preprocessing, Normalization, Horizon slicing, Chunking)
├── models/                    # 3 Major Imitation Learning Algorithms
│   ├── vision_encoder.py      # Versatile Vision Backbone (Supports ResNet, MobileNet, EfficientNet, ViT)
│   ├── bc/                    # MLP / RNN Behavior Cloning
│   ├── diffusion/             # 1D Temporal UNet Diffusion Policy (Chi et al. 2023)
│   └── act/                   # CVAE + Transformer ACT (Zhao et al. 2023)
├── scripts/                   # Execution scripts
│   ├── generate_scripted_demos.py  # [Recommended] Script-based high-quality single demo auto-generator
│   ├── replay_demos.py        # Verify collected HDF5 demos via simulation replay
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

### ⓪ Step 0: Activate Virtual Environment
Before running any scripts, you must activate the Isaac Lab virtual environment in your terminal:
```bash
source ~/isaac_lab/bin/activate
```

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
Performs high-speed GPU parallel training in a pure PyTorch environment without launching Isaac Sim. The Vision Encoder (default: ResNet-18) is trained end-to-end with the policy algorithms. 

You can easily swap out the vision backbone by passing the `backbone_type` parameter to the `VisionEncoder` (e.g., `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`).

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
Connects the trained model to the simulation robot to measure the actual closed-loop success rate based on visual input.

```bash
# Evaluate Diffusion Policy
python scripts/eval.py --algo=diffusion --num_episodes=10

# Evaluate ACT (Temporal Ensembling applied)
python scripts/eval.py --algo=act --num_episodes=10

# Evaluate Behavior Cloning
python scripts/eval.py --algo=bc --num_episodes=10
```

---

## 4. Camera Configuration (Visuomotor Setup)

In Isaac Lab, cameras must be explicitly defined as dataclass fields in the Scene configuration. We provide a custom `VisuomotorSceneCfg` in `configs/env_cfg.py` for this purpose.

```python
@configclass
class VisuomotorSceneCfg(DualArmSceneCfg):
    # Agent Vision Camera (Input to Vision Encoder)
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera", # Spawns at the root of the environment
        update_period=0.0,
        height=240, width=320,              # Camera Resolution
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, 0.0, 1.0),            # Camera Position (X, Y, Z)
            rot=(0.5, 0.5, 0.5, 0.5),       # Camera Rotation (Quaternion)
            convention="ros"
        ),
    )
```

### Adding a Wrist Camera (Eye-in-Hand)
To attach a camera dynamically to the robot's wrist without modifying the USD file, simply spawn a camera with a `prim_path` that is a child of the robot's hand link:

```python
    wrist_camera: CameraCfg = CameraCfg(
        # Spawns as a child of the robot's hand, so it automatically tracks hand movement!
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(...),
        # Offset is relative to the hand link
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
```
*Note: The vision encoder expects the specified input size, but ViT models will automatically resize the image to 224x224 internally.*

---

## 5. Relationship with Existing RL Environment

- All code in this folder (`dual_arm_il_visuo`) does not modify any files in `/home/optimus/isaac_lab/dual_arm0`.
- Simulation scenes (`DualArmSceneCfg`) and robot definitions are safely inherited (`configs/env_cfg.py`) and reused, meaning it causes absolutely no interference with ongoing training or tuning on the RL side.

---

# Dual Arm Visuomotor Imitation Learning (비주얼 모터 모방 학습) 프로젝트

기존 강화학습(RL, `dual_arm0`) 환경과 완벽히 격리된 **독립적인 Visuomotor 모방 학습(Imitation Learning, IL)** 프로젝트입니다.  
Isaac Sim 환경에서의 텔레오퍼레이션(수동 조작) 시연 데이터 수집부터 최신 로봇 조작 모방 학습 3대 알고리즘(**BC, Diffusion Policy, ACT**) 학습 및 시뮬레이션 평가까지 전 과정을 지원합니다.

**핵심 특징:** 이 프로젝트는 **카메라 영상을 활용(Visuomotor)**하도록 완전히 업그레이드 되었습니다. 시뮬레이션 내에 전면 카메라(RGB 320x240)가 추가되었으며, 모델의 Observation에서 물체의 실제 좌표(Ground truth state) 정보가 명시적으로 제거되었습니다. 모델은 오로지 **카메라 이미지(ResNet, MobileNet, EfficientNet, ViT 등 다양한 백본 지원)와 로봇 자신의 관절/TCP 상태만**을 보고 상황을 파악하여 행동해야 합니다.

---

## 1. 프로젝트 구조

```text
/home/optimus/isaac_lab/dual_arm_il_visuo/
├── README.md                  # 프로젝트 사용 가이드 (본 문서)
├── requirements.txt           # 필요한 파이썬 라이브러리 목록
├── configs/                   # 환경 및 모델 하이퍼파라미터
│   ├── env_cfg.py             # 텔레옵/IL 맞춤형 Isaac Lab 환경 설정 (카메라 추가, 그라스핑 마찰력 튜닝)
│   ├── bc_cfg.yaml            # Behavior Cloning 설정
│   ├── diffusion_cfg.yaml     # Diffusion Policy 설정
│   └── act_cfg.yaml           # ACT (Action Chunking with Transformers) 설정
├── teleop/                    # 텔레오퍼레이션 & 데이터 수집
│   ├── dual_arm_teleop.py     # Active-Arm Toggle SE(3) 키보드 제어기
│   └── collect_demos.py       # Isaac Sim 텔레옵 및 HDF5 에피소드 저장 스크립트 (이미지 포함)
├── dataset/                   # 데이터셋 파이프라인
│   └── il_dataset.py          # PyTorch HDF5 Dataset (이미지 전처리, 정규화, Horizon 슬라이싱, Chunking)
├── models/                    # 모방학습 3대 알고리즘 구현체
│   ├── vision_encoder.py      # 다목적 비전 백본 인코더 (ResNet, MobileNet, EfficientNet, ViT 지원)
│   ├── bc/                    # MLP / RNN Behavior Cloning
│   ├── diffusion/             # 1D Temporal UNet Diffusion Policy (Chi et al. 2023)
│   └── act/                   # CVAE + Transformer ACT (Zhao et al. 2023)
├── scripts/                   # 실행 스크립트
│   ├── generate_scripted_demos.py  # [추천] 스크립트 기반 고품질 단일 데모 자동 생성기
│   ├── replay_demos.py        # 수집된 HDF5 데모 시뮬레이션 재생 검증
│   ├── train.py               # 3종 알고리즘 통합 고속 GPU 학습 스크립트
│   └── eval.py                # Isaac Sim 환경에서 비전 기반 정책 롤아웃 평가
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

### ⓪ 0단계: 가상환경 활성화 (필수)
모든 터미널 작업(스크립트 실행) 전에는 반드시 Isaac Lab 가상환경을 활성화해야 합니다:
```bash
source ~/isaac_lab/bin/activate
```

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

### ③ 3단계: 비전-모방학습 모델 훈련 (`train.py`)
Isaac Sim을 켜지 않고 순수 PyTorch 환경에서 고속 GPU 병렬 학습을 수행합니다. (기본 설정인 ResNet-18을 포함한 다양한 비전 인코더도 함께 End-to-End로 학습됩니다.)

💡 **비전 백본(Vision Backbone) 변경 가능**:
정책 파일에서 `VisionEncoder`를 생성할 때 `backbone_type` 파라미터를 넘겨주어 쉽게 모델을 교체할 수 있습니다. 
지원 목록: `resnet18`(기본), `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`

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
학습된 모델을 시뮬레이션 로봇에 연결하여 실제 클로즈드 루프 성공률을 측정합니다. 모델은 제공되는 카메라 영상과 로봇 관절 상태만으로 예측을 수행합니다.

```bash
# Diffusion Policy 평가
python scripts/eval.py --algo=diffusion --num_episodes=10

# ACT 평가 (Temporal Ensembling 적용)
python scripts/eval.py --algo=act --num_episodes=10

# Behavior Cloning 평가
python scripts/eval.py --algo=bc --num_episodes=10
```

---

## 4. 카메라 설정 (Visuomotor 셋업)

Isaac Lab에서는 카메라를 씬(Scene) 설정의 데이터클래스 필드로 명시적으로 선언해야 시뮬레이션에 정상적으로 스폰(Spawn)됩니다. 이를 위해 `configs/env_cfg.py`에 `VisuomotorSceneCfg`를 새롭게 정의했습니다.

```python
@configclass
class VisuomotorSceneCfg(DualArmSceneCfg):
    # 에이전트 입력용 정면 비전 카메라
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera", # 환경의 최상위 경로에 스폰 (고정됨)
        update_period=0.0,
        height=240, width=320,              # 해상도 변경
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, 0.0, 1.0),            # 카메라 위치 (X, Y, Z)
            rot=(0.5, 0.5, 0.5, 0.5),       # 카메라 회전 (쿼터니언)
            convention="ros"
        ),
    )
```

### 📷 손목 카메라 (Eye-in-Hand) 추가 방법
로봇 원본 USD 파일을 건드리지 않고, 코드를 통해서만 손목에 카메라를 달려면 `prim_path`를 로봇 손목 객체의 자식 경로로 설정하면 됩니다.

```python
    wrist_camera: CameraCfg = CameraCfg(
        # 로봇 손목(panda_hand)의 하위 객체로 스폰되므로, 로봇이 움직이면 카메라도 자동으로 따라다닙니다!
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(...),
        # offset은 손목 기준 '상대 좌표'가 됩니다.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
```
*참고: ViT 백본을 사용할 경우, 입력된 해상도와 무관하게 내부적으로 224x224 크기로 자동 변환(Resize)되어 처리됩니다.*

---

## 5. 기존 RL 환경과의 관계

- 본 폴더(`dual_arm_il_visuo`)의 모든 코드는 `/home/optimus/isaac_lab/dual_arm0`의 파일을 수정하지 않습니다.
- 시뮬레이션 씬(`DualArmSceneCfg`) 및 로봇 정의는 환경에 설치된 설정을 안전하게 상속(`configs/env_cfg.py`)받아 재사용하므로, RL 쪽에서 현재 진행 중인 학습이나 튜닝에 아무런 간섭을 주지 않습니다.
