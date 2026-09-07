with open('README.md', 'r') as f:
    text = f.read()

tree_old = """├── scripts/                   # 실행 스크립트
│   ├── generate_scripted_demos.py # [추천] 스크립트 기반 고품질 데모 자동 생성기
│   ├── replay_demos.py        # 수집된 HDF5 데모 시뮬레이션 재생 검증"""

tree_new = """├── scripts/                   # 실행 스크립트
│   ├── generate_scripted_demos.py  # [추천] 스크립트 기반 고품질 단일 데모 자동 생성기
│   ├── generate_scripted_demos2.py # [강력 추천] 대규모 텐서 병렬화 초고속 자동 생성기
│   ├── replay_demos.py         # 수집된 HDF5 데모 시뮬레이션 재생 검증"""

text = text.replace(tree_old, tree_new)

usage_old = """### ① 1단계: 데모 데이터 수집 (2가지 방법 중 선택)

#### [방법 A: 추천] 스크립트 기반 자동 데모 생성기 (`generate_scripted_demos.py`)
수동 조작의 피로와 멈칫거림 없이, 역기구학(IK)과 Waypoint 플래너로 완벽한 고품질 데모를 초고속 생성합니다.

```bash
# GUI 화면을 보면서 50개 데모 자동 수집
python scripts/generate_scripted_demos.py --num_demos=50

# 초고속 Headless 모드 (화면 없이 GPU 최대 속도로 100개 데모를 1~2분 만에 생성)
python scripts/generate_scripted_demos.py --num_demos=100 --headless
```"""

usage_new = """### ① 1단계: 데모 데이터 수집 (3가지 방법 중 선택)

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
```"""

text = text.replace(usage_old, usage_new)

text = text.replace("#### [방법 B] 키보드 텔레오퍼레이션 수동 수집 (`collect_demos.py`)", "#### [방법 C] 키보드 텔레오퍼레이션 수동 수집 (`collect_demos.py`)")

with open('README.md', 'w') as f:
    f.write(text)
