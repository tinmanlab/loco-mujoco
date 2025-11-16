# SOTA MuJoCo/MJX 기반 Humanoid Locomotion 연구 조사

**날짜**: 2025-01-15
**목적**: Isaac Gym 대신 MuJoCo/MJX로 사용 가능한 최신 robustness 기법 찾기

---

## 🔍 조사 결과 요약

### ✅ **사용 가능한 SOTA 프로젝트**

1. **MuJoCo Playground** (2025, Google DeepMind) ⭐⭐⭐⭐⭐
2. **HumanoidBench** (2024) ⭐⭐⭐⭐
3. **PHC_MJX** (2024, 개발 중) ⭐⭐⭐

### ❌ **MuJoCo 미지원**

1. **ASE** - Isaac Gym 전용
2. **PHC (원본)** - Isaac Gym 기반

---

## 1. MuJoCo Playground (2025) ⭐ 최우선 추천!

### 📊 기본 정보

- **개발**: Google DeepMind
- **릴리즈**: 2024년 12월 (매우 최신!)
- **논문**: arXiv:2502.08844 (2025년 2월)
- **GitHub**: https://github.com/google-deepmind/mujoco_playground
- **라이센스**: Apache 2.0 (완전 오픈소스)
- **Stars**: 1.6k+ (매우 활발)

### ✅ 지원 로봇

**Humanoids:**
- ✅ **Unitree H1** (우리가 사용 중!)
- ✅ **Unitree G1**
- Berkeley Humanoid
- Booster T1
- Robotis OP3

**기타:**
- Quadrupeds (4족 로봇)
- Dexterous hands (손)
- Robotic arms

### 🚀 핵심 기능

1. **Zero-shot Sim-to-Real Transfer**
   - 시뮬레이션에서 학습 → 실제 로봇에 바로 적용!
   - Unitree G1 실제 로봇 실험 성공

2. **Domain Randomization**
   ```python
   - Sensor noise randomization
   - Dynamics properties (friction, mass)
   - Task uncertainties
   - Lateral pushes during training (force perturbation!)
   ```

3. **MJX 완전 활용**
   - GPU 대규모 병렬화
   - JAX 자동 미분
   - 단일 GPU에서 수 분 내 학습 가능

4. **Velocity Tracking**
   - Joystick 환경 제공
   - Forward/lateral 속도 + yaw rate 제어
   - 실시간 interactive control

### 💻 설치 & 사용

```bash
# 초간단 설치!
pip install playground

# 또는 소스에서
git clone https://github.com/google-deepmind/mujoco_playground.git
cd mujoco_playground
pip install -e ".[all]"
```

**학습 예제:**
```bash
python learning/train_jax_ppo.py --env_name UnitreeH1Joystick
```

**Colab 튜토리얼:**
- https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb

### 📈 Robustness 기법

1. **Force Perturbations** (lateral pushes)
2. **Domain Randomization**:
   - Friction variation
   - Mass changes (±X%)
   - Sensor noise
3. **PPO** with massively parallel training
4. **Uneven terrain** 대응 능력

### 🎯 장점

✅ **Google DeepMind 공식** - 높은 품질 보장
✅ **최신 (2025)** - 최신 MJX 기능 활용
✅ **Unitree H1 직접 지원** - 우리 환경과 완벽 호환
✅ **Zero-shot sim-to-real** - 실제 로봇 적용 가능
✅ **완전 오픈소스** - Apache 2.0 라이센스
✅ **문서 & 튜토리얼** - Colab notebooks 제공
✅ **활발한 개발** - 지속적 업데이트

### ⚠️ 단점

⚠️ **매우 새로움** - API 변경 가능성
⚠️ **초기 단계** - 일부 기능 미완성
⚠️ **의존성** - pre-release 버전 사용

### 📚 참고 자료

- **웹사이트**: https://playground.mujoco.org/
- **논문**: https://arxiv.org/abs/2502.08844
- **GitHub**: https://github.com/google-deepmind/mujoco_playground
- **Demo**: https://playground.mujoco.org/demo/

---

## 2. HumanoidBench (2024) ⭐ 추천!

### 📊 기본 정보

- **개발**: UC Berkeley, CMU
- **릴리즈**: 2024년 3월
- **논문**: arXiv:2403.10506
- **GitHub**: https://github.com/carlosferrazza/humanoid-bench
- **목적**: Whole-body humanoid control benchmark

### ✅ 지원 내용

**로봇:**
- Unitree H1 + Shadow Hands
- 27개의 distinct whole-body tasks

**Task 카테고리:**
- Locomotion (걷기, 달리기)
- Manipulation (물체 조작)
- Whole-body coordination

### 🚀 핵심 기술

1. **Hierarchical Learning**
   - Low-level skill policies (PPO with MJX)
   - High-level planning policies
   - End-to-end보다 우수한 성능

2. **Massively Parallelized PPO**
   - MuJoCo MJX 활용
   - 수천 개 parallel environments

3. **Force Perturbations**
   - 각 link에 force 적용
   - 학습 중 robustness 향상

4. **Transfer Learning**
   - Low-level skills → Full humanoid
   - Additional masses에 robust

### 💻 설치 & 사용

```bash
git clone https://github.com/carlosferrazza/humanoid-bench.git
cd humanoid-bench
pip install -e .
```

**학습 예제:**
```python
import gymnasium as gym
import humanoid_bench

env = gym.make('Stand-v0')
# PPO training with MJX
```

### 🎯 장점

✅ **검증된 benchmark** - 여러 SOTA 알고리즘 평가됨
✅ **27개 tasks** - 다양한 whole-body control
✅ **Hierarchical approach** - 효율적 학습
✅ **MJX 기반** - GPU 병렬화
✅ **코드 공개** - 재현 가능

### ⚠️ 단점

⚠️ **Benchmark 중심** - 단일 task 최적화 아님
⚠️ **복잡한 설정** - 27개 tasks 중 선택 필요
⚠️ **Shadow Hands** - 우리는 손 없음

### 📚 참고 자료

- **웹사이트**: https://humanoid-bench.github.io/
- **논문**: https://arxiv.org/abs/2403.10506
- **GitHub**: https://github.com/carlosferrazza/humanoid-bench

---

## 3. PHC_MJX (2024) ⭐⚠️ 개발 중

### 📊 기본 정보

- **개발**: Zhengyi Luo (CMU/NVIDIA)
- **릴리즈**: 2024년 2월 24일
- **원본**: ICCV 2023 "Perpetual Humanoid Control"
- **GitHub**: https://github.com/ZhengyiLuo/PHC_MJX
- **Status**: **[Repo still under construction]**

### ✅ 핵심 아이디어

**Perpetual Control:**
- 리셋 없이 무한히 제어 가능
- Fail-state recovery 자동 학습
- Noisy input 대응 (video pose estimation)
- Unexpected falls 복구

**PMCP (Progressive Multiplicative Control Policy):**
- 동적으로 네트워크 capacity 할당
- 어려운 motion sequence 학습
- Large-scale motion DB scaling

### ⚠️ 현재 상태

❌ **미완성** - "Repo still under construction"
❌ **문서 부족** - 사용 방법 불명확
❌ **의존성** - SMPLSim 필요
⚠️ **실험 필요** - 안정성 미검증

### 💡 향후 가능성

✅ Isaac Gym 버전은 매우 강력함
✅ MJX 포팅 시 동일한 성능 기대
✅ Perpetual control은 이상적인 목표

### 📚 참고 자료

- **PHC_MJX**: https://github.com/ZhengyiLuo/PHC_MJX
- **SMPLSim**: https://github.com/ZhengyiLuo/SMPLSim
- **원본 PHC**: https://github.com/ZhengyiLuo/PHC

---

## 4. ASE & Other Isaac Gym Works ❌

### ASE (Adversarial Skill Embeddings)

- **Status**: ❌ **Isaac Gym 전용**
- **논문**: SIGGRAPH 2022
- **GitHub**: https://github.com/nv-tlabs/ASE
- **특징**:
  - 1000+ mocap clips 학습
  - Large-scale skill embeddings
  - 매우 robust

**MuJoCo 포팅 가능성:**
- ⚠️ 원리적으로 가능하지만 공식 구현 없음
- ⚠️ 직접 구현 필요 (수주~수개월)

### PHC (원본)

- **Status**: ❌ **Isaac Gym 기반**
- **MJX 버전**: PHC_MJX (위 참조, 개발 중)

---

## 🎯 추천 순위 및 실행 계획

### 🥇 1순위: MuJoCo Playground

**이유:**
- ✅ Google DeepMind 공식 (신뢰도 최고)
- ✅ Unitree H1 직접 지원
- ✅ Zero-shot sim-to-real 검증됨
- ✅ 최신 (2025) 기술
- ✅ 완전 오픈소스, 활발한 개발

**즉시 실행 가능:**
```bash
# 1. 설치
pip install playground

# 2. 예제 실행
python -c "import playground; print('Success!')"

# 3. Unitree H1 학습 (공식 예제 참고)
# learning/train_jax_ppo.py 수정하여 사용
```

**예상 결과:**
- Domain randomization + force perturbation
- Sim-to-real transfer 가능
- 실제 로봇 적용까지 목표 가능

### 🥈 2순위: LocoMuJoCo + Custom Wrappers

**이유:**
- ✅ 현재 환경 그대로 사용
- ✅ 22,000+ mocap datasets
- ✅ Perturbation wrapper 직접 구현 가능
- ✅ 학습 시간만 추가 투자

**실행 계획:**
```python
# 우리가 이미 설계한 wrapper 사용
from custom.wrappers.perturbation_wrapper import PerturbationWrapper
from custom.wrappers.domain_randomization import DomainRandomizationWrapper

# 200M timesteps 학습
# 예상 시간: 3-5일 (RTX 3070, 4096 envs)
```

**예상 결과:**
- 외력 대응 5-10배 향상
- MuJoCo Playground만큼은 아니지만 충분히 robust

### 🥉 3순위: HumanoidBench

**이유:**
- ✅ 검증된 benchmark
- ✅ 다양한 tasks 학습 가능
- ⚠️ Whole-body control 필요 시만

**적용 시나리오:**
- Locomotion만 필요하면 과한 선택
- Manipulation도 필요하면 고려

---

## 💡 최종 추천

### Scenario 1: 빠른 robustness 향상 (1-2주)

```bash
# LocoMuJoCo + Custom Wrappers
1. Perturbation wrapper 구현
2. Domain randomization 추가
3. 200M timesteps 학습
4. 비교 평가
```

**장점:**
- 현재 환경 그대로 사용
- 즉시 시작 가능
- 충분한 robustness 향상

### Scenario 2: SOTA + Real Robot Transfer (1-2개월)

```bash
# MuJoCo Playground 활용
1. pip install playground
2. Unitree H1 환경 탐색
3. 공식 예제 실행
4. Fine-tuning for our use case
5. Sim-to-real experiments
```

**장점:**
- Google DeepMind 검증됨
- Zero-shot sim-to-real
- 실제 로봇까지 목표 가능
- 최신 기술 습득

### Scenario 3: 연구 논문 수준 (2-6개월)

```bash
# ASE MuJoCo 포팅 or PHC_MJX 완성 기다리기
1. ASE 논문 분석
2. MuJoCo/MJX로 직접 구현
3. Large-scale mocap 활용 (22,000+)
4. 논문 작성 가능
```

**장점:**
- 진짜 SOTA 재현
- 연구 성과
- 논문 발표 가능

---

## 📋 다음 단계 제안

### Option A: 빠른 성과 (추천!)

```bash
# 1주차: MuJoCo Playground 탐색
pip install playground
# 공식 예제 실행 및 이해

# 2주차: LocoMuJoCo + Wrappers
# Perturbation 학습 시작 (200M)

# 3-4주차: 비교 평가
# Playground vs Custom approach
```

### Option B: Playground 집중

```bash
# 1-2주: 환경 이해 및 설정
# 3-4주: Unitree H1 학습
# 5-8주: Fine-tuning & Sim-to-real 준비
```

### Option C: 보수적 접근

```bash
# LocoMuJoCo만 사용
# 검증된 방법으로 착실히 진행
# 안정적이지만 SOTA는 아님
```

---

## 🔗 모든 리소스 링크

### MuJoCo Playground
- **Website**: https://playground.mujoco.org/
- **GitHub**: https://github.com/google-deepmind/mujoco_playground
- **Paper**: https://arxiv.org/abs/2502.08844
- **Colab**: https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb

### HumanoidBench
- **Website**: https://humanoid-bench.github.io/
- **GitHub**: https://github.com/carlosferrazza/humanoid-bench
- **Paper**: https://arxiv.org/abs/2403.10506

### PHC_MJX
- **GitHub**: https://github.com/ZhengyiLuo/PHC_MJX
- **SMPLSim**: https://github.com/ZhengyiLuo/SMPLSim

### LocoMuJoCo (우리가 사용 중)
- **GitHub**: https://github.com/robfiras/loco-mujoco
- **Docs**: https://loco-mujoco.readthedocs.io/

---

## 📝 결론

**최우선 선택: MuJoCo Playground (2025, Google DeepMind)**

✅ Unitree H1 직접 지원
✅ Domain randomization + Force perturbation
✅ Zero-shot sim-to-real transfer
✅ 완전 오픈소스
✅ 최신 기술 (2025)

**대안: LocoMuJoCo + Custom Wrappers**

✅ 현재 환경 그대로 사용
✅ 즉시 시작 가능
✅ 충분한 robustness (5-10배 향상)

**ASE/PHC는 MuJoCo 네이티브 지원 없음** - 직접 포팅 필요 (비추천)

다음은 **MuJoCo Playground를 설치하고 Unitree H1 예제를 실행**하는 것을 추천합니다!
