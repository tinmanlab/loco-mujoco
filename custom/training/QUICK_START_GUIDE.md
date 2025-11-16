# SkeletonTorque 빠른 개선 가이드

**목표**: 빠르게 시도 가능한 것부터 단계적으로 진행

---

## 📋 Phase Overview

```
Phase 1: 다양한 Mocap (즉시 시작!)     → 2-3일
Phase 2: 간단한 Perturbation           → 1일 구현 + 5-7일 학습
Phase 3: Recovery + Advanced           → 추후 진행
```

---

## 🚀 Phase 1: Multi-Skill Learning (즉시 실행!)

### 무엇이 개선되나?

**현재 문제:**
- ✅ run 모션만 학습됨
- ❌ 다른 속도/스타일 불가능
- ❌ 제한적인 동작 레퍼토리

**Phase 1 후:**
- ✅ walk, run, pace 모두 가능
- ✅ 다양한 속도 제어
- ✅ 자연스러운 전환(transition)
- ✅ 더 넓은 일반화 능력

### 실행 방법

```bash
# 1. 환경 활성화
conda activate loco-mujoco

# 2. WandB offline 설정 (선택)
export WANDB_MODE=offline

# 3. 학습 시작!
cd /home/tinman/loco-mujoco/custom/training/skeleton_multiskill
python experiment.py
```

### 설정 요약

- **Motions**: walk, run, pace (3가지)
- **Timesteps**: 100M (~2-3일, RTX 3070)
- **Environments**: 4096 (병렬)
- **Algorithm**: AMP

### 예상 결과

```
학습 완료 후:
- walk: 느린 속도 (0.5-1.0 m/s)
- run: 빠른 속도 (2.0-3.0 m/s)
- pace: 중간 속도 (1.0-2.0 m/s)
```

### 추가 Mocap 옵션

더 많은 동작을 원하면 conf.yaml 수정:

```yaml
default_dataset_conf:
    task: [walk, run, pace, trot, jump]  # 5가지!

# LAFAN1 dataset도 추가
lafan1_dataset_conf:
    task: [walk1_subject1, run1_subject1, dance1_subject1]
```

**주의**: 더 많은 모션 = 더 긴 학습 시간 필요
- 3개 모션: 100M timesteps (~2-3일)
- 5개 모션: 150M timesteps (~3-4일)
- 10개 모션: 200M+ timesteps (~5-7일)

---

## ⚡ Phase 2: Simple Perturbation Training

### 무엇이 개선되나?

**Phase 1 후 문제:**
- ✅ 다양한 모션 가능
- ❌ 여전히 외력에 약함
- ❌ 조금만 밀면 넘어짐

**Phase 2 후:**
- ✅ 외력 50-100N 대응 (현재 대비 5-10배!)
- ✅ 균형 유지 능력 향상
- ✅ Episode length 2-3배 증가

### 간단한 Perturbation Wrapper

**파일**: `custom/wrappers/simple_perturbation.py`

```python
#!/usr/bin/env python3
"""
간단한 Perturbation Wrapper
- MJX 환경에서 작동
- 랜덤 외력 적용
- 최소한의 코드
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any


def create_perturbation_env(base_env, force_range=50.0, force_prob=0.1):
    """
    기존 환경을 wrapping하여 perturbation 추가

    Args:
        base_env: LocoMuJoCo MJX environment
        force_range: 최대 힘 크기 (N)
        force_prob: 매 step 적용 확률 (0-1)

    Returns:
        Wrapped environment
    """

    class PerturbedEnv:
        def __init__(self, env):
            self.env = env
            self.force_range = force_range
            self.force_prob = force_prob

            # Get body IDs for perturbation
            # SkeletonTorque: pelvis, torso
            self.perturb_bodies = [0, 1]  # pelvis=0, torso=1

        def reset(self, rng):
            return self.env.reset(rng)

        def step(self, env_state, action, rng):
            """Step with random perturbations"""

            # 1. 원래 step 먼저
            obs, reward, done, info, env_state = self.env.step(
                env_state, action
            )

            # 2. 확률적으로 perturbation 적용
            rng, force_rng, body_rng = jax.random.split(rng, 3)

            # Apply force?
            should_apply = jax.random.bernoulli(
                force_rng, self.force_prob
            )

            # Random force direction
            force = jax.random.normal(force_rng, shape=(3,))
            force = force / jnp.linalg.norm(force) * self.force_range

            # Random body
            body_id = jax.random.choice(
                body_rng, jnp.array(self.perturb_bodies)
            )

            # Apply to xfrc_applied
            xfrc = env_state.data.xfrc_applied
            xfrc = xfrc.at[body_id, :3].set(
                jnp.where(should_apply, force, jnp.zeros(3))
            )

            env_state = env_state.replace(
                data=env_state.data.replace(xfrc_applied=xfrc)
            )

            return obs, reward, done, info, env_state

        def __getattr__(self, name):
            """Delegate to base env"""
            return getattr(self.env, name)

    return PerturbedEnv(base_env)
```

### 사용 방법

**수정할 파일**: `experiment.py`

```python
# experiment.py 상단에 추가
from custom.wrappers.simple_perturbation import create_perturbation_env

# env 생성 후 wrapping
env = factory.make(**config.experiment.env_params,
                   **config.experiment.task_factory.params)

# Perturbation 추가!
env = create_perturbation_env(
    env,
    force_range=50.0,   # 50N 외력
    force_prob=0.1      # 10% 확률
)

# 나머지 코드는 동일...
```

### 실행

```bash
# 1. Wrapper 만들기
mkdir -p custom/wrappers
# simple_perturbation.py 생성 (위 코드 복사)

# 2. experiment.py 수정
# (위의 코드 추가)

# 3. 새 training 폴더
mkdir -p custom/training/skeleton_perturbation
cp custom/training/skeleton_multiskill/conf.yaml custom/training/skeleton_perturbation/

# 4. conf.yaml 수정
# total_timesteps: 150e6  # 조금 더 길게

# 5. 학습!
cd custom/training/skeleton_perturbation
python experiment.py
```

### 예상 결과

```
Before Phase 2:
- 10N 외력: 즉시 넘어짐
- Episode length: ~1000 steps

After Phase 2:
- 50N 외력: 대응 가능!
- Episode length: ~2500 steps
- Recovery attempts: 가끔 성공
```

---

## 🔄 Phase 3: Advanced (추후)

Phase 1, 2 성공 후 진행:

### 3.1: Recovery Motions

```yaml
default_dataset_conf:
    task: [walk, run, pace, getup, rollover]
```

### 3.2: Stronger Perturbation

```python
force_range=100.0   # 100N
force_prob=0.15     # 15%
```

### 3.3: Domain Randomization

```python
# Mass, friction, damping randomization
```

### 3.4: Hierarchical Control

```python
# Low-level + High-level policies
```

---

## 📊 단계별 예상 성과

| Phase | 외력 대응 | Episode 길이 | 학습 시간 | 난이도 |
|-------|----------|-------------|----------|--------|
| **현재** | ~10N | 1000 | - | - |
| **Phase 1** | ~10N | 1200 | 2-3일 | ⭐ |
| **Phase 2** | ~50N | 2500 | 5-7일 | ⭐⭐ |
| **Phase 3.1** | ~100N | 4000 | 7-10일 | ⭐⭐⭐ |
| **Phase 3.2** | ~150N | 5000+ | 추가 5일 | ⭐⭐⭐⭐ |

---

## ⚡ Quick Start Commands

### Phase 1 (지금 바로!)

```bash
conda activate loco-mujoco
cd /home/tinman/loco-mujoco/custom/training/skeleton_multiskill
python experiment.py
```

### 진행 상황 확인

```bash
# WandB offline 결과
ls -lh wandb/offline-*

# 또는 Jupyter/TensorBoard로 모니터링
```

### 학습 중단/재개

```bash
# Ctrl+C로 중단
# 재개: Hydra가 자동으로 checkpoint 관리
# (LocoMuJoCo는 기본적으로 checkpoint 저장)
```

---

## 🎯 우선순위 추천

### 가장 빠른 성과

```
1. Phase 1 시작 (지금!)
2. 2-3일 후 결과 확인
3. 만족스러우면 Phase 2 준비
4. Phase 2 wrapper 구현 (1일)
5. Phase 2 학습 (5-7일)
```

### 최대 robustness

```
1. Phase 1
2. Phase 2
3. Phase 3.1 (recovery)
4. Phase 3.2 (stronger perturbation)
→ Total: ~3-4주
```

### 균형잡힌 접근

```
1. Phase 1 (다양한 모션)
2. Phase 2 (간단한 perturbation)
3. 평가 후 Phase 3 결정
→ Total: ~2주
```

---

## 📝 체크리스트

### Phase 1 시작 전

- [ ] conda 환경 활성화 확인
- [ ] `/home/tinman/loco-mujoco/custom/training/skeleton_multiskill/` 존재 확인
- [ ] `conf.yaml` 확인
- [ ] `experiment.py` 존재 확인
- [ ] GPU 사용 가능 확인 (`nvidia-smi`)

### Phase 1 실행

- [ ] `python experiment.py` 실행
- [ ] 초기 출력 확인 (환경 생성, dataset 로딩)
- [ ] GPU 사용률 확인 (`nvidia-smi`)
- [ ] 예상 완료 시간: 2-3일

### Phase 2 준비

- [ ] Phase 1 완료 확인
- [ ] Checkpoint 저장 위치 확인
- [ ] `simple_perturbation.py` 생성
- [ ] `experiment.py` 수정
- [ ] 테스트 실행

---

## 🐛 문제 해결

### GPU 메모리 부족

```yaml
# conf.yaml에서
num_envs: 2048  # 4096 → 2048로 줄이기
```

### 학습 너무 느림

```yaml
# Timesteps 줄이기
total_timesteps: 50e6  # 100M → 50M
```

### WandB 에러

```bash
export WANDB_MODE=offline
# 또는 conf.yaml에서 wandb 비활성화
```

### Import 에러

```bash
# 현재 디렉토리에서 실행하는지 확인
cd /home/tinman/loco-mujoco
python custom/training/skeleton_multiskill/experiment.py
```

---

## 💡 Tips

1. **학습 중 모니터링**:
   ```bash
   watch -n 1 nvidia-smi  # GPU 상태
   tail -f wandb/debug.log  # WandB 로그
   ```

2. **Checkpoint 위치**:
   ```
   custom/training/skeleton_multiskill/outputs/YYYY-MM-DD/HH-MM-SS/
   ```

3. **빠른 테스트**:
   ```yaml
   total_timesteps: 1e6  # 1M (몇 분)
   debug: true
   ```

4. **Phase 1과 2 동시 진행**:
   - Phase 1 학습 중에 Phase 2 wrapper 구현
   - 시간 절약!

---

## 🚀 지금 바로 시작!

```bash
# 1. 터미널 열기
# 2. 다음 명령어 실행:

conda activate loco-mujoco
cd /home/tinman/loco-mujoco/custom/training/skeleton_multiskill
python experiment.py

# 3. GPU 모니터링 (새 터미널)
watch -n 1 nvidia-smi

# 4. 2-3일 후 결과 확인!
```

**Good luck!** 🎉
