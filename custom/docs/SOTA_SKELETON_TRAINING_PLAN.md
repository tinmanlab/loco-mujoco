# SkeletonTorque SOTA Training Plan
## Biomechanical + Robust + Multi-motion Humanoid Control

**목표**: `models/skeleton` XML 기반으로 SOTA 수준의 biomechanical humanoid locomotion 달성

**요구사항**:
- ✅ 다양한 모션 (walk, run, jump, recovery, etc.)
- ✅ 넘어지지 않는 안정성
- ✅ 외란에 강한 robustness
- ✅ Biomechanical realism
- ✅ MuJoCo/MJX 기반

---

## 🔍 SOTA 연구 분석 결과

### 최신 접근법 (2024-2025)

| 프로젝트 | 핵심 기법 | Biomechanical | Robustness | MuJoCo |
|---------|----------|---------------|-----------|--------|
| **ALMI** | Upper/Lower adversarial | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| **ResMimic** | Two-stage residual | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ (eval) |
| **HumanoidBench** | Hierarchical RL | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| **MuJoCo Playground** | PPO + DR | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| **LocoMuJoCo** | AMP/GAIL | ⭐⭐⭐⭐ | ⭐⭐ | ✅ |

### 공통 성공 요소

1. **Large-scale motion data** (15,000+ clips)
2. **Multi-stage training** (base skills → task-specific)
3. **Hierarchical control** (low-level + high-level)
4. **Perturbation training** (force + domain randomization)
5. **Recovery behaviors** (explicit fall recovery)

---

## 🎯 SkeletonTorque 최적 접근법

### 왜 단순 PPO + DR로는 부족한가?

❌ **MuJoCo Playground 한계**:
- 단순 velocity tracking만 학습
- Motion quality/naturalness 부족
- Biomechanical constraints 무시
- Recovery behaviors 없음

❌ **현재 AMP 학습 한계**:
- 단일 motion만 학습 (run)
- Perturbation 없음
- Recovery 없음
- Task complexity 부족

✅ **필요한 것**:
- **Natural motion patterns** (biomechanical realism)
- **Diverse skills** (walk, run, jump, recover)
- **Robust to perturbations** (force, mass, friction)
- **Long-term stability** (never fall)
- **Adaptive recovery** (자동 균형 회복)

---

## 🚀 제안: Multi-Stage Hierarchical Training

### Stage 1: Large-Scale Motion Imitation (GMT)
**목표**: Natural, biomechanically realistic movements

```yaml
# conf_stage1_gmt.yaml

experiment:
  name: "SkeletonTorque_GMT_Stage1"

  task_factory:
    name: ImitationFactory
    params:
      # LocoMuJoCo의 22,000+ mocap 최대 활용
      default_dataset_conf:
          task: [walk, run, pace, trot, jump, squat, dance]
      lafan1_dataset_conf:
          task: [walk1_subject1, walk2_subject2, run1_subject1,
                 dance1_subject1, dance2_subject4]
      # AMASS dataset도 추가 가능 (manual download)

  env_params:
    env_name: MjxSkeletonTorque
    headless: True
    horizon: 1000
    # Motion tracking reward (no task-specific goals yet)
    reward_type: MotionTrackingReward  # Pure imitation

  algorithm: AMP  # or GAIL

  # Large-scale training
  total_timesteps: 500e6  # 500M! (약 1주일)
  num_envs: 4096

  # Network capacity 증가 (diverse motions)
  hidden_layers: [1024, 512, 256, 256]

  # Imitation 강도
  proportion_env_reward: 0.0  # Pure motion imitation
  disc_lr: 3e-5
```

**기대 효과**:
- ✅ 15,000+ motion clips 학습
- ✅ Natural, biomechanically realistic movements
- ✅ Diverse skill repertoire
- ✅ "Style" embedding (ASE처럼)

**학습 시간**: ~7-10일 (RTX 3070, 4096 envs, 500M steps)

---

### Stage 2: Robustness Training
**목표**: Perturbation resistance + Recovery behaviors

```yaml
# conf_stage2_robust.yaml

experiment:
  name: "SkeletonTorque_Robust_Stage2"

  # Stage 1 checkpoint 로드
  pretrained_policy: "stage1_gmt/outputs/.../AMPJax_saved.pkl"

  task_factory:
    name: ImitationFactory
    params:
      default_dataset_conf:
          task: [walk, run, getup, rollover]  # Recovery motions 추가!

      # Perturbation wrapper
      wrappers:
        - name: PerturbationWrapper
          params:
            force_range: 150.0  # 더 강한 외력
            force_prob: 0.2     # 20% 확률
            force_duration: 15  # 더 긴 지속
            bodies: [pelvis, torso, left_thigh, right_thigh,
                     left_shoulder, right_shoulder]

        - name: DomainRandomizationWrapper
          params:
            mass_range: 0.3     # ±30%
            friction_range: 0.4 # ±40%
            damping_range: 0.3  # ±30%
            actuator_range: 0.15 # ±15%

        - name: ObservationNoiseWrapper
          params:
            position_noise: 0.02
            velocity_noise: 0.15
            imu_noise: 0.08

  env_params:
    env_name: MjxSkeletonTorque
    headless: True
    horizon: 2000  # 더 긴 episode (recovery 테스트)

    # Task reward 추가
    reward_type: CompositeReward
    reward_components:
      - type: MotionTrackingReward
        weight: 0.5  # Motion quality 유지
      - type: StabilityReward
        weight: 0.3  # 넘어지지 않기
      - type: RecoveryReward
        weight: 0.2  # 균형 회복

  # Fine-tuning 설정
  total_timesteps: 300e6  # 300M
  lr: 2e-5  # Lower lr (fine-tuning)

  # Curriculum learning
  curriculum:
    enabled: true
    initial_force: 20.0
    final_force: 150.0
    steps: 100e6
```

**기대 효과**:
- ✅ 150N 외력 대응
- ✅ 자동 균형 회복 (recovery motions)
- ✅ 다양한 지형/조건 대응
- ✅ Long-term stability (2000+ steps)

**학습 시간**: ~5-7일

---

### Stage 3: Hierarchical Task Learning
**목표**: High-level planning + Low-level execution

```yaml
# conf_stage3_hierarchical.yaml

experiment:
  name: "SkeletonTorque_Hierarchical_Stage3"

  # Two-level architecture
  architecture: Hierarchical

  # Low-level policy (from Stage 2)
  low_level:
    pretrained: "stage2_robust/outputs/.../AMPJax_saved.pkl"
    frozen: false  # Allow fine-tuning
    control_frequency: 50  # Hz

  # High-level policy (새로 학습)
  high_level:
    network: [256, 128]
    control_frequency: 5  # Hz (10x slower)

    # Latent skill selection
    skill_dim: 32  # Skill embedding dimension

    # High-level observations
    obs_space:
      - target_velocity  # [vx, vy, vyaw]
      - terrain_height_map
      - external_forces  # Privileged info
      - center_of_mass

    # High-level actions
    action_space:
      - skill_embedding  # [32-dim]
      - gait_phase  # [0-1]
      - step_frequency  # [0.5-2.0 Hz]

  # Task-specific training
  task_factory:
    name: TaskFactory
    params:
      tasks:
        - VelocityTracking
        - TerrainNavigation
        - ObstacleAvoidance
        - PushRecovery

  # Asymmetric Actor-Critic
  asymmetric:
    enabled: true
    critic_obs_extra:
      - ground_truth_friction
      - mass_distribution
      - future_perturbations  # 5 steps ahead

  total_timesteps: 200e6
```

**기대 효과**:
- ✅ Adaptive skill selection
- ✅ Long-horizon planning
- ✅ Complex task execution
- ✅ Generalization to new tasks

**학습 시간**: ~4-5일

---

## 🛠️ 구현 세부사항

### 1. Custom Wrappers

```python
# custom/wrappers/perturbation_wrapper_mjx.py

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any


class PerturbationWrapperMJX:
    """
    MJX 환경에 랜덤 외력 적용
    - Curriculum learning 지원
    - Body-specific force application
    - Recovery behavior triggering
    """

    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config

        # Curriculum settings
        self.curriculum_enabled = config.get('curriculum', {}).get('enabled', False)
        self.initial_force = config.get('curriculum', {}).get('initial_force', 20.0)
        self.final_force = config.get('force_range', 100.0)
        self.curriculum_steps = config.get('curriculum', {}).get('steps', 100e6)

        # Perturbation settings
        self.force_prob = config.get('force_prob', 0.1)
        self.force_duration = config.get('force_duration', 10)
        self.body_names = config.get('bodies', ['pelvis', 'torso'])

        # Get body IDs from names
        self.body_ids = [self.env.model.body(name).id
                        for name in self.body_names
                        if name in self.env.model.names]

    @partial(jax.jit, static_argnums=(0,))
    def get_current_force_range(self, timestep):
        """Curriculum learning: gradually increase force"""
        if not self.curriculum_enabled:
            return self.final_force

        progress = jnp.minimum(timestep / self.curriculum_steps, 1.0)
        current_force = (self.initial_force +
                        (self.final_force - self.initial_force) * progress)
        return current_force

    @partial(jax.jit, static_argnums=(0,))
    def apply_perturbation(self, env_state, timestep, rng):
        """Apply random external force to random body"""
        rng, force_rng, body_rng, dir_rng = jax.random.split(rng, 4)

        # Check if we should apply force
        should_apply = jax.random.bernoulli(force_rng, self.force_prob)

        # Get current force range (curriculum)
        force_range = self.get_current_force_range(timestep)

        # Random force magnitude and direction
        force_mag = jax.random.uniform(force_rng, minval=0.0, maxval=force_range)
        force_dir = jax.random.normal(dir_rng, shape=(3,))
        force_dir = force_dir / jnp.linalg.norm(force_dir)  # Normalize
        force = force_mag * force_dir

        # Random body selection
        body_idx = jax.random.randint(body_rng, shape=(),
                                     minval=0, maxval=len(self.body_ids))
        body_id = self.body_ids[body_idx]

        # Apply force to env_state
        xfrc_applied = env_state.data.xfrc_applied.at[body_id, :3].set(
            jnp.where(should_apply, force, jnp.zeros(3))
        )

        env_state = env_state.replace(
            data=env_state.data.replace(xfrc_applied=xfrc_applied)
        )

        return env_state, rng
```

### 2. Recovery Reward

```python
# custom/rewards/recovery_reward.py

import jax.numpy as jnp


def recovery_reward(obs, prev_obs, data, config):
    """
    Reward for recovering from near-fall states
    """
    # Get pelvis height and tilt
    pelvis_height = data.qpos[2]  # Assuming 2 is z-coordinate
    pelvis_tilt = jnp.abs(data.qpos[3:6])  # Roll, pitch, yaw

    # Get COM velocity
    com_vel = jnp.linalg.norm(data.qvel[:3])

    # Near-fall detection
    min_height = config.get('min_safe_height', 0.8)
    max_tilt = config.get('max_safe_tilt', 0.3)

    is_near_fall = jnp.logical_or(
        pelvis_height < min_height,
        jnp.max(pelvis_tilt) > max_tilt
    )

    # Previous state
    prev_pelvis_height = prev_obs[2]

    # Recovery detected if:
    # 1. Was near-fall
    # 2. Now recovering (height increasing, tilt decreasing)
    height_improvement = pelvis_height - prev_pelvis_height
    is_recovering = jnp.logical_and(
        is_near_fall,
        height_improvement > 0.0
    )

    # Reward recovery effort
    recovery_reward = jnp.where(
        is_recovering,
        height_improvement * 10.0,  # Encourage height recovery
        0.0
    )

    # Bonus for successful recovery
    recovery_success = jnp.logical_and(
        is_recovering,
        pelvis_height > min_height
    )
    recovery_bonus = jnp.where(recovery_success, 5.0, 0.0)

    return recovery_reward + recovery_bonus
```

### 3. Composite Reward

```python
# custom/rewards/composite_reward.py

from loco_mujoco.core.reward import RewardBase


class CompositeReward(RewardBase):
    """
    Multi-objective reward combining:
    - Motion tracking (biomechanical realism)
    - Stability (don't fall)
    - Recovery (balance recovery)
    - Task progress (velocity tracking, etc.)
    """

    def __init__(self, env, config):
        super().__init__(env)

        # Component weights
        self.weights = {
            'motion_tracking': config.get('motion_weight', 0.4),
            'stability': config.get('stability_weight', 0.3),
            'recovery': config.get('recovery_weight', 0.2),
            'task': config.get('task_weight', 0.1)
        }

        # Individual reward components
        from loco_mujoco.core.reward.imitation import MotionTrackingReward
        from custom.rewards.recovery_reward import recovery_reward

        self.motion_reward = MotionTrackingReward(env, config)
        self.recovery_fn = recovery_reward

    def __call__(self, obs, action, next_obs, absorbing):
        """Compute composite reward"""

        # 1. Motion tracking (biomechanical realism)
        motion_rew = self.motion_reward(obs, action, next_obs, absorbing)

        # 2. Stability reward
        pelvis_height = next_obs[2]
        min_height = 0.8
        stability_rew = jnp.where(
            pelvis_height > min_height,
            1.0,
            -10.0  # Heavy penalty for falling
        )

        # 3. Recovery reward
        recovery_rew = self.recovery_fn(
            next_obs, obs, self.env._data, self.config
        )

        # 4. Task reward (velocity tracking, etc.)
        target_vel = self.env.goal  # Assuming velocity goal
        actual_vel = next_obs[10:13]  # Assuming COM velocity
        vel_error = jnp.linalg.norm(target_vel - actual_vel)
        task_rew = jnp.exp(-vel_error)

        # Weighted sum
        total_reward = (
            self.weights['motion_tracking'] * motion_rew +
            self.weights['stability'] * stability_rew +
            self.weights['recovery'] * recovery_rew +
            self.weights['task'] * task_rew
        )

        return total_reward
```

---

## 📊 예상 결과

### Stage 1 완료 후:
- ✅ 15,000+ motion clips 재현 가능
- ✅ Natural, biomechanically realistic movements
- ✅ Diverse skill repertoire (walk, run, jump, dance, etc.)
- ⚠️ Still fragile to perturbations

### Stage 2 완료 후:
- ✅ 150N 외력 대응 (현재 대비 **10-15배 향상**)
- ✅ 자동 균형 회복 (recovery rate 80%+)
- ✅ 다양한 지형/조건 robust
- ✅ Episode length 5000+ steps (never fall)

### Stage 3 완료 후:
- ✅ Complex task execution (obstacle navigation, etc.)
- ✅ Adaptive skill selection
- ✅ Long-horizon planning
- ✅ **SOTA 수준 달성**

---

## 🎯 Timeline

```
Week 1-2: Stage 1 GMT 준비 및 시작
  - Wrapper 구현
  - Large-scale dataset 준비 (22,000+ mocap)
  - Training launch (500M steps, ~10일)

Week 3: Stage 1 학습 중 + Stage 2 준비
  - Perturbation wrapper 구현
  - Recovery reward 구현
  - Composite reward 구현

Week 4-5: Stage 2 Robustness 학습
  - Stage 1 checkpoint 로드
  - Perturbation training (300M steps, ~7일)

Week 6: Stage 2 평가 + Stage 3 준비
  - Robustness 테스트 (force application)
  - Hierarchical architecture 설계

Week 7-8: Stage 3 Hierarchical 학습
  - Two-level policy training (200M steps, ~5일)

Week 9: 최종 평가 및 튜닝
  - Sim-to-real 준비
  - Biomechanical validation
  - Performance benchmarking

Total: ~9주 (2개월)
```

---

## 💡 왜 이 접근이 최선인가?

### 1. LocoMuJoCo의 강점 최대 활용
- ✅ 22,000+ mocap datasets
- ✅ Biomechanically realistic skeletons
- ✅ MJX GPU parallelization
- ✅ Proven imitation learning algorithms (AMP, GAIL)

### 2. SOTA 연구 방법론 통합
- ✅ ResMimic의 two-stage approach
- ✅ ALMI의 adversarial learning concept
- ✅ HumanoidBench의 hierarchical control
- ✅ MuJoCo Playground의 perturbation training

### 3. Biomechanical Realism
- ✅ Large-scale motion imitation (Stage 1)
- ✅ Natural movement patterns
- ✅ Contact dynamics
- ✅ Energy-efficient gaits

### 4. SOTA Robustness
- ✅ Perturbation training (150N forces)
- ✅ Domain randomization (mass, friction, etc.)
- ✅ Recovery behaviors (explicit learning)
- ✅ Curriculum learning (gradual difficulty)

### 5. 실행 가능성
- ✅ 모든 요소를 LocoMuJoCo에서 구현 가능
- ✅ 기존 인프라 활용 (MJX, JAX, 4096 envs)
- ✅ 단계별 검증 가능
- ✅ 2개월 내 완료 가능

---

## 🔗 참고 자료

### 핵심 논문
1. **ALMI** (2024): https://arxiv.org/abs/2504.14305
2. **ResMimic** (2024): https://arxiv.org/abs/2510.05070
3. **HumanoidBench** (2024): https://arxiv.org/abs/2403.10506
4. **LocoMuJoCo** (2023): https://arxiv.org/abs/2311.02496

### 코드 베이스
- **ALMI**: https://github.com/TeleHuman/ALMI-Open
- **LocoMuJoCo**: https://github.com/robfiras/loco-mujoco
- **MuJoCo Playground**: https://github.com/google-deepmind/mujoco_playground

### Datasets
- **AMASS**: https://amass.is.tue.mpg.de/
- **LAFAN1**: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- **LocoMuJoCo Default**: Auto-download

---

## 📝 다음 단계

### Option A: 전체 Pipeline 구현 (추천!)
```bash
# 1. Wrapper 구현 (1-2일)
custom/wrappers/perturbation_wrapper_mjx.py
custom/wrappers/domain_randomization.py
custom/rewards/recovery_reward.py
custom/rewards/composite_reward.py

# 2. Stage 1 시작 (10일)
python custom/training/stage1_gmt/train.py

# 3. Stage 2 (7일)
# 4. Stage 3 (5일)
# 5. 평가
```

### Option B: 단계별 검증
```bash
# 먼저 Stage 1만 구현 및 검증
# 성공 후 Stage 2, 3 순차 진행
```

### Option C: Simplified Version
```bash
# Hierarchical 없이 Stage 1 + 2만
# 충분한 robustness 달성 가능
```

---

## 🎯 결론

**SkeletonTorque SOTA 학습을 위한 최선의 방법:**

✅ **Multi-Stage Training**:
1. Large-scale motion imitation (GMT)
2. Perturbation + Recovery training
3. Hierarchical task learning

✅ **LocoMuJoCo 최대 활용**:
- 22,000+ mocap datasets
- MJX GPU parallelization
- Biomechanical skeletons

✅ **SOTA 방법론 통합**:
- ResMimic two-stage
- ALMI adversarial
- HumanoidBench hierarchical
- Playground perturbation

✅ **2개월 내 완료 가능**
✅ **Biomechanical + Robust + Multi-motion**

**다음: Stage 1 GMT wrapper 구현부터 시작!**
