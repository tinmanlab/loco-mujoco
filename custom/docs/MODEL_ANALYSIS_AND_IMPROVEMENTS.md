# 현재 모델 분석 및 개선 방안

## 📊 현재 학습된 모델 분석

### UnitreeH1 AMP Training

**학습 기법: AMP (Adversarial Motion Priors)**
- **알고리즘**: Imitation Learning + PPO
- **핵심 아이디어**: Discriminator를 사용해 mocap data의 "style"을 학습
- **Reward**: Task reward (50%) + Style reward from discriminator (50%)
- **Dataset**: "run" mocap (달리기 동작)
- **Timesteps**: 75M
- **환경 수**: 4096 parallel environments (RTX 3070 최적화)

**학습 설정 (conf.yaml 분석):**
```yaml
Algorithm: AMP
Dataset: run (default LocoMuJoCo dataset)
Total timesteps: 75M
Learning rate: 6e-5
Discriminator lr: 5e-5
Network: [512, 256] hidden layers
```

**결과:**
- ✅ Mean Episode Return: 154.12
- ✅ Mean Episode Length: 965.83
- ✅ 달리기 동작을 자연스럽게 수행
- ❌ **외력에 매우 취약** - 조금만 밀어도 넘어짐
- ❌ 한 가지 동작만 학습 (run)

---

## 🔍 왜 외력에 약한가?

### 1. **Perturbation Training 없음**
현재 학습 설정에 **외력 적용이 전혀 포함되지 않음**:
- 학습 중 무작위 외력 적용 ❌
- Domain randomization ❌
- Adversarial perturbation ❌

→ 로봇이 "완벽한 환경"에서만 걷는 법을 배움

### 2. **AMP의 한계**
AMP는 **mocap trajectory를 따라가는 것**이 목표:
- 외력 대응이 reward에 포함되지 않음
- "예쁘게 걷기"만 학습, "균형 회복"은 학습하지 않음

### 3. **단일 동작만 학습**
- "run" mocap만 사용
- 다양한 상황 대응 능력 부족
- Recovery motion 없음

---

## 🎯 개선 방안

### Level 1: 기본 Robustness 향상 (교육/연구용 → 실용 연구용)

#### 1.1 Perturbation Training 추가
**Custom wrapper 구현 필요:**

```python
class PerturbationWrapper:
    def __init__(self, env, force_range=50.0, prob=0.1):
        self.env = env
        self.force_range = force_range  # 최대 힘 (N)
        self.prob = prob  # 매 step 확률

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # 랜덤으로 외력 적용
        if np.random.rand() < self.prob:
            body_id = np.random.randint(self.env.model.nbody)
            force = np.random.uniform(-self.force_range,
                                     self.force_range, 3)
            self.env.data.xfrc_applied[body_id, :3] = force

        return obs, reward, done, info
```

**효과:**
- ✅ 외력에 대한 대응 능력 학습
- ✅ 균형 회복 능력 향상
- ⚠️ 학습 시간 증가 (100M+ timesteps 필요)

#### 1.2 다양한 Motion 학습
```yaml
default_dataset_conf:
    task: [walk, run, pace, trot]  # 여러 동작 동시 학습
```

**효과:**
- ✅ 다양한 속도/스타일로 이동 가능
- ✅ 상황에 맞는 동작 선택 능력
- ✅ 전이 학습(transition) 능력 향상

#### 1.3 Domain Randomization
```yaml
env_params:
  randomize_friction: true      # 마찰력 랜덤화
  randomize_mass: true          # 질량 랜덤화 (±20%)
  randomize_actuator: true      # 액추에이터 노이즈
```

**효과:**
- ✅ Sim-to-real gap 감소
- ✅ 다양한 환경에서 robust
- ✅ 실제 로봇 적용 가능성 증가

---

### Level 2: State-of-the-Art Robustness (연구 최전선)

#### 2.1 **ASE (Adversarial Skill Embeddings)**
- **논문**: "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" (SIGGRAPH 2022)
- **특징**:
  - 다양한 mocap 동작 학습 (1000+ clips)
  - Latent skill embedding space
  - High-level task + Low-level skill 분리
- **장점**:
  - ✅ 매우 다양한 동작 레퍼토리
  - ✅ 새로운 task에 빠르게 적응
  - ✅ Robust한 recovery behaviors

#### 2.2 **PHC (Perpetual Humanoid Control)**
- **논문**: "Perpetual Humanoid Control for Real-time Simulated Avatars" (ICCV 2023)
- **특징**:
  - Self-supervised learning
  - Real-time performance
  - Long-term stability
- **장점**:
  - ✅ 무한히 안정적인 제어
  - ✅ 실시간 interactive control
  - ✅ 외력 대응 능력 탁월

#### 2.3 **CALM (Composable Adversarial Learning for Motion)**
- **특징**:
  - Compositional motion primitives
  - Hierarchical policy structure
  - Adaptive recovery behaviors
- **장점**:
  - ✅ Motion primitive 조합 가능
  - ✅ 자동 recovery 학습
  - ✅ 매우 robust

---

### Level 3: 상용/실제 로봇 수준

#### 3.1 **Hierarchical Control**
```
High-level Policy (Task Planning)
    ↓
Mid-level Policy (Motion Selection)
    ↓
Low-level Policy (Joint Control)
```

**구현 방법:**
- Teacher-Student training
- Curriculum learning
- Meta-learning for fast adaptation

**효과:**
- ✅ 복잡한 task 수행 가능
- ✅ Long-horizon planning
- ✅ Human-like behavior

#### 3.2 **Model Predictive Control (MPC) Hybrid**
```
Learning-based Policy + MPC
    ↓
Whole-body trajectory optimization
    ↓
Safety constraints enforcement
```

**효과:**
- ✅ 안전성 보장 (safety constraints)
- ✅ 물리적으로 타당한 동작
- ✅ 최적화된 에너지 효율

#### 3.3 **Sim-to-Real Transfer**
- **Domain Adaptation**:
  - System Identification
  - Residual Policy Learning
  - Privileged Information Training
- **Real Robot Testing**:
  - Safety controller overlay
  - Gradual deployment
  - Real-world data fine-tuning

---

## 📚 현재 사용 중인 기법의 위치

### 학습 기법 발전 타임라인:

```
2018: DeepMimic (Original motion imitation)
       ↓
2021: AMP (Adversarial Motion Priors) ← 현재 사용 중!
       ↓
2022: ASE (Large-scale skill embeddings)
       ↓
2023: PHC (Perpetual control)
       ↓
2024: Diffusion policies, Foundation models
```

**현재 수준 평가:**
- 🎓 **교육/연구용**: ✅ 적합
- 🔬 **고급 연구용**: ⚠️ 부분 적합 (robustness 부족)
- 🏭 **상용/실제 로봇**: ❌ 부적합 (안전성, 강건성 부족)

---

## 💡 빠른 개선을 위한 추천 방안

### 즉시 적용 가능 (1-2일):

1. **Perturbation Wrapper 구현**
   - 파일: `custom/wrappers/perturbation.py`
   - 학습 중 랜덤 외력 적용
   - 비교적 쉬운 구현

2. **다양한 mocap 사용**
   ```yaml
   default_dataset_conf:
       task: [walk, run, pace]
   lafan1_dataset_conf:
       task: [walk1_subject1, run1_subject1]
   ```

3. **학습 시간 연장**
   - 75M → 150M timesteps
   - 더 많은 데이터로 일반화 능력 향상

### 단기 목표 (1-2주):

1. **ASE 논문 구현 시도**
   - GitHub에 공개된 구현 참고
   - loco-mujoco에 맞게 수정

2. **Domain Randomization 추가**
   - 물리 파라미터 랜덤화
   - 노이즈 추가

3. **Recovery Policy 별도 학습**
   - Falling → Recovery 전용 policy
   - Main policy와 통합

### 장기 목표 (1-2개월):

1. **Hierarchical RL 구현**
   - High-level + Low-level 분리
   - Meta-learning 적용

2. **실제 로봇 테스트 준비**
   - Sim-to-real 기법 적용
   - Safety layer 구현

---

## 🔗 참고 자료

### 핵심 논문:
1. **AMP** (현재 사용): [arXiv:2104.02180](https://arxiv.org/abs/2104.02180)
2. **ASE**: [arXiv:2205.01906](https://arxiv.org/abs/2205.01906)
3. **PHC**: [arXiv:2305.06456](https://arxiv.org/abs/2305.06456)
4. **DeepMimic**: [arXiv:1804.02717](https://arxiv.org/abs/1804.02717)

### 구현 코드:
- AMP Official: https://github.com/xbpeng/DeepMimic
- ASE: https://github.com/nv-tlabs/ASE
- PHC: https://github.com/ZhengyiLuo/PHC

### LocoMuJoCo 관련:
- 공식 문서: https://loco-mujoco.readthedocs.io/
- 예제: `/home/tinman/loco-mujoco/examples/`
- 22,000+ mocap datasets 활용 가능!

---

## 📝 다음 단계 제안

### 실험 1: Perturbation Training
```bash
# 1. Perturbation wrapper 구현
# 2. conf.yaml 수정
# 3. 150M timesteps 학습
# 4. 비교 평가
```

**예상 결과:**
- 외력 대응 능력 3-5배 향상
- Episode length 증가 (더 오래 서있음)
- Recovery rate 향상

### 실험 2: Multi-task Learning
```yaml
default_dataset_conf:
    task: [walk, run, pace, trot]
```

**예상 결과:**
- 다양한 속도 제어 가능
- 자연스러운 전환(transition)
- Generalization 능력 향상

### 실험 3: ASE 구현
```
1. ASE 코드 분석
2. LocoMuJoCo 환경에 맞게 수정
3. 대규모 mocap dataset 활용 (22,000+ clips)
4. 비교 실험
```

**예상 결과:**
- State-of-the-art robustness
- 매우 다양한 동작 레퍼토리
- 연구 논문 수준 결과

---

**결론:**
현재 모델은 **교육용/기초 연구용으로 적합**하지만, 실용성을 위해서는 **perturbation training과 더 advanced한 알고리즘**이 필요합니다. 가장 빠른 개선 방법은 **Perturbation Wrapper를 추가하고 학습 시간을 늘리는 것**입니다.
