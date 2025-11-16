# LocoMuJoCo 설치 및 최적화 요약

## 📦 설치 완료 내역

### 1. 환경 정보
- **환경 이름**: `loco-mujoco`
- **Python 버전**: 3.11.14
- **패키지 관리**: Conda
- **loco-mujoco 버전**: 1.0.1 (editable mode)
- **JAX 버전**: 0.7.1 with CUDA 12 support
- **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM)
- **CUDA 버전**: 13.0 (호환)

### 2. 설치된 주요 패키지
- **MuJoCo**: 3.2.7
- **MuJoCo MJX**: 3.2.7 (JAX 병렬 환경)
- **JAX/JAXlib**: 0.7.1
- **JAX CUDA Plugin**: 0.7.1
- **Gymnasium**: 1.2.1
- **Flax**: 0.12.0 (신경망 라이브러리)
- **Optax**: 0.2.6 (최적화)
- **기타**: hydra-core, wandb 호환 준비

---

## 🚀 환경 사용법

### 환경 활성화
```bash
conda activate loco-mujoco
```

### 환경 비활성화
```bash
conda deactivate
```

### Python 실행 (환경 활성화 후)
```bash
python your_script.py
```

---

## ✅ 테스트 결과

### 1. 기본 환경 테스트 ✓
- **UnitreeH1 환경**: 정상 작동
- **Action dim**: 19
- **Observation dim**: 49
- **Dataset 자동 다운로드**: 성공

### 2. MJX GPU 가속 테스트 ✓
- **병렬 환경 수**: 64개
- **총 스텝 수**: 64,000 steps
- **실행 시간**: 6.07초
- **성능**: 10,551 steps/sec
- **FPS per env**: 164.9
- **GPU 인식**: CudaDevice(id=0) ✓

### 3. 다중 환경 테스트 ✓
| 환경 | 상태 | Action Dim | Obs Dim |
|------|------|------------|---------|
| UnitreeH1 (Humanoid) | ✓ PASSED | 19 | 49 |
| UnitreeG1 (Humanoid) | ✓ PASSED | 23 | 57 |
| Atlas (Humanoid) | ✓ PASSED | 27 | 65 |

---

## 🎯 VRAM 최적화 결과

### RTX 3070 (8GB) 최적 설정

#### 테스트 결과 요약
| N_Envs | VRAM 사용량 | Steps/sec | FPS/env | 효율성 |
|--------|-------------|-----------|---------|--------|
| 16 | 6240 MB | 2,913 | 182.1 | ★★☆☆☆ |
| 32 | 6240 MB | 5,865 | 183.3 | ★★★☆☆ |
| 64 | 6240 MB | 10,324 | 161.3 | ★★★★☆ |
| 128 | 6240 MB | 19,718 | 154.1 | ★★★★☆ |
| 256 | 6240 MB | 34,181 | 133.5 | ★★★★★ |
| **512** | **6240 MB** | **52,353** | **102.3** | **★★★★★** |

### 권장 설정

#### 1. 최대 성능 우선 (개발/연구용)
```python
num_envs = 512
num_minibatches = 64  # 512 / 8
vram_usage = 76.2%    # 6240 MB / 8192 MB
```

#### 2. 안정성 우선 (장시간 학습)
```python
num_envs = 256
num_minibatches = 32  # 256 / 8
vram_usage = 76.2%
```

#### 3. 메모리 여유 필요 (대형 모델/디버깅)
```python
num_envs = 128
num_minibatches = 16  # 128 / 8
vram_usage = 76.2%
```

### Mini-batch 크기 권장사항
- **일반 규칙**: `num_minibatches = num_envs / 4 ~ num_envs / 8`
- **PPO**: mini_batch_size = 16-64
- **GAIL/AMP**: mini_batch_size = 16-64
- **큰 모델 사용 시**: 더 작은 배치 사용

---

## 📚 예제 실행 가이드

### 1. 기본 환경 테스트
```bash
conda activate loco-mujoco
python test_basic_env.py
```

### 2. MJX GPU 성능 테스트
```bash
python test_mjx_gpu.py
```

### 3. VRAM 최적화 테스트
```bash
python test_vram_optimization.py
```

### 4. 다중 환경 테스트
```bash
python test_multiple_envs.py
```

### 5. PPO 학습 예제 (공식)
```bash
cd examples/training_examples/jax_rl
python experiment.py
```
- **예상 학습 시간**: ~20분 (RTX 3080 Ti 기준)
- **RTX 3070 예상**: ~25-30분
- **총 스텝**: 100M steps
- **환경**: UnitreeGo2
- **알고리즘**: PPO

### 6. 데이터셋 시각화 (렌더링 필요)
```bash
cd examples/tutorials
python 00_replay_datasets.py
```

---

## 🔧 추가 설정 (선택사항)

### MyoSkeleton 환경 사용
```bash
conda activate loco-mujoco
loco-mujoco-myomodel-init
```

### 데이터셋 로딩 속도 개선 (캐시 설정)
```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

### AMASS 데이터셋 사용
[loco_mujoco/smpl/README.md](loco_mujoco/smpl) 참조

---

## 💡 학습 설정 최적화 팁

### RTX 3070 (8GB) 권장 설정

#### conf.yaml 수정 예시
```yaml
experiment:
  num_envs: 512          # VRAM 최적화 결과 기반
  num_minibatches: 64    # 512 / 8
  num_steps: 50          # 기본값 유지
  hidden_layers: [512, 256]  # 모델 크기 조정 가능
  lr: 1e-4
  total_timesteps: 10e7
```

#### 메모리 부족 시 조정
```yaml
experiment:
  num_envs: 256          # 환경 수 줄이기
  num_minibatches: 32
  hidden_layers: [256, 128]  # 모델 크기 줄이기
```

### XLA 최적화 플래그
코드에 이미 적용됨:
```python
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
```

---

## 📊 성능 벤치마크 요약

### RTX 3070 성능
- **병렬 환경 512개**: 52,353 steps/sec
- **효율성**: 8.39 steps/sec/MB VRAM
- **VRAM 사용률**: 76.2% (6240 MB / 8192 MB)
- **개별 환경 FPS**: 102.3 FPS

### 예상 학습 시간 (100M steps 기준)
```
100,000,000 steps / 52,353 steps/sec ≈ 1,910 초 ≈ 32분
```
(실제로는 네트워크 업데이트 시간 추가로 더 소요)

---

## 🐛 문제 해결

### VRAM 부족 에러
1. `num_envs` 줄이기: 512 → 256 → 128
2. `hidden_layers` 줄이기: [512, 256] → [256, 128]
3. `num_minibatches` 비례해서 줄이기

### JAX/CUDA 오류
```bash
# CUDA 버전 확인
nvidia-smi

# JAX 재설치
pip install --upgrade "jax[cuda12]"
```

### 데이터셋 다운로드 실패
```bash
# 캐시 삭제 후 재시도
rm -rf ~/.cache/huggingface
python your_script.py
```

---

## 📖 추가 자료

### 공식 문서
- [LocoMuJoCo Documentation](https://loco-mujoco.readthedocs.io/)
- [GitHub Repository](https://github.com/robfiras/loco-mujoco)
- [Discord Community](https://discord.gg/gEqR3xCVdn)

### 예제 위치
- **튜토리얼**: [examples/tutorials/](examples/tutorials/)
- **학습 예제**: [examples/training_examples/](examples/training_examples/)
- **궤적 생성**: [examples/trajectory_generation/](examples/trajectory_generation/)

### 주요 알고리즘
- **PPO**: [examples/training_examples/jax_rl/](examples/training_examples/jax_rl/)
- **GAIL**: [examples/training_examples/jax_gail/](examples/training_examples/jax_gail/)
- **AMP**: [examples/training_examples/jax_amp/](examples/training_examples/jax_amp/)
- **DeepMimic**: [examples/training_examples/jax_rl_mimic/](examples/training_examples/jax_rl_mimic/)

---

## ✨ 다음 단계

1. **기본 예제 실행**: `test_mjx_gpu.py`로 설정 확인
2. **튜토리얼 학습**: `examples/tutorials/` 순서대로 실습
3. **PPO 학습 시작**: `examples/training_examples/jax_rl/experiment.py`
4. **설정 최적화**: 본인 GPU에 맞게 `num_envs`, `num_minibatches` 조정
5. **고급 알고리즘 실험**: GAIL, AMP, DeepMimic 시도

---

**설치 완료!** 🎉

이제 loco-mujoco로 로봇 학습을 시작할 준비가 되었습니다!
