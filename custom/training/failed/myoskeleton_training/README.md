# MyoSkeleton 걷기 학습

## 설명
MyoSkeleton 환경에서 근골격계 모델이 걷는 것을 학습합니다.
PPO 알고리즘과 MJX (MuJoCo JAX) 병렬 환경을 사용합니다.

## 환경 정보
- **모델**: MyoSkeleton (Musculoskeletal humanoid)
- **Action dim**: 151 (근육 제어)
- **Observation dim**: 316
- **Horizon**: 1000 steps

## 학습 설정 (RTX 3070 최적화)
- **병렬 환경**: 2,048개
- **Mini-batches**: 32 (batch size 64)
- **Network**: [512, 512, 256]
- **총 timesteps**: 100M
- **예상 학습 시간**: ~25-30분

## 사용법

### 1. 환경 활성화
```bash
conda activate loco-mujoco
```

### 2. 학습 시작
```bash
cd /home/tinman/loco-mujoco/myoskeleton_training
python train.py
```

### 3. WandB에서 모니터링
학습 시작 후 터미널에 출력되는 WandB 링크를 클릭하여 실시간 모니터링

### 4. 학습 중단
```
Ctrl + C
```

## 설정 변경

`conf.yaml` 파일을 수정하여 학습 설정 변경 가능:

```yaml
experiment:
  num_envs: 2048          # 환경 수 (메모리에 따라 조정)
  num_minibatches: 32     # 배치 크기
  hidden_layers: [512, 512, 256]  # 네트워크 크기
  lr: 3e-4                # 학습률
  total_timesteps: 100e6  # 총 학습 스텝
```

## 출력
- **모델 파일**: `outputs/<날짜>/<시간>/PPOJax_saved.pkl`
- **로그**: WandB 대시보드

## 팁

### 메모리 부족 시
```yaml
num_envs: 1024  # 환경 수 줄이기
hidden_layers: [256, 256]  # 네트워크 줄이기
```

### 더 빠른 학습
```yaml
num_envs: 4096  # 환경 수 늘리기 (VRAM 여유 있으면)
total_timesteps: 50e6  # 스텝 줄이기
```

### WandB 없이 학습
```python
# train.py에서 wandb 관련 코드 주석 처리
# wandb.login()
# wandb.init(...)
# wandb.finish()
```

## GPU 확인
```bash
watch -n 1 nvidia-smi
```
