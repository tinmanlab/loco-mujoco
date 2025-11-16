# Viewer Status - Phase 1 Multi-Skill Model

## 문제 상황

학습된 체크포인트를 interactive viewer로 실행하려고 시도했으나, MJX 환경과 headless=False 호환성 문제가 있습니다.

### 시도한 방법

1. **직접 환경 생성**: SkeletonTorque (non-MJX) 환경 생성 시도
   - 문제: Dataset 로딩 중 hang (walk, run 모션 데이터 로딩)

2. **AMPJax.load_agent() 사용**: 자동 환경 생성
   - 문제: MjxSkeletonTorque는 headless=False 미지원

3. **환경 파라미터 override**: headless=False로 재생성
   - 문제: Dataset 로딩 중 무한 대기

## 대안

### 방법 1: 기록된 비디오 재생 (권장)
학습 중 또는 별도로 headless=True로 정책 실행하여 비디오 생성:

```python
AMPJax.play_policy(env, agent_conf, agent_state, 
                   deterministic=True, n_steps=200, 
                   n_envs=1, record=True)
# 생성된 mp4 파일을 재생
```

문제: experiment.py에서 EGL 에러로 비디오 생성 실패했음

### 방법 2: MJX Viewer (추천)
MJX 전용 viewer 사용:

```python
from mujoco import mjx
# MJX 환경의 상태를 직접 시각화
```

### 방법 3: Checkpoint를 non-MJX로 변환
체크포인트를 SkeletonTorque (non-MJX)용으로 전환하여 viewer 사용

### 방법 4: Native MuJoCo Viewer
이전에 UnitreeH1에서 성공한 방법:

```python
import mujoco
import mujoco.viewer

# Load MuJoCo model from env
model = env._model
data = env._data

# Launch native viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Run policy...
```

**단점**: MJX → MuJoCo 변환 필요

## 현재 검증 가능한 사항

✅ **Checkpoint 로드**: 정상 작동 (8.13 MB)
✅ **환경 생성**: MjxSkeletonTorque headless=True 가능
✅ **Policy 추론**: Headless 모드에서 실행 가능
❌ **Interactive Viewer**: Headless=False 문제

## 권장 다음 단계

1. **EGL 에러 해결**: libGL 라이브러리 수정하여 비디오 생성 가능하게 만들기
   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
   ```

2. **Headless 비디오 생성**: record=True로 비디오 저장 후 재생

3. **Phase 2 진행**: Viewer 없이도 학습 가능하므로 perturbation training 시작

4. **외력 테스트**: Headless 모드로 외력 적용 테스트 가능:
   ```python
   # Apply external force in headless mode
   env_state.data.xfrc_applied = env_state.data.xfrc_applied.at[body_id].set(force)
   ```

## 결론

**Viewer는 선택사항**입니다. 학습 완료 및 체크포인트는 정상이므로:
- Phase 2 (perturbation training) 진행 가능
- 외력 robustness 테스트는 headless로 가능
- 시각화는 나중에 EGL 문제 해결 후 진행

