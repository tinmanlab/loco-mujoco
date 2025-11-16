"""
MyoSkeleton GPU 활성화 및 JIT 컴파일 확인
"""
import os
import jax
import time
import jax.numpy as jnp
from loco_mujoco import TaskFactory

# GPU 최적화 플래그
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

print("=" * 80)
print("MyoSkeleton GPU & JIT 컴파일 확인")
print("=" * 80)

# 1. JAX 백엔드 확인
print("\n1. JAX 백엔드 확인:")
print(f"   - JAX version: {jax.__version__}")
print(f"   - Default backend: {jax.default_backend()}")
print(f"   - Devices: {jax.devices()}")
print(f"   - Device type: {jax.devices()[0].platform}")

if jax.default_backend() != 'gpu':
    print("   ⚠️  WARNING: GPU가 기본 백엔드가 아닙니다!")
    print("   ⚠️  CPU fallback이 발생할 수 있습니다.")
else:
    print("   ✓ GPU가 기본 백엔드로 설정됨")

# 2. 환경 생성
print("\n2. MjxMyoSkeleton 환경 생성...")
factory = TaskFactory.get_factory_cls('RLFactory')
env = factory.make(
    env_name='MjxMyoSkeleton',
    horizon=1000,
    headless=True,
    goal_type='GoalRandomRootVelocity',
    terminal_state_type='HeightBasedTerminalStateHandler',
    reward_type='LocomotionReward'
)

print(f"   ✓ 환경 생성 완료")
print(f"   - Action dim: {env.info.action_space.shape[0]}")
print(f"   - Obs dim: {env.info.observation_space.shape[0]}")

# 3. JIT 컴파일 및 GPU 확인
print("\n3. JIT 컴파일 및 GPU 사용 확인...")

n_envs = 128
key = jax.random.key(0)
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# JIT + vmap 함수 생성
print("   - JIT 컴파일 중...")
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_action = jax.jit(jax.vmap(env.sample_action_space))

# Reset
state = rng_reset(env_keys)
print("   ✓ JIT 컴파일 완료 (reset)")

# 첫 step으로 JIT 컴파일 트리거
keys = jax.random.split(key, n_envs + 1)
key, action_keys = keys[0], keys[1:]
actions = rng_sample_action(action_keys)

print("   - Step JIT 컴파일 중...")
state = rng_step(state, actions)
jax.block_until_ready(state)
print("   ✓ JIT 컴파일 완료 (step)")

# 4. GPU 메모리 사용 확인
import subprocess
try:
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    vram_used = int(result.strip())
    print(f"\n4. GPU 메모리 사용:")
    print(f"   - VRAM 사용량: {vram_used} MB")
    if vram_used < 1000:
        print("   ⚠️  WARNING: VRAM 사용량이 너무 적습니다. CPU fallback 가능성!")
    else:
        print("   ✓ GPU가 정상적으로 사용되고 있습니다")
except:
    print("\n4. GPU 메모리: 확인 불가")

# 5. 성능 테스트로 GPU 사용 확인
print("\n5. 성능 테스트 (GPU vs CPU 비교):")

# GPU 테스트
n_steps = 100
start = time.time()
for _ in range(n_steps):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    actions = rng_sample_action(action_keys)
    state = rng_step(state, actions)
jax.block_until_ready(state)
elapsed = time.time() - start

steps_per_sec = (n_steps * n_envs) / elapsed
print(f"   - {n_envs} envs, {n_steps} steps")
print(f"   - 총 steps: {n_steps * n_envs:,}")
print(f"   - 시간: {elapsed:.2f}초")
print(f"   - 성능: {steps_per_sec:,.0f} steps/sec")

# GPU 사용 시 기대 성능
expected_min = 5000  # 최소 5000 steps/sec
if steps_per_sec < expected_min:
    print(f"   ⚠️  WARNING: 성능이 {expected_min} steps/sec 미만입니다.")
    print(f"   ⚠️  CPU fallback이 발생했을 가능성이 있습니다.")
else:
    print(f"   ✓ 성능 정상 (GPU 사용 중)")

# 6. 데이터가 GPU에 있는지 확인
print("\n6. 데이터 위치 확인:")
if hasattr(state, 'reward'):
    reward_device = state.reward.devices()
    print(f"   - Reward device: {reward_device}")
    if 'gpu' in str(reward_device).lower() or 'cuda' in str(reward_device).lower():
        print(f"   ✓ 데이터가 GPU에 있음")
    else:
        print(f"   ⚠️  WARNING: 데이터가 CPU에 있을 수 있음")

# 7. XLA 컴파일 캐시 확인
print("\n7. XLA 컴파일 상태:")
print(f"   - XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")

# 최종 결론
print("\n" + "=" * 80)
print("최종 진단:")
print("=" * 80)

issues = []
if jax.default_backend() != 'gpu':
    issues.append("JAX 백엔드가 GPU가 아님")
if steps_per_sec < expected_min:
    issues.append(f"성능이 낮음 ({steps_per_sec:.0f} < {expected_min})")

if issues:
    print("⚠️  발견된 이슈:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n해결 방법:")
    print("   1. JAX CUDA 재설치: pip install --upgrade 'jax[cuda12]'")
    print("   2. 환경 변수 확인: echo $CUDA_VISIBLE_DEVICES")
    print("   3. nvidia-smi로 GPU 사용 확인")
else:
    print("✓ 모든 확인 완료!")
    print("✓ GPU가 정상적으로 활성화되어 있습니다.")
    print("✓ JIT 컴파일이 정상적으로 작동합니다.")
    print("✓ MyoSkeleton 학습 준비 완료!")

print("=" * 80)
