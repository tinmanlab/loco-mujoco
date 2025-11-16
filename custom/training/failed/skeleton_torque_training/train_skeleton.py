#!/usr/bin/env python3
"""
SkeletonTorque 걷기 학습 (PPO)
Torque-actuated skeleton (27 actions, 68 obs)
"""
import os
import jax
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
import time
from datetime import datetime

# XLA 최적화
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

def train():
    print("="*80)
    print("SkeletonTorque 걷기 학습 시작")
    print("="*80)

    # 1. JAX 설정 확인
    print("\n1. JAX 설정:")
    print(f"   - Version: {jax.__version__}")
    print(f"   - Backend: {jax.default_backend()}")
    print(f"   - Devices: {jax.devices()}")

    # 2. 출력 디렉토리
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = f"/home/tinman/loco-mujoco/skeleton_torque_training/outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n2. 출력 디렉토리: {output_dir}")

    # 3. 환경 생성
    print("\n3. 환경 생성...")
    factory = TaskFactory.get_factory_cls('RLFactory')
    env = factory.make(
        env_name='MjxSkeletonTorque',
        horizon=1000,
        headless=True
    )
    print(f"   ✓ 환경: MjxSkeletonTorque")
    print(f"   ✓ Action dim: {env.info.action_space.shape[0]}")
    print(f"   ✓ Obs dim: {env.info.observation_space.shape[0]}")

    # 4. PPO 설정 (RTX 3070 최적화)
    print("\n4. PPO 에이전트 설정...")
    agent_conf = PPOJax.create_agent_config(
        env_name='MjxSkeletonTorque',
        hidden_layers=[512, 256, 256],
        lr=3e-4,
        num_envs=4096,  # RTX 3070 최적
        num_steps=50,
        total_timesteps=50_000_000,  # 50M timesteps
        num_minibatches=32,
        update_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_env=True,
        anneal_lr=True,
        result_dir=output_dir,
    )
    print(f"   ✓ 네트워크: {agent_conf.config.hidden_layers}")
    print(f"   ✓ Learning rate: {agent_conf.config.lr}")
    print(f"   ✓ 병렬 환경 수: {agent_conf.config.num_envs}")
    print(f"   ✓ Total timesteps: {agent_conf.config.total_timesteps:,}")

    # 5. 학습 시작
    print("\n5. 학습 함수 빌드 (JIT 컴파일 중)...")
    print("   (이 과정은 시간이 걸릴 수 있습니다...)")

    start_time = time.time()
    _rng = jax.random.PRNGKey(0)
    train_fn = PPOJax.get_train_fn(env, agent_conf)

    print("   ✓ JIT 컴파일 설정 완료")
    print("\n6. 학습 시작!")
    expected_updates = int(agent_conf.config.total_timesteps /
                          (agent_conf.config.num_envs * agent_conf.config.num_steps))
    print(f"   - 총 timesteps: {agent_conf.config.total_timesteps:,}")
    print(f"   - 환경 수: {agent_conf.config.num_envs}")
    print(f"   - 스텝 당 환경: {agent_conf.config.num_steps}")
    print(f"   - 예상 update 수: {expected_updates}")

    print("\n" + "="*80)
    print("학습 중...")
    print("="*80)
    print()

    try:
        out = train_fn(_rng)

        training_time = time.time() - start_time
        print("\n" + "="*80)
        print("✓ 학습 완료!")
        print("="*80)
        print(f"\n학습 시간: {training_time/60:.1f}분")
        print(f"FPS: {agent_conf.config.total_timesteps/training_time:.0f}")

        # 모델 저장
        agent_state = out.runner_state
        save_path = os.path.join(output_dir, "PPOJax_skeleton_torque.pkl")
        PPOJax.save_agent(save_path, agent_conf, agent_state)
        print(f"\n✓ 모델 저장: {save_path}")

        # 최종 메트릭
        if hasattr(out, 'metrics'):
            print("\n최종 성능:")
            final_return = float(out.metrics['returned_episode_returns'].mean(axis=1)[-1])
            final_length = float(out.metrics['returned_episode_lengths'].mean(axis=1)[-1])
            print(f"   - Mean Episode Return: {final_return:.2f}")
            print(f"   - Mean Episode Length: {final_length:.2f}")

        return out

    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단됨")
        return None
    except Exception as e:
        print(f"\n\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train()
