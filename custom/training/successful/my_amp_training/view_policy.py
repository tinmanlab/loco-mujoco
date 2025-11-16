#!/usr/bin/env python3
"""
학습된 AMP policy를 MuJoCo viewer로 시각화
"""
import pickle
import jax
import jax.numpy as jnp
from loco_mujoco import TaskFactory
from omegaconf import OmegaConf

# 체크포인트 경로
CHECKPOINT_PATH = "/home/tinman/loco-mujoco/my_amp_training/outputs/2025-10-31/02-01-09/AMPJax_saved.pkl"
CONFIG_PATH = "/home/tinman/loco-mujoco/my_amp_training/conf.yaml"

def load_checkpoint(checkpoint_path):
    """체크포인트 로드"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print("✓ Checkpoint loaded")
    return checkpoint

def create_environment(config):
    """환경 생성"""
    print("\nCreating environment...")

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **config.experiment.env_params,
        **config.experiment.task_factory.params
    )

    print(f"✓ Environment created: {config.experiment.env_params.env_name}")
    print(f"  - Action dim: {env.info.action_space.shape[0]}")
    print(f"  - Obs dim: {env.info.observation_space.shape[0]}")

    return env

def run_policy_viewer(env, checkpoint, n_steps=1000, render=True):
    """
    학습된 policy를 실행하며 viewer로 시각화

    Args:
        env: 환경
        checkpoint: 로드된 체크포인트
        n_steps: 실행할 스텝 수
        render: MuJoCo viewer 사용 여부
    """
    print(f"\nRunning policy for {n_steps} steps...")
    print(f"Render: {render}")

    # 초기화
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)

    env_state = env.mjx_reset(reset_rng)
    obs = env_state.observation

    # Policy 함수
    train_state = checkpoint['agent_state']['train_state']
    params = train_state['params']
    run_stats = train_state.get('run_stats', {})  # RunningMeanStd stats
    apply_fn = checkpoint['agent_conf']['network'].apply

    def get_action(obs, rng):
        """Deterministic policy"""
        variables = {'params': params}
        if run_stats:
            variables['run_stats'] = run_stats
        (pi, _), _ = apply_fn(variables, obs, mutable=['run_stats'])
        return pi.mean()

    print("\nStarting episode...")
    total_reward = 0.0

    for step in range(n_steps):
        # Action 선택
        rng, action_rng = jax.random.split(rng)
        action = get_action(obs, action_rng)

        # Step
        env_state = env.mjx_step(env_state, action)
        obs = env_state.observation
        reward = env_state.reward
        done = env_state.done
        total_reward += float(reward)

        # Render
        if render:
            try:
                env.mjx_render(env_state, record=False)
            except Exception as e:
                print(f"Rendering error at step {step}: {e}")
                print("Continuing without rendering...")
                render = False

        # Progress
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{n_steps}, Total reward: {total_reward:.2f}")

        # Episode end
        if done:
            print(f"\nEpisode ended at step {step + 1}")
            print(f"Total reward: {total_reward:.2f}")

            # Reset
            rng, reset_rng = jax.random.split(rng)
            env_state = env.mjx_reset(reset_rng)
            obs = env_state.observation
            total_reward = 0.0
            print("\nStarting new episode...")

    print(f"\n✓ Completed {n_steps} steps")

    # Viewer 닫기
    if hasattr(env, '_viewer') and env._viewer is not None:
        env._viewer.close()

def main():
    print("="*80)
    print("AMP Policy Viewer")
    print("="*80)

    # Config 로드
    config = OmegaConf.load(CONFIG_PATH)
    print(f"\n✓ Config loaded from: {CONFIG_PATH}")

    # 체크포인트 로드
    checkpoint = load_checkpoint(CHECKPOINT_PATH)

    # 환경 생성 (headless 환경에서는 viewer 불가능하므로 headless=True로 유지)
    config.experiment.env_params.headless = True
    env = create_environment(config)
    print("\n⚠ Running in headless mode - no visual rendering available")

    # Policy 실행 (headless 환경이므로 render=False)
    try:
        run_policy_viewer(env, checkpoint, n_steps=2000, render=False)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(env, '_viewer') and env._viewer is not None:
            env._viewer.close()

    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == "__main__":
    main()
