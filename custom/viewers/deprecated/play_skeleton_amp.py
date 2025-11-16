#!/usr/bin/env python3
"""
학습된 SkeletonTorque AMP policy를 Interactive Viewer로 실행
마우스로 외력을 줄 수 있습니다!

Controls:
- Ctrl + Left Click: Apply external force (외력 적용!)
- Left Click + Drag: Rotate camera
- Right Click + Drag: Move camera
- Mouse Wheel: Zoom
- Space: Pause/Resume
- Double Click: Select body
"""
import pickle
import jax
import numpy as np
from loco_mujoco import TaskFactory

# 체크포인트 경로
CHECKPOINT_PATH = "/home/tinman/loco-mujoco/custom/training/successful/skeleton_amp_training/outputs/2025-10-31/12-20-59/AMPJax_saved.pkl"

print("="*80)
print("SkeletonTorque AMP Policy - Interactive Viewer")
print("="*80)
print("\nViewer Controls:")
print("  - Ctrl + Left Click: Apply external force (외력 적용!)")
print("  - Left Click + Drag: Rotate camera")
print("  - Right Click + Drag: Move camera  ")
print("  - Mouse Wheel: Zoom")
print("  - Space: Pause/Resume")
print("  - Double Click: Select body")
print("="*80)

# 1. 체크포인트 로드
print("\n1. Loading checkpoint...")
with open(CHECKPOINT_PATH, 'rb') as f:
    checkpoint = pickle.load(f)
print(f"   ✓ Checkpoint loaded from: {CHECKPOINT_PATH}")

# 2. 환경 생성 (headless=False for interactive viewer!)
print("\n2. Creating environment...")
factory = TaskFactory.get_factory_cls('ImitationFactory')
env = factory.make(
    env_name='SkeletonTorque',
    default_dataset_conf={'task': 'walk'},
    headless=False,  # Interactive mode!
    horizon=1000
)
print(f"   ✓ Environment created: SkeletonTorque")
print(f"   - Action dim: {env.info.action_space.shape[0]}")
print(f"   - Obs dim: {env.info.observation_space.shape[0]}")

# 3. Policy 준비
print("\n3. Preparing policy...")
params = checkpoint['agent_state']['train_state']['params']
run_stats = checkpoint['agent_state']['train_state'].get('run_stats', {})
apply_fn = checkpoint['agent_conf']['network'].apply

def get_action(obs, rng):
    """학습된 policy로 action 선택"""
    variables = {'params': params}
    if run_stats:
        variables['run_stats'] = run_stats
    (pi, _), _ = apply_fn(variables, obs, mutable=['run_stats'])
    return np.array(pi.mean())

print("   ✓ Policy ready")

# 4. Interactive simulation
print("\n4. Starting interactive simulation...")
print("   Press Ctrl+C to stop\n")

rng = jax.random.PRNGKey(0)
total_reward = 0.0
episode_count = 0
max_episodes = 10  # 10 episodes

try:
    for episode in range(max_episodes):
        print(f"Episode {episode + 1}/{max_episodes}")

        # Reset
        obs = env.reset()
        episode_reward = 0.0

        # Episode loop
        for step in range(1000):
            # Get action from policy
            rng, action_rng = jax.random.split(rng)
            action = get_action(obs, action_rng)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # Render with interactive viewer
            env.render()

            if done:
                break

        total_reward += episode_reward
        print(f"  Episode reward: {episode_reward:.2f}")
        print()

    print(f"\n✓ Completed {max_episodes} episodes")
    print(f"  Average reward: {total_reward/max_episodes:.2f}")

except KeyboardInterrupt:
    print("\n\n⚠ Stopped by user")

print("\n" + "="*80)
print("Done!")
print("="*80)
