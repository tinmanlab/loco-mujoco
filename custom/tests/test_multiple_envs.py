"""
Test multiple loco-mujoco environments
"""
import numpy as np
from loco_mujoco import ImitationFactory

print("=" * 60)
print("Testing Multiple LocoMuJoCo Environments")
print("=" * 60)

# List of environments to test
env_configs = [
    ("UnitreeH1", "walk", "Humanoid UnitreeH1"),
    ("UnitreeG1", "walk", "Humanoid UnitreeG1"),
    ("Atlas", "walk", "Humanoid Atlas"),
    ("UnitreeA1", "trot", "Quadruped UnitreeA1"),
]

results = []

for env_name, task, description in env_configs:
    print(f"\n{'-' * 60}")
    print(f"Testing: {description}")
    print(f"Environment: {env_name}, Task: {task}")
    print(f"{'-' * 60}")

    try:
        # Create environment
        print(f"   Creating environment...")
        env = ImitationFactory.make(
            env_name,
            default_dataset_conf=dict(task=task),
            n_substeps=20
        )

        # Get info
        action_dim = env.info.action_space.shape[0]
        obs_dim = env.info.observation_space.shape[0]
        print(f"   ✓ Environment created")
        print(f"   ✓ Action dim: {action_dim}, Obs dim: {obs_dim}")

        # Reset
        obs = env.reset()
        print(f"   ✓ Environment reset")

        # Take a few steps
        total_reward = 0
        for i in range(10):
            action = np.random.randn(action_dim) * 0.1
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        avg_reward = total_reward / 10
        print(f"   ✓ 10 steps completed, avg reward: {avg_reward:.6f}")

        # Try loading dataset
        try:
            dataset = env.create_dataset()
            print(f"   ✓ Dataset loaded")
            results.append((description, "✓ PASSED", action_dim, obs_dim))
        except Exception as e:
            print(f"   ! Dataset error: {e}")
            results.append((description, "✓ PASSED (no dataset)", action_dim, obs_dim))

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        results.append((description, f"✗ FAILED: {str(e)[:40]}", 0, 0))

# Print summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Environment':<30} {'Status':<20} {'Action':<8} {'Obs':<8}")
print(f"{'-' * 60}")
for desc, status, action_dim, obs_dim in results:
    print(f"{desc:<30} {status:<20} {action_dim:<8} {obs_dim:<8}")

print(f"\n{'=' * 60}")
passed = sum(1 for _, status, _, _ in results if "PASSED" in status)
print(f"✓ {passed}/{len(env_configs)} environments tested successfully!")
print(f"{'=' * 60}")
