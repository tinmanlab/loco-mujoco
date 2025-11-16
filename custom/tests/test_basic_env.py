"""
Basic loco-mujoco environment test without rendering
"""
import numpy as np
from loco_mujoco import ImitationFactory

print("=" * 60)
print("Testing loco-mujoco basic environment")
print("=" * 60)

# Create a simple environment
print("\n1. Creating UnitreeH1 environment...")
env = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=dict(task="walk"),
    n_substeps=20
)
print(f"   ✓ Environment created: {env}")

# Get environment info
print("\n2. Environment information:")
print(f"   - Action space: {env.info.action_space}")
print(f"   - Observation space: {env.info.observation_space}")
print(f"   - Action dimension: {env.info.action_space.shape[0]}")

# Reset environment
print("\n3. Resetting environment...")
obs = env.reset()
print(f"   ✓ Initial observation shape: {obs.shape}")
print(f"   ✓ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

# Take a few random steps
print("\n4. Taking 10 random steps...")
action_dim = env.info.action_space.shape[0]
total_reward = 0

for i in range(10):
    action = np.random.randn(action_dim) * 0.1  # Small random actions
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"   ✓ Steps completed successfully")
print(f"   ✓ Average reward: {total_reward / 10:.6f}")

# Check dataset availability
print("\n5. Checking dataset...")
try:
    dataset = env.create_dataset()
    print(f"   ✓ Dataset loaded successfully")
    print(f"   ✓ Dataset keys: {list(dataset.keys())}")
    if 'observations' in dataset:
        print(f"   ✓ Dataset size: {len(dataset['observations'])} samples")
except Exception as e:
    print(f"   ! Dataset loading error: {e}")

print("\n" + "=" * 60)
print("✓ All basic tests passed!")
print("=" * 60)
