#!/usr/bin/env python3
"""
Simple headless policy test - Phase 1 Multi-Skill Model
"""
import pickle
import jax
import numpy as np
from loco_mujoco import TaskFactory

# Checkpoint path
CHECKPOINT_PATH = "outputs/2025-11-15/22-49-53/AMPJax_saved.pkl"

print("="*60)
print("Phase 1 Multi-Skill Model - Quick Test")
print("="*60)

# 1. Load checkpoint
print("\n[1] Loading checkpoint...")
with open(CHECKPOINT_PATH, 'rb') as f:
    checkpoint = pickle.load(f)
print(f"  ✓ Loaded: {CHECKPOINT_PATH}")

# 2. Create environment
print("\n[2] Creating environment...")
factory = TaskFactory.get_factory_cls('ImitationFactory')
env = factory.make(
    env_name='MjxSkeletonTorque',
    default_dataset_conf={'task': ['walk', 'run']},  # Multi-skill!
    headless=True,
    horizon=1000
)
print(f"  ✓ Environment: MjxSkeletonTorque")
print(f"  - Tasks: walk, run (multi-skill)")

# 3. Prepare policy
print("\n[3] Preparing policy...")
params = checkpoint['agent_state']['train_state']['params']
run_stats = checkpoint['agent_state']['train_state'].get('run_stats', {})
apply_fn = checkpoint['agent_conf']['network'].apply

def get_action(obs, rng):
    """Get action from trained policy"""
    variables = {'params': params}
    if run_stats:
        variables['run_stats'] = run_stats
    (pi, _), _ = apply_fn(variables, obs, mutable=['run_stats'])
    return np.array(pi.mean())

print("  ✓ Policy ready")

# 4. Run test episodes
print("\n[4] Running test episodes...")
rng = jax.random.PRNGKey(0)
n_episodes = 3
episode_returns = []
episode_lengths = []

for episode in range(n_episodes):
    obs = env.reset()
    episode_reward = 0.0
    steps = 0

    for step in range(1000):  # horizon = 1000
        rng, _rng = jax.random.split(rng)
        action = get_action(obs, _rng)

        obs, reward, absorbing, done, info = env.step(action)
        episode_reward += reward
        steps += 1

        if done:
            break

    episode_returns.append(float(episode_reward))
    episode_lengths.append(steps)

    print(f"  Episode {episode+1}: Return={episode_reward:.1f}, Length={steps}")

# 5. Summary
print("\n" + "="*60)
print("Test Summary:")
print("="*60)
print(f"Episodes: {n_episodes}")
print(f"Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
print(f"\n✅ Policy is functional!")
print("="*60)
