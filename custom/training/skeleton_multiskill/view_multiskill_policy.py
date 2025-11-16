#!/usr/bin/env python3
"""
Phase 1 Multi-Skill Model Viewer
- Walk + Run 모션 학습된 정책 실행
- Interactive viewer with mouse controls

Controls:
- Ctrl + Left Click: Apply external force
- Left Click + Drag: Rotate camera
- Right Click + Drag: Move camera
- Mouse Wheel: Zoom
- Space: Pause/Resume
- Double Click: Select body
"""
import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax

# Checkpoint path
CHECKPOINT_PATH = "outputs/2025-11-15/22-49-53/AMPJax_saved.pkl"

print("="*80)
print("Phase 1 Multi-Skill Policy Viewer")
print("="*80)
print("\nModel:")
print("  - Motions: walk + run")
print("  - Timesteps: 100M")
print("  - Network: [512, 256, 256]")
print("\nViewer Controls:")
print("  - Ctrl + Left Click: Apply external force")
print("  - Left Click + Drag: Rotate camera")
print("  - Right Click + Drag: Move camera")
print("  - Mouse Wheel: Zoom")
print("  - Space: Pause/Resume")
print("  - Double Click: Select body")
print("="*80)

# 1. Load checkpoint
print("\n[1] Loading agent (this may take a minute)...")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"  ❌ Checkpoint not found: {CHECKPOINT_PATH}")
    exit(1)

# Load agent configuration and state
agent_conf, agent_state = AMPJax.load_agent(CHECKPOINT_PATH)
print(f"  ✓ Agent loaded: {CHECKPOINT_PATH}")

# Get the environment config from agent_conf
env_config = agent_conf.config.experiment

# Create new environment with headless=False for interactive viewing
print("\n[2] Creating interactive environment...")
factory = TaskFactory.get_factory_cls(env_config.task_factory.name)

# Override headless to False for viewer
env_params = dict(env_config.env_params)
env_params['headless'] = False

env = factory.make(
    **env_params,
    **env_config.task_factory.params
)
print(f"  ✓ Interactive environment created")
print(f"  - Viewer enabled")
print(f"  - Motions: walk + run")

# 3. Prepare policy
print("\n[3] Preparing policy...")
params = agent_state['train_state']['params']
run_stats = agent_state['train_state'].get('run_stats', {})
apply_fn = agent_conf['network'].apply

def get_action(obs, rng):
    """Get action from trained policy"""
    variables = {'params': params}
    if run_stats:
        variables['run_stats'] = run_stats
    (pi, _), _ = apply_fn(variables, obs, mutable=['run_stats'])
    return np.array(pi.mean())  # Deterministic mode

print("  ✓ Policy ready")

# 4. Run interactive simulation
print("\n[4] Starting interactive viewer...")
print("  - Running continuous episodes")
print("  - Press Ctrl+C to exit\n")

rng = jax.random.PRNGKey(0)
episode = 0
max_episodes = 20  # Run 20 episodes

try:
    while episode < max_episodes:
        episode += 1
        print(f"\n=== Episode {episode}/{max_episodes} ===")

        obs = env.reset()
        episode_reward = 0.0
        steps = 0

        while steps < 1000:  # horizon
            rng, _rng = jax.random.split(rng)
            action = get_action(obs, _rng)

            obs, reward, absorbing, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            if done:
                break

        print(f"  Return: {episode_reward:.1f} | Length: {steps} steps")

except KeyboardInterrupt:
    print("\n\nViewer stopped by user.")

print("\n" + "="*80)
print("Viewer closed.")
print("="*80)
