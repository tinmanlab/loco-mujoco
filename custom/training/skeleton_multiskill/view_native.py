#!/usr/bin/env python3
"""
Phase 1 Multi-Skill Model - Native MuJoCo Viewer
Walk + Run 정책을 native MuJoCo viewer로 실행

Controls:
- Ctrl + Left Click: Apply force
- Double Click: Select body
- Right Click: Pan
- Scroll: Zoom
- Space: Pause
"""
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax

CHECKPOINT_PATH = "outputs/2025-11-16/00-59-52/AMPJax_saved.pkl"

print("="*80)
print("Walk Model - Native MuJoCo Viewer")
print("="*80)
print("\nModel: walk (75M timesteps)")
print("\nControls:")
print("  - Ctrl + Left Click: Apply external force")
print("  - Double Click: Select body")
print("  - Right Click: Pan camera")
print("  - Scroll: Zoom")
print("  - Space: Pause/Resume")
print("="*80)

# Load agent
print("\n[1] Loading agent...")
agent_conf, agent_state = AMPJax.load_agent(CHECKPOINT_PATH)
print(f"  ✓ Loaded: {CHECKPOINT_PATH}")

# Create MJX environment (headless for computation)
print("\n[2] Creating MJX environment...")
env_config = agent_conf.config.experiment
factory = TaskFactory.get_factory_cls(env_config.task_factory.name)

mjx_env = factory.make(
    **env_config.env_params,
    **env_config.task_factory.params
)
print(f"  ✓ MJX environment created")

# Get MuJoCo model for viewer
print("\n[3] Setting up native viewer...")
model = mjx_env._model
data = mjx_env._data
print(f"  ✓ MuJoCo model extracted")

# Prepare policy
params = agent_state.train_state.params
run_stats = getattr(agent_state.train_state, 'run_stats', {})
apply_fn = agent_conf.network.apply

def get_action(obs, rng):
    """Get action from policy"""
    variables = {'params': params}
    if run_stats:
        variables['run_stats'] = run_stats
    (pi, _), _ = apply_fn(variables, obs, mutable=['run_stats'])
    return np.array(pi.mean())

print("\n[4] Starting viewer...")
print("  - Episode will auto-reset")
print("  - Close window to exit\n")

# Initialize
rng = jax.random.PRNGKey(0)
obs = mjx_env.reset()
episode = 0
steps = 0
episode_reward = 0.0

# Launch native viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    while viewer.is_running():
        # Get action from policy
        rng, _rng = jax.random.split(rng)
        action = get_action(obs, _rng)

        # Step MJX environment
        obs, reward, absorbing, done, info = mjx_env.step(action)
        episode_reward += reward
        steps += 1

        # Copy MJX state to MuJoCo for rendering
        data.qpos[:] = np.array(mjx_env._data.qpos)
        data.qvel[:] = np.array(mjx_env._data.qvel)

        # Sync viewer
        viewer.sync()

        # Reset on done
        if done or steps >= 1000:
            episode += 1
            print(f"Episode {episode}: Return={episode_reward:.1f}, Length={steps}")
            obs = mjx_env.reset()
            episode_reward = 0.0
            steps = 0

print("\nViewer closed.")
print("="*80)
