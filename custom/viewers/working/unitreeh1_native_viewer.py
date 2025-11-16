#!/usr/bin/env python3
"""
UnitreeH1 AMP - Native MuJoCo Interactive Viewer
순수 MuJoCo viewer 사용 - 마우스로 외력을 직접 가할 수 있습니다!
"""
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax
from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

path = "/home/tinman/loco-mujoco/custom/training/successful/my_amp_training/outputs/2025-10-31/02-01-09/AMPJax_saved.pkl"

print("="*80)
print("UnitreeH1 AMP - Native MuJoCo Interactive Viewer")
print("="*80)
print("\n이 viewer는 순수 MuJoCo native viewer를 사용합니다!")
print("\nControls:")
print("  - Ctrl + Right Click + Drag: 마우스로 외력 적용!")
print("  - Left Click + Drag: 카메라 회전")
print("  - Right Click + Drag: 카메라 이동 (패닝)")
print("  - Scroll: 카메라 줌")
print("  - Space: 일시정지/재개")
print("  - Backspace: 시뮬레이션 리셋")
print("  - Double Click: body 선택")
print("="*80)

# Load checkpoint
print("\n1. Loading UnitreeH1 checkpoint...")
agent_conf, agent_state = AMPJax.load_agent(path)
config = agent_conf.config
print(f"   ✓ Loaded: {path}")

# Get factory and create MuJoCo environment
print("\n2. Creating MuJoCo environment...")
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

OmegaConf.set_struct(config, False)
# MjxUnitreeH1 -> UnitreeH1
if "Mjx" in config.experiment.env_params.env_name:
    config.experiment.env_params.env_name = config.experiment.env_params.env_name.replace("Mjx", "")
config.experiment.env_params["headless"] = True  # headless로 환경만 생성
if "goal_params" in config.experiment.env_params:
    config.experiment.env_params.goal_params["visualize_goal"] = False

env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
print(f"   ✓ Environment: {config.experiment.env_params.env_name}")
print(f"   - Action dim: {env.info.action_space.shape[0]}")
print(f"   - Obs dim: {env.info.observation_space.shape[0]}")

# Prepare policy
print("\n3. Preparing policy...")
train_state = agent_state.train_state
if config.experiment.n_seeds > 1:
    train_state = jax.tree.map(lambda x: x[0], train_state)

# Deterministic policy
train_state.params["log_std"] = jnp.ones_like(train_state.params["log_std"]) * -jnp.inf

def sample_action(ts, obs, rng):
    y, updates = agent_conf.network.apply(
        {'params': ts.params, 'run_stats': ts.run_stats},
        obs, mutable=["run_stats"]
    )
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=rng)
    return a, ts

policy = jax.jit(sample_action)
rng = jax.random.key(0)
print("   ✓ Policy ready")

# Get MuJoCo model and data from loco-mujoco environment
print("\n4. Extracting MuJoCo model and data...")
model = env._model
data = env._data
print(f"   ✓ MuJoCo model extracted")
print(f"   - nq (positions): {model.nq}")
print(f"   - nv (velocities): {model.nv}")
print(f"   - nu (actuators): {model.nu}")

# Reset environment to get initial observation
obs = env.reset()

print("\n5. Launching Native MuJoCo Viewer...")
print("   Ctrl+Right Click으로 외력을 가할 수 있습니다!")
print("   Close the viewer window to exit\n")
print("="*80)

# Launch native MuJoCo viewer with passive mode
# This allows us to control the simulation while having interactive features
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Set camera to better view the robot
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    step_count = 0
    max_steps = 10000

    while viewer.is_running() and step_count < max_steps:
        step_start = time.time()

        # Get action from policy
        rng, _rng = jax.random.split(rng)
        action, train_state = policy(train_state, obs, _rng)
        action = np.array(action)  # Convert to numpy

        # Step environment (this updates env's internal data)
        obs, reward, absorbing, done, info = env.step(action)

        # Copy the updated state to our MuJoCo data
        # (env.step updates env._data, we need to sync with our data object)
        data.qpos[:] = env._data.qpos
        data.qvel[:] = env._data.qvel

        # Step the MuJoCo simulation forward (for physics)
        # Note: env.step already did this, so we just sync

        # Sync the viewer (this updates the visualization)
        viewer.sync()

        # Reset if done
        if done:
            obs = env.reset()
            data.qpos[:] = env._data.qpos
            data.qvel[:] = env._data.qvel
            step_count = 0
            print("  Episode ended, resetting...")

        step_count += 1

        # Maintain reasonable framerate (30 FPS)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

env.stop()
print("\n" + "="*80)
print("Done!")
print("="*80)
