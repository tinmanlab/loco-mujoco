#!/usr/bin/env python3
"""
SkeletonTorque AMP - MuJoCo Interactive Viewer
Ctrl+Click으로 마우스로 외력을 직접 가할 수 있습니다!
"""
import os
import jax
import jax.numpy as jnp
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax
from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

path = "/home/tinman/loco-mujoco/custom/training/successful/skeleton_amp_training/outputs/2025-10-31/12-20-59/AMPJax_saved.pkl"

print("="*80)
print("SkeletonTorque AMP - Interactive MuJoCo Viewer")
print("="*80)
print("\nControls:")
print("  - Ctrl + Left Click: 마우스로 외력 적용!")
print("  - Left Click + Drag: 카메라 회전")
print("  - Right Click + Drag: 카메라 이동")
print("  - Mouse Wheel: 줌")
print("  - Space: 일시정지/재개")
print("  - Double Click: body 선택")
print("="*80)

# Load checkpoint
print("\n1. Loading checkpoint...")
agent_conf, agent_state = AMPJax.load_agent(path)
config = agent_conf.config
print(f"   ✓ Loaded: {path}")

# Get factory and create MuJoCo (not MJX) environment
print("\n2. Creating MuJoCo environment...")
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

OmegaConf.set_struct(config, False)
# MjxSkeletonTorque -> SkeletonTorque
if "Mjx" in config.experiment.env_params.env_name:
    config.experiment.env_params.env_name = config.experiment.env_params.env_name.replace("Mjx", "")
config.experiment.env_params["headless"] = False  # Interactive mode!
# visualize_goal을 끄기 (앞 화살표 제거)
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

# Run interactive viewer
print("\n4. Starting interactive viewer...")
print("   Ctrl+Click으로 외력을 가할 수 있습니다!")
print("   Press Ctrl+C to stop\n")
print("="*80)

try:
    obs = env.reset()

    for i in range(10000):
        # Sample action
        rng, _rng = jax.random.split(rng)
        action, train_state = policy(train_state, obs, _rng)
        action = jnp.atleast_2d(action)

        # Step environment
        obs, reward, absorbing, done, info = env.step(action)

        # Render (record=False로 녹화 없이 viewer만!)
        env.render(record=False)

        # Reset if done
        if done:
            obs = env.reset()

    env.stop()

except KeyboardInterrupt:
    print("\n\n⚠ Stopped by user")
    env.stop()

print("\n" + "="*80)
print("Done!")
print("="*80)
