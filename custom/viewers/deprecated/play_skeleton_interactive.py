#!/usr/bin/env python3
"""
학습된 SkeletonTorque AMP policy를 Interactive MuJoCo Viewer로 실행
마우스로 외력을 줄 수 있습니다!

Controls:
- Ctrl + Left Click: Apply external force (외력 적용!)
- Left Click + Drag: Rotate camera
- Right Click + Drag: Move camera
- Mouse Wheel: Zoom
- Space: Pause/Resume
- Double Click: Select body
"""
import os

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# 체크포인트 경로
path = "/home/tinman/loco-mujoco/custom/training/successful/skeleton_amp_training/outputs/2025-10-31/12-20-59/AMPJax_saved.pkl"

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

# Load agent from checkpoint
print("\nLoading checkpoint...")
agent_conf, agent_state = AMPJax.load_agent(path)
config = agent_conf.config
print(f"✓ Checkpoint loaded from: {path}")

# Get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# Create env with INTERACTIVE MODE (headless=False)
print("\nCreating interactive environment...")
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False  # INTERACTIVE MODE!
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

print(f"✓ Environment created: {config.experiment.env_params.env_name}")
print(f"  - Action dim: {env.info.action_space.shape[0]}")
print(f"  - Obs dim: {env.info.observation_space.shape[0]}")

# Run with MuJoCo viewer (not MJX)
print("\nStarting interactive simulation with MuJoCo viewer...")
print("Press Ctrl+C to stop\n")

try:
    # Use play_policy_mujoco for interactive viewer
    AMPJax.play_policy_mujoco(env, agent_conf, agent_state,
                              deterministic=False,
                              n_steps=10000,
                              record=False,
                              train_state_seed=0)
except KeyboardInterrupt:
    print("\n\n⚠ Stopped by user")

print("\n" + "="*80)
print("Done!")
print("="*80)
