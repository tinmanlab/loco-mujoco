#!/usr/bin/env python3
"""
SkeletonTorque AMP 학습된 policy를 Interactive MuJoCo Viewer로 실행
마우스로 직접 외력을 가할 수 있습니다!

Controls:
- Ctrl + Left Click: Apply external force to body (외력 적용!)
- Left Click + Drag: Rotate camera
- Right Click + Drag: Move camera
- Mouse Wheel: Zoom
- Space: Pause/Resume
- Double Click: Select body
- Backspace: Reset simulation
"""
import os
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax
from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# 체크포인트 로드
path = "/home/tinman/loco-mujoco/custom/training/successful/skeleton_amp_training/outputs/2025-10-31/12-20-59/AMPJax_saved.pkl"

print("="*80)
print("SkeletonTorque AMP Policy - Interactive MuJoCo Viewer")
print("="*80)
print("\n마우스로 직접 외력을 가할 수 있습니다!")
print("\nViewer Controls:")
print("  - Ctrl + Left Click: Apply external force (외력 적용!)")
print("  - Left Click + Drag: Rotate camera")
print("  - Right Click + Drag: Move camera")
print("  - Mouse Wheel: Zoom")
print("  - Space: Pause/Resume")
print("  - Double Click: Select body")
print("  - Backspace: Reset simulation")
print("="*80)

# Load checkpoint
print("\n1. Loading checkpoint...")
agent_conf, agent_state = AMPJax.load_agent(path)
config = agent_conf.config
print(f"   ✓ Checkpoint loaded from: {path}")

# Get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# 중요: 순수 MuJoCo 환경을 만들기 위해 env_name을 수정
print("\n2. Creating MuJoCo (not MJX) environment for interactive viewer...")
OmegaConf.set_struct(config, False)

# MjxSkeletonTorque -> SkeletonTorque (순수 MuJoCo)
original_env_name = config.experiment.env_params.env_name
if "Mjx" in original_env_name:
    config.experiment.env_params.env_name = original_env_name.replace("Mjx", "")
    print(f"   Changed: {original_env_name} -> {config.experiment.env_params.env_name}")

# headless=False for interactive viewer
config.experiment.env_params["headless"] = False

env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

print(f"   ✓ Environment: {config.experiment.env_params.env_name}")
print(f"   - Action dim: {env.info.action_space.shape[0]}")
print(f"   - Obs dim: {env.info.observation_space.shape[0]}")
print(f"   - Interactive: YES (MuJoCo native viewer)")

# Run policy with MuJoCo native viewer
print("\n3. Starting interactive viewer...")
print("   Press Ctrl+C to stop\n")
print("="*80)

try:
    # use_mujoco=True로 play_policy 호출 -> 순수 MuJoCo viewer 실행
    # render=True, record=False로 interactive viewer만 실행
    AMPJax.play_policy_mujoco(env, agent_conf, agent_state,
                              deterministic=False,
                              n_steps=10000,
                              render=True,
                              record=False,
                              train_state_seed=0)
except KeyboardInterrupt:
    print("\n\n⚠ Stopped by user")

print("\n" + "="*80)
print("Done!")
print("="*80)
