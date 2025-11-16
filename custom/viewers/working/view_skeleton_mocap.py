#!/usr/bin/env python3
"""
SkeletonTorque mocap 데이터를 viewer로 재생
"""
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

print("="*80)
print("SkeletonTorque mocap 데이터 재생")
print("="*80)

# SkeletonTorque 환경 생성 (walk 데이터)
env = ImitationFactory.make(
    "SkeletonTorque",
    default_dataset_conf=DefaultDatasetConf(["walk"]),
    n_substeps=20
)

print("\n✓ SkeletonTorque 환경 생성 완료")
print(f"  - Action dim: {env.info.action_space.shape[0]}")
print(f"  - Obs dim: {env.info.observation_space.shape[0]}")
print("\n재생 시작...\n")

# mocap trajectory 재생
env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)

print("\n✓ 재생 완료!")
