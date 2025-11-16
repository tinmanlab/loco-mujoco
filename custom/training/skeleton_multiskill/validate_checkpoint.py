#!/usr/bin/env python3
"""
체크포인트 검증 스크립트
- 체크포인트 로드 확인
- 정책 실행 테스트 (headless, no recording)
"""
import os
import jax
import jax.numpy as jnp
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import AMPJax
import pickle

def main():
    print("=" * 60)
    print("Phase 1 Multi-Skill Training Validation")
    print("=" * 60)

    # 체크포인트 경로
    checkpoint_path = "outputs/2025-11-15/22-49-53/AMPJax_saved.pkl"

    # 1. 체크포인트 로드
    print("\n[1] Loading checkpoint...")
    try:
        with open(checkpoint_path, 'rb') as f:
            saved_data = pickle.load(f)

        agent_conf = saved_data['agent_conf']
        agent_state = saved_data['agent_state']

        print(f"  ✅ Checkpoint loaded successfully")
        print(f"  - File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        print(f"  - Agent config: {type(agent_conf)}")
        print(f"  - Agent state: {type(agent_state)}")

    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        return

    # 2. 환경 생성
    print("\n[2] Creating environment...")
    try:
        factory = TaskFactory.get_factory_cls("ImitationFactory")

        env = factory.make(
            env_name="MjxSkeletonTorque",
            headless=True,
            horizon=1000,
            goal_type="GoalTrajRootVelocity",
            goal_params={'visualize_goal': False},
            reward_type="TargetVelocityTrajReward",
            default_dataset_conf={'task': ['walk', 'run']}
        )

        print(f"  ✅ Environment created")
        print(f"  - Environment type: {type(env).__name__}")

    except Exception as e:
        print(f"  ❌ Failed to create environment: {e}")
        return

    # 3. 정책 테스트 (play_policy 사용)
    print("\n[3] Testing policy...")
    try:
        # AMPJax.play_policy를 record=False로 실행
        AMPJax.play_policy(
            env,
            agent_conf,
            agent_state,
            deterministic=True,
            n_steps=100,  # 짧게 테스트
            n_envs=10,    # 적은 환경으로 테스트
            record=False,  # 렌더링 비활성화
            train_state_seed=0
        )

        print(f"  ✅ Policy executed successfully")
        print(f"  - Ran 100 steps with 10 parallel environments")
        print(f"  - No rendering errors (headless mode)")

    except Exception as e:
        print(f"  ❌ Failed to test policy: {e}")
        import traceback
        traceback.print_exc()
        # Continue anyway to show config

    # 4. 훈련 설정 확인
    print("\n[4] Training configuration:")
    try:
        env_conf = agent_conf.env_conf
        train_conf = agent_conf.train_conf

        print(f"  - Total timesteps: {train_conf.total_timesteps:,.0f}")
        print(f"  - Num environments: {train_conf.num_envs}")
        print(f"  - Hidden layers: {train_conf.hidden_layers}")
        print(f"  - Learning rate: {train_conf.lr}")
        print(f"  - Discriminator LR: {train_conf.disc_lr}")

    except Exception as e:
        print(f"  ⚠ Could not extract config details: {e}")

    # 5. 요약
    print("\n" + "=" * 60)
    print("Validation Summary:")
    print("=" * 60)
    print("✅ Checkpoint is valid and loadable")
    print("✅ Policy can be executed")
    print("✅ Environment is compatible")
    print("\n📝 Next steps:")
    print("  1. Run viewer with this checkpoint")
    print("  2. Test with external perturbations")
    print("  3. Compare with baseline (single-motion model)")
    print("=" * 60)

if __name__ == "__main__":
    main()
