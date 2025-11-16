"""
MyoSkeleton 걷기 학습 스크립트
PPO 알고리즘 사용
"""
import os
import sys
import jax
import jax.numpy as jnp
import wandb
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./", config_name="conf")
def train(config: DictConfig):
    try:
        print("=" * 80)
        print("MyoSkeleton 걷기 학습 시작")
        print("=" * 80)

        # GPU 최적화
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

        # JAX 설정 확인
        print(f"\n1. JAX 설정:")
        print(f"   - Version: {jax.__version__}")
        print(f"   - Backend: {jax.default_backend()}")
        print(f"   - Devices: {jax.devices()}")

        # Hydra output directory
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(f"\n2. 출력 디렉토리: {result_dir}")

        # WandB 초기화
        print(f"\n3. WandB 초기화...")
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(
            project=config.wandb.project,
            config=config_dict,
            name=f"myoskeleton_{config.experiment.num_envs}envs"
        )
        print(f"   ✓ WandB 프로젝트: {config.wandb.project}")

        # 환경 생성
        print(f"\n4. 환경 생성...")
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

        print(f"   ✓ 환경: {config.experiment.env_params.env_name}")
        print(f"   ✓ Action dim: {env.info.action_space.shape[0]}")
        print(f"   ✓ Obs dim: {env.info.observation_space.shape[0]}")
        print(f"   ✓ 병렬 환경 수: {config.experiment.num_envs}")

        # PPO 에이전트 설정
        print(f"\n5. PPO 에이전트 설정...")
        agent_conf = PPOJax.init_agent_conf(env, config)
        print(f"   ✓ 네트워크: {config.experiment.hidden_layers}")
        print(f"   ✓ Learning rate: {config.experiment.lr}")
        print(f"   ✓ Mini-batches: {config.experiment.num_minibatches}")

        # 학습 함수 빌드
        print(f"\n6. 학습 함수 빌드 (JIT 컴파일 중)...")
        train_fn = PPOJax.build_train_fn(env, agent_conf)

        # JIT 컴파일
        if config.experiment.n_seeds > 1:
            train_fn = jax.jit(jax.vmap(train_fn))
        else:
            train_fn = jax.jit(train_fn)

        print(f"   ✓ JIT 컴파일 완료")

        # RNG 키 생성
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))

        # 학습 시작
        print(f"\n7. 학습 시작!")
        print(f"   - 총 timesteps: {int(config.experiment.total_timesteps):,}")
        print(f"   - 환경 수: {config.experiment.num_envs}")
        print(f"   - 스텝 당 환경: {config.experiment.num_steps}")
        expected_iterations = int(config.experiment.total_timesteps) // (config.experiment.num_envs * config.experiment.num_steps)
        print(f"   - 예상 iteration 수: {expected_iterations:,}")
        print(f"\n{'=' * 80}")
        print("학습 중... (WandB에서 진행상황 확인)")
        print(f"{'=' * 80}\n")

        # 학습 실행
        out = train_fn(_rng)

        # 학습 완료
        print(f"\n{'=' * 80}")
        print("학습 완료!")
        print(f"{'=' * 80}")

        # 결과 저장
        if hasattr(out, 'runner_state'):
            import pickle
            save_path = os.path.join(result_dir, "PPOJax_saved.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(out.runner_state, f)
            print(f"\n모델 저장: {save_path}")

        # WandB 종료
        wandb.finish()
        print(f"\n✓ 완료!")

    except Exception as e:
        import traceback
        print(f"\n✗ 오류 발생:")
        print(traceback.format_exc())
        wandb.finish()
        raise e

if __name__ == "__main__":
    train()
