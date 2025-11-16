#!/usr/bin/env python3
"""
간단한 Perturbation Wrapper for MJX environments
- 랜덤 외력 적용
- 최소한의 코드
- Phase 2 빠른 시작용
"""
import jax
import jax.numpy as jnp


def create_perturbation_env(base_env, force_range=50.0, force_prob=0.1):
    """
    기존 MJX 환경에 perturbation 추가

    Args:
        base_env: LocoMuJoCo MJX environment
        force_range: 최대 힘 크기 (N), default: 50.0
        force_prob: 매 step 적용 확률 (0-1), default: 0.1

    Returns:
        Wrapped environment with perturbation

    Example:
        >>> env = factory.make("MjxSkeletonTorque", ...)
        >>> env = create_perturbation_env(env, force_range=50.0)
    """

    class PerturbedEnv:
        """Wrapper that applies random external forces"""

        def __init__(self, env):
            self.env = env
            self.force_range = force_range
            self.force_prob = force_prob

            # Bodies to apply force to
            # For SkeletonTorque: pelvis (0), torso (1)
            self.perturb_bodies = [0, 1]

            print(f"✓ Perturbation wrapper initialized:")
            print(f"  - Force range: {force_range} N")
            print(f"  - Probability: {force_prob*100}%")
            print(f"  - Target bodies: {self.perturb_bodies}")

        def reset(self, rng):
            """Reset environment"""
            return self.env.reset(rng)

        def step(self, env_state, action, rng=None):
            """
            Step with random perturbations

            Note: Handles both old and new API
            - Old: step(env_state, action)
            - New: step(env_state, action, rng)
            """

            # Call base environment step
            if rng is None:
                # Old API (env.step returns rng in env_state)
                result = self.env.step(env_state, action)
            else:
                # New API
                result = self.env.step(env_state, action, rng)

            # Unpack results
            obs, reward, done, info, env_state = result

            # Apply perturbation if rng available
            if rng is not None or hasattr(env_state, 'rng'):
                if rng is None:
                    rng = env_state.rng

                # Split RNG
                rng, force_rng, body_rng = jax.random.split(rng, 3)

                # Decide whether to apply force
                should_apply = jax.random.bernoulli(
                    force_rng, self.force_prob
                )

                # Generate random force
                force_dir = jax.random.normal(force_rng, shape=(3,))
                force_dir = force_dir / (jnp.linalg.norm(force_dir) + 1e-8)
                force = force_dir * self.force_range

                # Random body selection
                body_id = jax.random.choice(
                    body_rng,
                    jnp.array(self.perturb_bodies, dtype=jnp.int32)
                )

                # Apply force to xfrc_applied
                xfrc = env_state.data.xfrc_applied
                new_force = jnp.where(should_apply, force, jnp.zeros(3))
                xfrc = xfrc.at[body_id, :3].set(new_force)

                # Update env_state
                env_state = env_state.replace(
                    data=env_state.data.replace(xfrc_applied=xfrc)
                )

            return obs, reward, done, info, env_state

        def __getattr__(self, name):
            """Delegate other attributes to base env"""
            return getattr(self.env, name)

    return PerturbedEnv(base_env)


# Alternative: Class-based wrapper
class SimplePerturbationWrapper:
    """
    Class-based perturbation wrapper
    (Alternative to functional wrapper above)
    """

    def __init__(self, env, config):
        """
        Args:
            env: Base environment
            config: Dict with keys:
                - force_range: float (default 50.0)
                - force_prob: float (default 0.1)
                - bodies: list of int (default [0, 1])
        """
        self.env = env
        self.force_range = config.get('force_range', 50.0)
        self.force_prob = config.get('force_prob', 0.1)
        self.perturb_bodies = config.get('bodies', [0, 1])

        print(f"✓ SimplePerturbationWrapper initialized")
        print(f"  Force: {self.force_range}N @ {self.force_prob*100}%")

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, env_state, action, rng=None):
        # Same implementation as above
        result = self.env.step(env_state, action, rng) if rng else self.env.step(env_state, action)
        obs, reward, done, info, env_state = result

        if rng is not None or hasattr(env_state, 'rng'):
            if rng is None:
                rng = env_state.rng

            rng, force_rng, body_rng = jax.random.split(rng, 3)
            should_apply = jax.random.bernoulli(force_rng, self.force_prob)

            force_dir = jax.random.normal(force_rng, shape=(3,))
            force_dir = force_dir / (jnp.linalg.norm(force_dir) + 1e-8)
            force = force_dir * self.force_range

            body_id = jax.random.choice(
                body_rng, jnp.array(self.perturb_bodies, dtype=jnp.int32)
            )

            xfrc = env_state.data.xfrc_applied
            new_force = jnp.where(should_apply, force, jnp.zeros(3))
            xfrc = xfrc.at[body_id, :3].set(new_force)

            env_state = env_state.replace(
                data=env_state.data.replace(xfrc_applied=xfrc)
            )

        return obs, reward, done, info, env_state

    def __getattr__(self, name):
        return getattr(self.env, name)
