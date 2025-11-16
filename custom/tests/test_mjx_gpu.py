"""
Test MJX (MuJoCo JAX) environment with GPU acceleration
"""
import os
import jax
import time
import jax.numpy as jnp
from loco_mujoco import ImitationFactory

# Optimize GPU performance
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

print("=" * 60)
print("Testing MJX (MuJoCo JAX) with GPU Acceleration")
print("=" * 60)

# Check JAX devices
print(f"\n1. JAX Configuration:")
print(f"   - JAX version: {jax.__version__}")
print(f"   - Available devices: {jax.devices()}")
print(f"   - Default backend: {jax.default_backend()}")

# Create MJX environment
print(f"\n2. Creating MJX UnitreeG1 environment...")
env = ImitationFactory.make("MjxUnitreeG1", default_dataset_conf=dict(task="walk"))
print(f"   ✓ Environment created")

# Setup parallel environments
n_envs = 64  # Moderate number for GPU testing
print(f"\n3. Setting up {n_envs} parallel environments...")
key = jax.random.key(0)
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# JIT compile functions
print(f"\n4. JIT compiling functions...")
rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))
print(f"   ✓ Functions compiled")

# Reset all environments
print(f"\n5. Resetting all environments...")
state = rng_reset(env_keys)
print(f"   ✓ All environments reset")

# Warmup (JIT compilation happens here)
print(f"\n6. Warmup (JIT compilation)...")
for _ in range(10):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)
print(f"   ✓ Warmup complete")

# Performance test
print(f"\n7. Performance test (1000 steps)...")
n_steps = 1000
start_time = time.time()

for i in range(n_steps):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)

# Wait for GPU to finish
jax.block_until_ready(state)
elapsed = time.time() - start_time

total_steps = n_steps * n_envs
steps_per_sec = total_steps / elapsed

print(f"   ✓ Performance test complete")
print(f"\n" + "=" * 60)
print(f"RESULTS:")
print(f"=" * 60)
print(f"   - Total environments: {n_envs}")
print(f"   - Steps per environment: {n_steps}")
print(f"   - Total steps: {total_steps:,}")
print(f"   - Time elapsed: {elapsed:.2f} seconds")
print(f"   - Steps per second: {steps_per_sec:,.0f}")
print(f"   - FPS (per env): {n_steps/elapsed:.1f}")
print("=" * 60)

# Check state information
print(f"\n8. Checking state information...")
print(f"   - State type: {type(state)}")
if hasattr(state, 'obs'):
    print(f"   - Observation shape: {state.obs.shape}")
if hasattr(state, 'reward'):
    print(f"   - Reward shape: {state.reward.shape}")
    print(f"   - Average reward: {jnp.mean(state.reward):.6f}")
if hasattr(state, 'done'):
    print(f"   - Done count: {jnp.sum(state.done)}")

print("\n" + "=" * 60)
print("✓ MJX GPU test completed successfully!")
print("=" * 60)
