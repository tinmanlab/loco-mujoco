"""
Test VRAM usage and optimize environment count and batch size
"""
import os
import jax
import time
import subprocess
import jax.numpy as jnp
from loco_mujoco import ImitationFactory

# Optimize GPU performance
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, free, total = map(int, result.strip().split(','))
        return {'used': used, 'free': free, 'total': total}
    except:
        return {'used': 0, 'free': 0, 'total': 0}

print("=" * 70)
print("LocoMuJoCo VRAM Optimization Test")
print("=" * 70)

# Check initial GPU state
print("\n1. Initial GPU State:")
initial_mem = get_gpu_memory()
print(f"   - Total VRAM: {initial_mem['total']} MB")
print(f"   - Used VRAM: {initial_mem['used']} MB")
print(f"   - Free VRAM: {initial_mem['free']} MB")
print(f"   - JAX backend: {jax.default_backend()}")

# Create environment
print("\n2. Creating MJX environment...")
env = ImitationFactory.make("MjxUnitreeG1", default_dataset_conf=dict(task="walk"))
env_mem = get_gpu_memory()
print(f"   ✓ Environment created")
print(f"   - VRAM used: +{env_mem['used'] - initial_mem['used']} MB")

# Test different environment counts
env_counts = [16, 32, 64, 128, 256, 512]
results = []

print("\n3. Testing different environment counts:")
print(f"{'N_Envs':<10} {'VRAM (MB)':<12} {'Steps/sec':<12} {'FPS/env':<12} {'Status':<10}")
print("-" * 70)

for n_envs in env_counts:
    try:
        # Setup
        key = jax.random.key(0)
        keys = jax.random.split(key, n_envs + 1)
        key, env_keys = keys[0], keys[1:]

        # JIT compile
        rng_reset = jax.jit(jax.vmap(env.mjx_reset))
        rng_step = jax.jit(jax.vmap(env.mjx_step))
        rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

        # Reset
        state = rng_reset(env_keys)

        # Warmup
        for _ in range(5):
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            action = rng_sample_uni_action(action_keys)
            state = rng_step(state, action)

        # Check memory after warmup
        mem_after = get_gpu_memory()
        vram_used = mem_after['used']

        # Performance test
        n_steps = 200
        start_time = time.time()

        for i in range(n_steps):
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            action = rng_sample_uni_action(action_keys)
            state = rng_step(state, action)

        jax.block_until_ready(state)
        elapsed = time.time() - start_time

        total_steps = n_steps * n_envs
        steps_per_sec = total_steps / elapsed
        fps_per_env = n_steps / elapsed

        results.append({
            'n_envs': n_envs,
            'vram_mb': vram_used,
            'steps_per_sec': steps_per_sec,
            'fps_per_env': fps_per_env,
            'status': 'OK'
        })

        print(f"{n_envs:<10} {vram_used:<12} {steps_per_sec:<12,.0f} {fps_per_env:<12.1f} {'✓ OK':<10}")

        # Clean up
        del state, rng_reset, rng_step, rng_sample_uni_action
        jax.clear_caches()

    except Exception as e:
        error_msg = str(e)[:30]
        results.append({
            'n_envs': n_envs,
            'vram_mb': 0,
            'steps_per_sec': 0,
            'fps_per_env': 0,
            'status': f'FAILED'
        })
        print(f"{n_envs:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'✗ FAIL':<10}")
        break

# Find optimal configuration
print("\n" + "=" * 70)
print("ANALYSIS & RECOMMENDATIONS")
print("=" * 70)

valid_results = [r for r in results if r['status'] == 'OK']
if valid_results:
    # Best throughput
    best_throughput = max(valid_results, key=lambda x: x['steps_per_sec'])
    print(f"\n1. Best Throughput:")
    print(f"   - N_Envs: {best_throughput['n_envs']}")
    print(f"   - Steps/sec: {best_throughput['steps_per_sec']:,.0f}")
    print(f"   - VRAM: {best_throughput['vram_mb']} MB")

    # Best efficiency (steps per MB)
    for r in valid_results:
        r['efficiency'] = r['steps_per_sec'] / max(r['vram_mb'], 1)
    best_efficiency = max(valid_results, key=lambda x: x['efficiency'])
    print(f"\n2. Best Efficiency (Steps/sec per MB VRAM):")
    print(f"   - N_Envs: {best_efficiency['n_envs']}")
    print(f"   - Steps/sec: {best_efficiency['steps_per_sec']:,.0f}")
    print(f"   - VRAM: {best_efficiency['vram_mb']} MB")
    print(f"   - Efficiency: {best_efficiency['efficiency']:.2f} steps/sec/MB")

    # Recommended for RTX 3070 (8GB)
    safe_results = [r for r in valid_results if r['vram_mb'] < initial_mem['total'] * 0.8]
    if safe_results:
        recommended = max(safe_results, key=lambda x: x['steps_per_sec'])
        print(f"\n3. Recommended Configuration (80% VRAM limit):")
        print(f"   - N_Envs: {recommended['n_envs']}")
        print(f"   - Steps/sec: {recommended['steps_per_sec']:,.0f}")
        print(f"   - VRAM: {recommended['vram_mb']} MB ({recommended['vram_mb']/initial_mem['total']*100:.1f}%)")
        print(f"   - FPS per env: {recommended['fps_per_env']:.1f}")

print("\n" + "=" * 70)
print("BATCH SIZE RECOMMENDATIONS:")
print("=" * 70)
print("""
For training algorithms:
- PPO mini-batch: Use 1/4 to 1/8 of N_Envs
- GAIL/AMP: Use similar mini-batch sizes
- Adjust based on model size (larger models = smaller batches)

Example for training:
- If using 128 envs: mini_batch_size = 16-32
- If using 256 envs: mini_batch_size = 32-64
""")

print("=" * 70)
