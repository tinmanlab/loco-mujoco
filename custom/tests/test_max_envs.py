"""
Find maximum number of environments that can run on RTX 3070
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

print("=" * 80)
print("Finding Maximum Number of Environments for RTX 3070")
print("=" * 80)

# Check initial state
initial_mem = get_gpu_memory()
print(f"\n1. Initial GPU State:")
print(f"   - Total VRAM: {initial_mem['total']} MB")
print(f"   - Used VRAM: {initial_mem['used']} MB")
print(f"   - Free VRAM: {initial_mem['free']} MB")

# Create environment
print(f"\n2. Creating MJX environment...")
env = ImitationFactory.make("MjxUnitreeG1", default_dataset_conf=dict(task="walk"))
env_mem = get_gpu_memory()
print(f"   ✓ Environment created")
print(f"   - VRAM after env creation: {env_mem['used']} MB (+{env_mem['used'] - initial_mem['used']} MB)")

# Test progressively larger environment counts
# Start from 512 and increase aggressively
env_counts = [512, 1024, 2048, 4096, 8192, 16384, 32768]

print(f"\n3. Testing maximum environment count:")
print(f"\n{'N_Envs':<10} {'VRAM (MB)':<12} {'VRAM %':<10} {'Steps/sec':<15} {'Status':<15}")
print("-" * 80)

max_working = 0
max_vram = 0
max_throughput = 0

for n_envs in env_counts:
    try:
        # Setup
        key = jax.random.key(0)
        keys = jax.random.split(key, n_envs + 1)
        key, env_keys = keys[0], keys[1:]

        print(f"{n_envs:<10} ", end='', flush=True)

        # JIT compile
        rng_reset = jax.jit(jax.vmap(env.mjx_reset))
        rng_step = jax.jit(jax.vmap(env.mjx_step))
        rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

        # Reset
        state = rng_reset(env_keys)
        jax.block_until_ready(state)

        # Check memory after reset
        mem_after_reset = get_gpu_memory()
        vram_used = mem_after_reset['used']
        vram_pct = (vram_used / initial_mem['total']) * 100

        print(f"{vram_used:<12} {vram_pct:<10.1f} ", end='', flush=True)

        # Warmup
        for _ in range(3):
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            action = rng_sample_uni_action(action_keys)
            state = rng_step(state, action)

        jax.block_until_ready(state)

        # Check memory after warmup
        mem_after_warmup = get_gpu_memory()
        vram_warmup = mem_after_warmup['used']

        # Performance test (shorter for high env counts)
        n_steps = 100 if n_envs <= 2048 else 50
        start_time = time.time()

        for i in range(n_steps):
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            action = rng_sample_uni_action(action_keys)
            state = rng_step(state, action)

        jax.block_until_ready(state)
        elapsed = time.time() - start_time

        # Check final memory
        mem_final = get_gpu_memory()
        vram_final = mem_final['used']
        vram_final_pct = (vram_final / initial_mem['total']) * 100

        total_steps = n_steps * n_envs
        steps_per_sec = total_steps / elapsed

        print(f"{steps_per_sec:<15,.0f} ✓ OK (final: {vram_final}MB, {vram_final_pct:.1f}%)")

        # Update max working
        max_working = n_envs
        max_vram = vram_final
        max_throughput = steps_per_sec

        # Clean up
        del state, rng_reset, rng_step, rng_sample_uni_action
        jax.clear_caches()

        # Small delay for memory to settle
        time.sleep(0.5)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)[:50]

        # Check if it's OOM
        if "out of memory" in error_msg.lower() or "OOM" in error_msg:
            print(f"{'N/A':<15} ✗ OOM")
        else:
            print(f"{'N/A':<15} ✗ {error_type}")

        # If we failed, we've hit the limit
        print(f"\n   ! Failed at {n_envs} environments")
        print(f"   ! Error: {error_type}: {error_msg}")
        break

# Summary
print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}")

if max_working > 0:
    print(f"\n✓ Maximum working configuration:")
    print(f"   - Max environments: {max_working:,}")
    print(f"   - VRAM usage: {max_vram} MB ({max_vram/initial_mem['total']*100:.1f}%)")
    print(f"   - Max throughput: {max_throughput:,.0f} steps/sec")
    print(f"   - Per-env FPS: {max_throughput/max_working:.1f}")

    # Calculate recommended settings
    recommended = max_working
    if max_vram / initial_mem['total'] > 0.9:
        recommended = int(max_working * 0.8)  # Use 80% for safety

    print(f"\n💡 Recommended configuration for stability:")
    print(f"   - N_Envs: {recommended:,}")
    print(f"   - Mini-batch suggestions:")
    print(f"     • Mini-batches (1/4): {recommended // 4}")
    print(f"     • Mini-batches (1/8): {recommended // 8}")
    print(f"     • Mini-batches (1/16): {recommended // 16}")
else:
    print("\n✗ Unable to determine maximum configuration")

print(f"\n{'=' * 80}")
print("NOTE: VRAM usage patterns in JAX/MJX")
print(f"{'=' * 80}")
print("""
JAX uses lazy memory allocation and XLA compilation optimizations:
1. Initial allocation reserves buffer space
2. Actual data is stored efficiently in device memory
3. XLA compiler optimizes memory layout
4. Small env counts may pre-allocate same buffer as large counts

This is NORMAL behavior and allows better performance!
The true limit will show when actual computation exceeds GPU capacity.
""")

print(f"{'=' * 80}")
