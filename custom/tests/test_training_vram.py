"""
Test VRAM usage during ACTUAL TRAINING (not just environment stepping)
This includes network parameters, gradients, optimizer states, etc.
"""
import os
import jax
import time
import subprocess
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
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

# Simple Actor-Critic network (similar to PPO)
class ActorCritic(nn.Module):
    action_dim: int
    hidden_sizes: tuple = (512, 256)

    @nn.compact
    def __call__(self, x):
        # Shared layers
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.tanh(x)

        # Actor head
        actor = nn.Dense(self.action_dim)(x)

        # Critic head
        critic = nn.Dense(1)(x)

        return actor, critic

print("=" * 80)
print("VRAM Usage Test: ACTUAL TRAINING vs Environment Stepping")
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
obs_dim = env.info.observation_space.shape[0]
action_dim = env.info.action_space.shape[0]
print(f"   ✓ Environment created")
print(f"   - Obs dim: {obs_dim}, Action dim: {action_dim}")
print(f"   - VRAM: {env_mem['used']} MB (+{env_mem['used'] - initial_mem['used']} MB)")

# Test different environment counts with ACTUAL TRAINING
env_counts = [64, 128, 256, 512, 1024, 2048, 4096]

print(f"\n3. Testing with ACTUAL TRAINING (network + optimizer + gradients):")
print(f"\n{'N_Envs':<10} {'Env Only':<12} {'+ Network':<12} {'+ Train':<12} {'VRAM %':<10} {'Steps/s':<12} {'Status':<10}")
print("-" * 80)

results = []

for n_envs in env_counts:
    try:
        # Setup RNG
        key = jax.random.key(0)
        keys = jax.random.split(key, n_envs + 1)
        key, env_keys = keys[0], keys[1:]

        print(f"{n_envs:<10} ", end='', flush=True)

        # ===== STEP 1: Environment only =====
        rng_reset = jax.jit(jax.vmap(env.mjx_reset))
        rng_step = jax.jit(jax.vmap(env.mjx_step))
        rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

        state = rng_reset(env_keys)
        jax.block_until_ready(state)

        mem_env_only = get_gpu_memory()
        vram_env = mem_env_only['used']
        print(f"{vram_env:<12} ", end='', flush=True)

        # ===== STEP 2: Add network =====
        # Create network
        network = ActorCritic(action_dim=action_dim, hidden_sizes=(512, 256))

        # Initialize network
        key, init_key = jax.random.split(key)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = network.init(init_key, dummy_obs)

        # Replicate params across all envs (simulate batch processing)
        key, *action_keys = jax.random.split(key, n_envs + 1)
        action_keys = jnp.stack(action_keys)

        jax.block_until_ready(params)

        mem_with_network = get_gpu_memory()
        vram_network = mem_with_network['used']
        print(f"{vram_network:<12} ", end='', flush=True)

        # ===== STEP 3: Add optimizer and training state =====
        # Create optimizer (Adam with same settings as PPO)
        tx = optax.adam(learning_rate=1e-4)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx
        )

        jax.block_until_ready(train_state)

        mem_with_optimizer = get_gpu_memory()
        vram_optimizer = mem_with_optimizer['used']
        print(f"{vram_optimizer:<12} ", end='', flush=True)

        # ===== STEP 4: Simulate training loop =====
        # Define loss function
        def loss_fn(params, obs_batch, action_batch):
            # Forward pass
            action_pred, value_pred = network.apply(params, obs_batch)

            # Dummy loss (normally PPO loss here)
            action_loss = jnp.mean((action_pred - action_batch) ** 2)
            value_loss = jnp.mean(value_pred ** 2)

            return action_loss + value_loss

        # Compute gradients function
        @jax.jit
        def train_step(train_state, obs_batch, action_batch):
            loss, grads = jax.value_and_grad(loss_fn)(
                train_state.params, obs_batch, action_batch
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        # Warmup with training
        for _ in range(3):
            # Step environments
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            actions = rng_sample_uni_action(action_keys)
            state = rng_step(state, actions)

            # Get observations (dummy)
            obs_batch = jnp.ones((n_envs, obs_dim))
            action_batch = actions

            # Training step
            train_state, loss = train_step(train_state, obs_batch, action_batch)

        jax.block_until_ready(train_state)

        mem_after_warmup = get_gpu_memory()
        vram_final = mem_after_warmup['used']
        vram_pct = (vram_final / initial_mem['total']) * 100

        print(f"{vram_pct:<10.1f} ", end='', flush=True)

        # ===== STEP 5: Performance test =====
        n_steps = 50
        start_time = time.time()

        for i in range(n_steps):
            # Environment steps
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            actions = rng_sample_uni_action(action_keys)
            state = rng_step(state, actions)

            # Training step
            obs_batch = jnp.ones((n_envs, obs_dim))
            train_state, loss = train_step(train_state, obs_batch, actions)

        jax.block_until_ready(train_state)
        elapsed = time.time() - start_time

        total_steps = n_steps * n_envs
        steps_per_sec = total_steps / elapsed

        print(f"{steps_per_sec:<12,.0f} ✓ OK")

        results.append({
            'n_envs': n_envs,
            'vram_env': vram_env,
            'vram_network': vram_network,
            'vram_optimizer': vram_optimizer,
            'vram_final': vram_final,
            'vram_pct': vram_pct,
            'steps_per_sec': steps_per_sec,
            'status': 'OK'
        })

        # Clean up
        del state, train_state, params, network, tx
        del rng_reset, rng_step, rng_sample_uni_action
        jax.clear_caches()
        time.sleep(0.5)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)[:50]

        if "out of memory" in error_msg.lower() or "OOM" in error_msg:
            print(f"{'N/A':<10} ✗ OOM")
        else:
            print(f"{'N/A':<10} ✗ {error_type}")

        print(f"\n   ! Failed at {n_envs} environments")
        print(f"   ! Error: {error_type}: {error_msg}")
        break

# Analysis
print(f"\n{'=' * 80}")
print("DETAILED ANALYSIS")
print(f"{'=' * 80}")

if results:
    print(f"\n{'N_Envs':<10} {'Env MB':<10} {'Net MB':<10} {'Opt MB':<10} {'Final MB':<10} {'Delta MB':<10}")
    print("-" * 80)

    for r in results:
        delta_env = r['vram_network'] - r['vram_env']
        delta_net = r['vram_optimizer'] - r['vram_network']
        delta_total = r['vram_final'] - r['vram_env']

        print(f"{r['n_envs']:<10} {r['vram_env']:<10} {r['vram_network']:<10} "
              f"{r['vram_optimizer']:<10} {r['vram_final']:<10} {delta_total:<10}")

    # Find max working
    max_result = results[-1]

    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS FOR RTX 3070 (8GB)")
    print(f"{'=' * 80}")

    # Safe recommendation (80% VRAM)
    safe_results = [r for r in results if r['vram_pct'] < 80]
    if safe_results:
        rec = max(safe_results, key=lambda x: x['steps_per_sec'])
    else:
        rec = results[0]

    print(f"\n1. SAFE Configuration (< 80% VRAM):")
    print(f"   - N_Envs: {rec['n_envs']:,}")
    print(f"   - VRAM: {rec['vram_final']} MB ({rec['vram_pct']:.1f}%)")
    print(f"   - Steps/sec: {rec['steps_per_sec']:,.0f}")
    print(f"   - Mini-batch (1/8): {rec['n_envs'] // 8}")
    print(f"   - Mini-batch (1/4): {rec['n_envs'] // 4}")

    # Maximum configuration
    print(f"\n2. MAXIMUM Configuration (use with caution):")
    print(f"   - N_Envs: {max_result['n_envs']:,}")
    print(f"   - VRAM: {max_result['vram_final']} MB ({max_result['vram_pct']:.1f}%)")
    print(f"   - Steps/sec: {max_result['steps_per_sec']:,.0f}")
    print(f"   - Mini-batch (1/8): {max_result['n_envs'] // 8}")

    # Memory breakdown
    print(f"\n3. VRAM Breakdown (at {max_result['n_envs']} envs):")
    env_only = max_result['vram_env'] - initial_mem['used']
    network_add = max_result['vram_network'] - max_result['vram_env']
    optimizer_add = max_result['vram_optimizer'] - max_result['vram_network']
    runtime_add = max_result['vram_final'] - max_result['vram_optimizer']

    print(f"   - Environment data: {env_only} MB")
    print(f"   - Network params: {network_add} MB")
    print(f"   - Optimizer state: {optimizer_add} MB")
    print(f"   - Runtime overhead: {runtime_add} MB")
    print(f"   - Total: {max_result['vram_final'] - initial_mem['used']} MB")

print(f"\n{'=' * 80}")
