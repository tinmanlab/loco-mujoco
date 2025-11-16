# Custom Scripts and Training Results

This folder contains all custom scripts, test files, and training results created during loco-mujoco experimentation.

## 📁 Folder Structure

```
custom/
├── tests/              # VRAM and GPU acceleration test scripts
├── viewers/
│   ├── working/        # Functional viewer scripts
│   └── deprecated/     # Old/broken viewer scripts (kept for reference)
├── training/
│   ├── successful/     # Successfully completed training runs
│   └── failed/         # Failed/incomplete training attempts
├── docs/               # Custom documentation
└── recordings/         # Video recordings from MuJoCo viewer
```

---

## 🧪 Tests (`tests/`)

GPU acceleration and VRAM optimization test scripts for RTX 3070 (8GB VRAM).

| File | Status | Description |
|------|--------|-------------|
| `test_basic_env.py` | ✅ Working | Basic environment creation and GPU detection test |
| `test_mjx_gpu.py` | ✅ Working | MJX GPU acceleration verification |
| `test_multiple_envs.py` | ✅ Working | Test 64-4096 parallel environments |
| `test_vram_optimization.py` | ✅ Working | VRAM usage measurement across env counts |
| `test_max_envs.py` | ✅ Working | Maximum environment count test (4096 envs @ 76% VRAM) |
| `test_training_vram.py` | ✅ Working | VRAM usage during actual training |
| `test_myoskeleton_gpu.py` | ⚠️ Deprecated | MyoSkeleton GPU test (muscle model issues) |

**Key Finding:** RTX 3070 can handle **4096 parallel environments** using only 76% VRAM due to JAX's efficient memory management.

---

## 👁️ Viewers (`viewers/`)

### Working (`viewers/working/`)

| File | Status | Environment | Description |
|------|--------|-------------|-------------|
| `view_skeleton_mocap.py` | ✅ Working | SkeletonTorque | Plays mocap trajectory with official viewer |
| `interactive_viewer.py` | ✅ Working | SkeletonTorque | Interactive viewer for trained policy (Ctrl+Click for force) |
| `unitreeh1_interactive.py` | ✅ Working | UnitreeH1 | Interactive viewer for UnitreeH1 trained policy |

**Usage:**
```bash
conda activate loco-mujoco
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# View mocap data
python custom/viewers/working/view_skeleton_mocap.py

# Interactive viewer with trained policy
python custom/viewers/working/unitreeh1_interactive.py
```

**Interactive Controls:**
- `Ctrl + Left Click`: Apply external force
- `Left Click + Drag`: Rotate camera
- `Right Click + Drag`: Move camera
- `Mouse Wheel`: Zoom
- `Space`: Pause/Resume
- `Double Click`: Select body

### Deprecated (`viewers/deprecated/`)

| File | Status | Issue |
|------|--------|-------|
| `play_skeleton_amp.py` | ❌ Failed | Observation dimension mismatch (65 vs 71) |
| `play_skeleton_interactive.py` | ❌ Failed | ffmpeg dependency + record=True hardcoded |
| `play_skeleton_mouse_control.py` | ❌ Failed | Segmentation fault (ffmpeg issue) |

---

## 🏋️ Training (`training/`)

### Successful (`training/successful/`)

#### 1. UnitreeH1 AMP (`my_amp_training/`)
- **Status:** ✅ Completed (75M timesteps)
- **Algorithm:** AMP (Adversarial Motion Priors)
- **Environment:** MjxUnitreeH1
- **Dataset:** walk mocap (default LocoMuJoCo dataset)
- **Results:**
  - Mean Episode Return: **154.12**
  - Mean Episode Length: **965.83**
- **Checkpoint:** `outputs/2025-10-31/02-01-09/AMPJax_saved.pkl`
- **Config:** `conf.yaml` (4096 envs, lr: 6e-5, disc_lr: 5e-5)

#### 2. SkeletonTorque AMP (`skeleton_amp_training/`)
- **Status:** ✅ Completed (50M timesteps)
- **Algorithm:** AMP (Adversarial Motion Priors)
- **Environment:** MjxSkeletonTorque (27 DOF)
- **Dataset:** walk mocap (35,200 transitions)
- **Checkpoint:** `outputs/2025-10-31/12-20-59/AMPJax_saved.pkl`
- **Config:** `conf.yaml` (4096 envs, lr: 6e-5, disc_lr: 5e-5)

### Failed (`training/failed/`)

#### 1. MyoSkeleton (`myoskeleton_training/`)
- **Status:** ❌ Failed
- **Issue:** DefaultReward incompatible with muscle-actuated model
- **Error:** `IndexError: index is out of bounds for axis 0 with size 0`
- **Cause:** `foots_on_ground[0]` accessed but MyoSkeleton has no foot contact sensors
- **Lesson:** Use official examples with appropriate reward functions for muscle models

#### 2. SkeletonTorque PPO (`skeleton_torque_training/`)
- **Status:** ❌ Incomplete (wrong approach)
- **Issue:** Started basic PPO without mocap data
- **Lesson:** SkeletonTorque should use imitation learning (AMP/GAIL/DeepMimic) with mocap data

---

## 📚 Documentation (`docs/`)

| File | Description |
|------|-------------|
| `SETUP_SUMMARY.md` | Virtual environment setup and installation guide |
| `VRAM_OPTIMIZATION_FINAL.md` | VRAM optimization findings and recommendations |

---

## 🎬 Recordings (`recordings/`)

Video recordings from MuJoCo viewer sessions stored in `recordings/LocoMuJoCo_recordings/`.

---

## 🔧 Environment Setup

```bash
# Activate environment
conda activate loco-mujoco

# Set library paths for MuJoCo viewer
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
```

---

## 📊 Training Configuration Summary

### Optimal Settings for RTX 3070 (8GB VRAM)

```yaml
experiment:
  num_envs: 4096          # Maximum parallel environments
  num_steps: 14           # Steps per environment per update
  lr: 6e-5                # Actor learning rate
  disc_lr: 5e-5           # Discriminator learning rate (AMP)
  update_epochs: 4        # PPO update epochs
  disc_minibatch_size: 4096
  num_minibatches: 32
  hidden_layers: [512, 256, 256]
```

**Performance:**
- 4096 envs × 14 steps = 57,344 transitions per update
- VRAM usage: ~76% (6.2GB / 8GB)
- Allows headroom for system overhead

---

## 🎯 Best Practices Learned

1. **Use imitation learning** (AMP/GAIL) with mocap data for humanoid locomotion
2. **Follow official examples** from `loco-mujoco/examples/`
3. **Check environments/README.md** for mocap availability per robot
4. **Use official viewer methods** (`play_trajectory()`, `play_policy_mujoco()`)
5. **Avoid custom wrappers** unless necessary
6. **Set `visualize_goal: false`** in config for cleaner interactive viewing
7. **Use `record=False`** in viewer to avoid ffmpeg dependency

---

## 📝 Notes

- All custom files have been moved here from the root directory
- Original loco-mujoco structure remains intact
- Deprecated files kept for reference but should not be used
- For production use, refer to `viewers/working/` scripts only

---

**Created:** 2025-10-31
**Hardware:** RTX 3070 (8GB VRAM)
**Conda Environment:** loco-mujoco (Python 3.11.14)
