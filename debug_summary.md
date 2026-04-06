# neuronx-cc Double Free / Compilation Debug Summary

## Problem Timeline

### 1. Unrecognized Arguments Error
- Running `switch_source.sh` → `runner.sh` produced `XlaRuntimeError` with `NCC_EARG002` — unrecognized arguments passed to `neuronx-cc` via `NEURON_CC_FLAGS`.
- **Root cause**: `NEURON_CC_FLAGS` is a flat string env var. The compiler splits on whitespace — no shell involved to interpret quotes. Single quotes (`'`) and escaped double quotes (`\"`) became literal characters.

### 2. Fix Attempts for Flag Parsing
| Attempt | Action | Result |
|---------|--------|--------|
| 1 | Replaced single quotes with escaped double quotes | Still failed — literal `\"` in tokens |
| 2 | Removed inner quotes, repeated flags per sub-option | Fixed hlo2tensorizer, but `--dump` still unrecognized |
| 3 | Commented out all unrecognized flags | Fixed argument errors |

### 3. Commented-Out Flags (Unrecognized by This neuronx-cc Version)
- `--internal-max-instruction-limit`
- `--internal-num-neuroncores-per-sengine`
- `--no-internal-hlo-remat`
- `--tensorizer-options`
- `--internal-hlo2tensorizer-options`
- `--internal-disable-dge-levels`
- `--internal-backend-options`
- `--ccop-pipeline-buffer-size`
- `--internal-compiler-debug-mode`
- `--dump`

### 4. Shell Syntax Error
- Commenting out all lines inside an `if` block body caused `syntax error near unexpected token 'fi'`. Fixed by adding `true` no-op command.

### 5. Double Free Corruption
- After fixing all flag issues, got `double free or corruption (out)` crash during compilation.
- **Suspected**: `LD_PRELOAD=/usr/lib/libtcmalloc.so` (tcmalloc) — commented out the entire block.
- **Result**: Crash persisted. Confirmed tcmalloc is NOT set anywhere (env, `/etc/ld.so.preload`, `/etc/environment`, `/etc/profile.d/`, venv activate).
- **Conclusion**: The double-free is inside the **neuronx-cc compiler subprocess** itself, not caused by tcmalloc or external interference. The ~365MB memory usage at crash time indicates the compiler crashed very early.

---

## Environment Details

### Versions (All Dev Builds)
| Package | Version |
|---------|---------|
| neuronx-cc | `2.0.244164.0a0+d62175bc` (dev, NOT `2.22.12471.0` found outside venv) |
| libneuronxla | `3.0.1229.0+577335d9` (dev) |
| jax-neuronx | `0.7.0.1.0.7571+fa4019c3` |
| JAX | `0.6.2` |
| jaxlib | `0.6.2` |
| Python | `3.10.19` |

### Venv
- Path: `/fsx/akshiaws/jaxmoe3/`
- Activated via: `source ../akshiaws/jaxmoe3/bin/activate`

### Hardware
- Platform: **trn2** (`NEURON_PLATFORM_TARGET_OVERRIDE=trn2`)
- Instance: `trn2.48xlarge-64`
- LNC: 2, `devices_per_node`: 64
- System RAM: 247Gi total, 107Gi available

---

## Active NEURON_CC_FLAGS
```
--framework=XLA
--target=trn2
--model-type transformer
--enable-mixed-precision-accumulation
-O1
--auto-cast=none
--hbm-scratchpad-page-size=1024
```

## Recognized Flags for This neuronx-cc Version
`--framework`, `--target`, `--logical-nc-config`/`--lnc`, `--enable-fast-loading-neuron-binaries`, `--enable-fast-context-switch`, `--auto-cast`, `--auto-cast-type`, `--output`, `--optlevel`/`-O`, `--model-type`, `--distribution-strategy`, `--enable-dge`, `--verbose`, `--logfile`, `--logfile-verbose`, `--enable-mixed-precision-accumulation`, `--disable-hlo-operand-type-check`, `--enable-saturate-infinity`, `--hbm-scratchpad-page-size`, `--execute-repetition`

---

## Key Files

| File | Role |
|------|------|
| `runners/switch_source.sh` | Entry point — sets env vars, calls `runner.sh` |
| `runner.sh` | Main runner — SLURM setup, PJRT/XLA/Neuron config, compiler flags, launches training |
| `axlearn/common/trainer.py` | `SpmdTrainer.compile_train_step()` (line ~1223) → `lowered_train_step.compile()` — crash site |
| `axlearn/experiments/text/gpt/envy.py` | Switch-Base model config (line 543+): 64 experts, hidden_dim=12×128, 12 heads |

### Model Config (Switch-Base)
- 64 experts, hidden_dim=12×128, 12 heads, 12 kv_heads
- hybridnorm ffn_structure
- Neuron mesh config for trn2.48xlarge-64
- TP=4, EP=4, SEQ=4, batch=4, `AXLEARN_NUM_LAYERS=1`, `AXLEARN_REPEATED=1`

---

## Compilation Path
```
libneuronxla (libncc.py)
  → parses NEURON_CC_FLAGS via shlex.split()
  → neuron_cc_wrapper.py: call_neuron_compiler()
    → subprocess.run() spawns neuronx-cc
      → neuronx-cc crashes with double-free
    → libneuronxla reports "process pool terminated abruptly"
```

---

## Root Cause Analysis

The three tightly coupled packages (`jax-neuronx`, `libneuronxla`, `neuronx-cc`) are **all dev builds** and likely have **version mismatch issues**:

- **neuronx-cc** (the compiler subprocess) is the crashing component.
- **libneuronxla** generates the HLO input fed to neuronx-cc — a mismatch between these two dev builds could trigger the bug.
- Re-enabling commented-out flags would NOT help — they are all `--internal-*` flags unrecognized by this neuronx-cc version.

## Recommended Next Steps

1. Get a **matched stable release set** of all three packages (`jax-neuronx` + `libneuronxla` + `neuronx-cc`) rather than downgrading one in isolation.
2. Check available stable releases: `pip index versions neuronx-cc`, `pip index versions libneuronxla`, `pip index versions jax-neuronx`.
3. Check pinned dependency requirements: `pip show jax-neuronx | grep Requires`.
4. Install a consistent, tested release set into a fresh venv.
