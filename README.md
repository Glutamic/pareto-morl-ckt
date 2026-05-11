# morl-ckt-sizing

Multi-objective RL analog circuit sizing using GPI-PD with ngspice simulation backend.

## Project structure

```
morl-ckt-sizing/
├── train.py              # Training entry point (Click CLI)
├── env.py                # MorlNgspiceEnv (gymnasium.Env, vector rewards)
├── utils.py              # YAML loading, global goal, spec normalization
├── logging_setup.py      # JSONL logging, run directory management
├── experiments/          # YAML experiment configs (--config)
├── scripts/              # Report generation (generate_report.py, generate_index.py)
├── tests/                # Env interface tests (test_env.py)
├── logs/                 # Training output (auto-generated)
└── eval_engines/
    ├── ngspice/
    │   ├── CircuitClass.py              # Simulation result parsing
    │   ├── ngspice_wrapper_parallel.py  # Ngspice wrapper (parallel)
    │   └── ngspice_inputs/
    │       ├── correct_inputs.py        # Path fix utility
    │       ├── netlist/                 # .cir netlist files
    │       └── yaml_files/              # Circuit YAML configs
    └── pdk/                # Process design kits (gf180, sky130A)
```

## Quick start

```bash
conda activate morl

# Quick test
python tests/test_env.py

# Quick training test (1k steps, no wandb)
python train.py --config experiments/comp_quick_test.yaml

# Full training
python train.py --config experiments/comp_tt_100k.yaml

# Override params for ad-hoc changes
python train.py --config experiments/comp_tt_100k.yaml --total_timesteps 5000 --num_envs 8 --wandb_mode disabled
```

## Available circuits

| Circuit | YAML | Netlist prefix |
|---------|------|----------------|
| Comparator | `comparator_gf180.yaml` | `comparator_gf180_*.cir` |
| Two-stage opamp | `two_stage_opamp_gf180.yaml` | `two_stage_opamp_cont_gf180_*.cir` |
| Folded cascode opamp | `cascode_miller_opamp_gf180.yaml` | `gym_folded_cascode_opamp_cont_gf180_*.cir` |
| LDO | `ldo_sky130.yaml` | `ldo_cont_sky130_*.cir` |

## Path handling

`.cir` netlist files use **relative paths** for `.include`/`.lib` directives and are
portable across machines. The ngspice wrapper auto-resolves them to absolute paths
at runtime — no manual path fixing needed.

If you add new `.cir` files that contain absolute paths, convert them:

```bash
python eval_engines/ngspice/ngspice_inputs/correct_inputs.py
```

Use `--to-abs` to go the other direction if needed.
