#!/bin/bash
# Comparator (2 objectives: delay, power) — simplest circuit for initial testing.
# TT-only mode for speed.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate morl

cd "$(dirname "$0")/.."

python train.py \
  --yaml ./eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_gf180.yaml \
  --env_name COMP \
  --total_timesteps 100000 \
  --timesteps_per_iter 10000 \
  --seed 42 \
  --lr 3e-4 \
  --gamma 0.99 \
  --lookup_style normd \
  --no-corner_sim \
  --episode_len 30 \
  --wandb_project "MORL-Circuit-Sizing" \
  --run_name "GPI-PD-COMP-TTonly"
