#!/bin/bash
# Two-Stage OpAmp (6 objectives → reduced to 2: gain, ibias for now).
source $(conda info --base)/etc/profile.d/conda.sh
conda activate morl

cd "$(dirname "$0")/.."

python train.py \
  --yaml ./eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp_gf180.yaml \
  --env_name TSA \
  --total_timesteps 100000 \
  --timesteps_per_iter 10000 \
  --seed 42 \
  --lr 3e-4 \
  --gamma 0.99 \

  --no-corner_sim \
  --episode_len 30 \
  --wandb_project "MORL-Circuit-Sizing" \
  --run_name "GPI-PD-TSA-TTonly"
