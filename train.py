"""Training script for MORL-based analog circuit parameter optimization.

Uses GPI-PD (Continuous Action) from morl-baselines with a custom ngspice env.
"""

import os
import sys

import click
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "morl-baselines"))
)
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPIPDContinuousAction,
)

from env import MorlNgspiceEnv


def make_env(yaml_path, lookup_style="normd", corner_sim=False, episode_len=30):
    """Create a MorlNgspiceEnv instance."""
    return MorlNgspiceEnv(
        env_config={
            "yaml_path": yaml_path,
            "lookup_style": lookup_style,
            "corner_sim": corner_sim,
            "episode_len": episode_len,
        }
    )


@click.command()
# --- circuit ---
@click.option("--yaml", "yaml_path", required=True, help="Path to circuit YAML config.")
@click.option("--env_name", default="COMP", help="Circuit name for logging (COMP/TSA/CMA/LDO).")
# --- training ---
@click.option("--total_timesteps", default=100000, help="Total training timesteps.")
@click.option("--timesteps_per_iter", default=10000, help="Timesteps per GPI-LS iteration.")
@click.option("--seed", default=42, help="Random seed.")
# --- rl ---
@click.option("--lr", default=3e-4, help="Learning rate.")
@click.option("--gamma", default=0.99, help="Discount factor.")
@click.option("--batch_size", default=256, help="Batch size.")
@click.option("--buffer_size", default=400000, help="Replay buffer size.")
@click.option("--learning_starts", default=1000, help="Random exploration steps before training.")
# --- env ---
@click.option("--lookup_style", default="normd", type=click.Choice(["normd", "tanh"]))
@click.option("--corner_sim/--no-corner_sim", default=False, help="Enable multi-corner worst-case simulation.")
@click.option("--episode_len", default=30, help="Max steps per episode.")
# --- GPI-PD ---
@click.option("--dyna/--no-dyna", default=False, help="Enable Dyna-style dynamics model.")
@click.option("--use_gpi/--no-use_gpi", default=True, help="Use GPI for action selection at evaluation.")
# --- logging ---
@click.option("--wandb_project", default="MORL-Circuit-Sizing", help="W&B project name.")
@click.option("--wandb_entity", default=None, help="W&B entity.")
@click.option("--run_name", default=None, help="W&B run name.")
@click.option("--wandb_mode", default="online", type=click.Choice(["online", "offline", "disabled"]), help="W&B mode.")
def main(**kwargs):
    cfg = {k: v for k, v in kwargs.items()}
    yaml_path = cfg["yaml_path"]
    reward_dim = _get_reward_dim(yaml_path)

    # --- wandb mode ---
    wandb_mode = cfg.get("wandb_mode", "online")
    if wandb_mode != "online":
        os.environ["WANDB_MODE"] = wandb_mode
    use_wandb = wandb_mode != "disabled"

    # --- create envs ---
    train_env = make_env(
        yaml_path,
        lookup_style=cfg["lookup_style"],
        corner_sim=cfg["corner_sim"],
        episode_len=cfg["episode_len"],
    )

    eval_env = make_env(
        yaml_path,
        lookup_style=cfg["lookup_style"],
        corner_sim=cfg["corner_sim"],
        episode_len=cfg["episode_len"],
    )

    # --- seed ---
    seed = cfg["seed"]
    train_env.action_space.seed(seed)
    np.random.seed(seed)

    # --- agent ---
    experiment_name = cfg.get("run_name") or f"GPI-PD-{cfg['env_name']}"
    agent = GPIPDContinuousAction(
        env=train_env,
        learning_rate=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        per=True,
        dyna=cfg["dyna"],
        use_gpi=cfg["use_gpi"],
        project_name=cfg["wandb_project"],
        wandb_entity=cfg["wandb_entity"],
        experiment_name=experiment_name,
        log=use_wandb,
        seed=seed,
    )

    ref_point = np.full(reward_dim, -2.0)

    agent.train(
        total_timesteps=cfg["total_timesteps"],
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        weight_selection_algo="gpi-ls",
        timesteps_per_iter=cfg["timesteps_per_iter"],
        eval_freq=1000,
        eval_mo_freq=cfg["timesteps_per_iter"],
    )


def _get_reward_dim(yaml_path):
    """Quickly read reward dimension from a YAML config."""
    from utils import extract_global_goal, load_yaml
    yaml_data = load_yaml(yaml_path)
    _, specs_id = extract_global_goal(yaml_data["target_specs"])
    return len(specs_id)


if __name__ == "__main__":
    main()
