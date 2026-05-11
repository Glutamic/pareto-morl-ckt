"""Training script for MORL-based analog circuit parameter optimization.

Uses GPI-PD (Continuous Action) from morl-baselines with a custom ngspice env.
"""

import logging
import os
import sys
from pathlib import Path

import click
import numpy as np

from logging_setup import get_run_dir, setup_logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "morl-baselines"))
)
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPIPDContinuousAction,
)

from env import MorlNgspiceEnv, make_env_fn


def make_env(yaml_path, corner_sim=False, episode_len=30):
    return MorlNgspiceEnv(
        env_config={
            "yaml_path": yaml_path,
            "corner_sim": corner_sim,
            "episode_len": episode_len,
        }
    )


@click.command()
@click.option("--config", "config_path", required=True,
              help="Path to experiment YAML config.")
@click.option("--total_timesteps", default=None, type=int,
              help="Override total training timesteps.")
@click.option("--seed", default=None, type=int,
              help="Override random seed.")
@click.option("--wandb_mode", default=None,
              type=click.Choice(["online", "offline", "disabled"]),
              help="Override wandb mode.")
@click.option("--num_envs", default=None, type=int,
              help="Override number of parallel environments.")
@click.option("--verbose/--quiet", default=False, help="Show DEBUG-level messages on console.")
def main(config_path, total_timesteps, seed, wandb_mode, num_envs, verbose):
    import yaml as _yaml
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)

    # CLI overrides (only apply if explicitly passed)
    overrides = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "wandb_mode": wandb_mode,
        "num_envs": num_envs,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["verbose"] = verbose

    yaml_path = cfg["yaml_path"]

    # --- logging setup ---
    run_dir = get_run_dir(cfg["log_dir"], cfg["env_name"])
    console_level = "DEBUG" if cfg.get("verbose") else "INFO"
    _log_handles = setup_logging(run_dir, cfg, console_level=console_level)
    logger = logging.getLogger("morl_ckt.train")

    logger.info("Training started",
                extra={"json_event": {"event": "training_start", "config": cfg}})

    reward_dim = _get_reward_dim(yaml_path)

    wandb_mode = cfg.get("wandb_mode", "online")
    if wandb_mode != "online":
        os.environ["WANDB_MODE"] = wandb_mode
    use_wandb = wandb_mode != "disabled"

    num_envs = cfg.get("num_envs", 1)

    if num_envs > 1:
        from mo_gymnasium.wrappers.vector import MOAsyncVectorEnv
        env_fns = [make_env_fn(yaml_path, corner_sim=cfg["corner_sim"], episode_len=cfg["episode_len"])
                   for _ in range(num_envs)]
        train_env = MOAsyncVectorEnv(env_fns)
    else:
        train_env = make_env(
            yaml_path,
            corner_sim=cfg["corner_sim"],
            episode_len=cfg["episode_len"],
        )

    eval_env = make_env(
        yaml_path,
        corner_sim=cfg["corner_sim"],
        episode_len=cfg["episode_len"],
    )

    seed = cfg["seed"]
    if num_envs > 1:
        train_env.single_action_space.seed(seed)
        ref_env = make_env(yaml_path, corner_sim=cfg["corner_sim"], episode_len=cfg["episode_len"])
    else:
        train_env.action_space.seed(seed)
        ref_env = train_env
    np.random.seed(seed)

    logger.info("=== MORL Circuit Sizing ===")
    logger.info("YAML: %s", yaml_path)
    logger.info("Reward dim: %d, Params: %d, Obs dim: %d",
                reward_dim, ref_env.num_params,
                ref_env.observation_space.shape[0])
    logger.info("Global goal: %s", ref_env.global_goal)
    logger.info("Specs: %s", ref_env.specs_id)
    if num_envs > 1:
        logger.info("Num envs: %d (parallel)", num_envs)
        ref_env.close()
    logger.info("Total timesteps: %d, per iter: %d",
                cfg["total_timesteps"], cfg["timesteps_per_iter"])
    logger.info("Buffer: %d, Batch: %d, Learning starts: %d",
                cfg["buffer_size"], cfg["batch_size"], cfg["learning_starts"])
    logger.info("Corner sim: %s, GPI: %s, Dyna: %s",
                cfg["corner_sim"], cfg["use_gpi"], cfg["dyna"])
    logger.info("Log dir: %s", run_dir)
    logger.info("===========================")

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
        wandb_entity=cfg.get("wandb_entity"),
        experiment_name=experiment_name,
        log=use_wandb,
        seed=seed,
    )

    ref_point = np.full(reward_dim, -50.0)

    eval_env_fn = None
    if num_envs > 1:
        eval_env_fn = make_env_fn(yaml_path, corner_sim=cfg["corner_sim"], episode_len=cfg["episode_len"])

    agent.train(
        total_timesteps=cfg["total_timesteps"],
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        weight_selection_algo="gpi-ls",
        timesteps_per_iter=cfg["timesteps_per_iter"],
        eval_freq=1000,
        eval_mo_freq=cfg["timesteps_per_iter"],
        eval_env_fn=eval_env_fn,
        num_eval_workers=num_envs,
        num_eval_episodes_for_front=1,
    )

    # Auto-generate HTML reports after training
    try:
        import subprocess
        report_script = Path(__file__).parent / "scripts" / "generate_report.py"
        index_script = Path(__file__).parent / "scripts" / "generate_index.py"
        subprocess.run([sys.executable, str(report_script), str(run_dir)], check=False)
        subprocess.run([sys.executable, str(index_script)], check=False)
    except Exception:
        pass


def _get_reward_dim(yaml_path):
    from utils import extract_global_goal, load_yaml
    yaml_data = load_yaml(yaml_path)
    _, specs_id = extract_global_goal(yaml_data["target_specs"])
    return len(specs_id)


if __name__ == "__main__":
    main()
