"""Training script for MORL-based analog circuit parameter optimization.

Uses GPI-PD (Continuous Action) from morl-baselines with a custom ngspice env.
"""

import logging
import os
import sys
import time
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

# --- MOSAC imports ---
from morl_baselines.single_policy.ser.mosac_continuous_action import MOSAC
from morl_baselines.common.weights import random_weights, equally_spaced_weights
from morl_baselines.common.pareto import ParetoArchive
from morl_baselines.common.evaluation import (
    log_episode_info,
    policy_evaluation_mo,
    log_all_multi_policy_metrics,
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
@click.option("--algo", "algo_choice", default=None,
              type=click.Choice(["gpi_pd", "mosac"]),
              help="Algorithm to use (default: gpi_pd). Overrides config if set.")
@click.option("--verbose/--quiet", default=False, help="Show DEBUG-level messages on console.")
def main(config_path, total_timesteps, seed, wandb_mode, num_envs, algo_choice, verbose):
    import yaml as _yaml
    with open(config_path) as f:
        cfg = _yaml.safe_load(f)

    # CLI overrides (only apply if explicitly passed)
    overrides = {
        "total_timesteps": total_timesteps,
        "seed": seed,
        "wandb_mode": wandb_mode,
        "num_envs": num_envs,
        "algo": algo_choice,
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

    algo = cfg.get("algo", "gpi_pd")
    logger.info("Algorithm: %s", algo)

    if algo == "mosac":
        train_mosac(cfg, yaml_path, run_dir, reward_dim)
        # Auto-generate reports
        try:
            import subprocess
            report_script = Path(__file__).parent / "scripts" / "generate_report.py"
            index_script = Path(__file__).parent / "scripts" / "generate_index.py"
            subprocess.run([sys.executable, str(report_script), str(run_dir)], check=False)
            subprocess.run([sys.executable, str(index_script)], check=False)
        except Exception:
            logger.warning("Report generation failed (non-fatal)", exc_info=True)
        return

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
        logger.warning("Report generation failed (non-fatal)", exc_info=True)


def _get_reward_dim(yaml_path):
    from utils import extract_global_goal, load_yaml
    yaml_data = load_yaml(yaml_path)
    _, specs_id = extract_global_goal(yaml_data["target_specs"])
    return len(specs_id)


def train_mosac(cfg, yaml_path, run_dir, reward_dim):
    """Run MOSAC training with random per-episode weights.

    Args:
        cfg: Full config dict (with CLI overrides applied).
        yaml_path: Path to the circuit YAML config.
        run_dir: Pathlib Path to the run directory.
        reward_dim: Number of reward dimensions (specs).
    """
    logger = logging.getLogger("morl_ckt.train")

    # ---- environments ----
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
    train_env.action_space.seed(seed)
    np.random.seed(seed)

    # ---- agent ----
    initial_weights = random_weights(reward_dim, dist="dirichlet")
    agent = MOSAC(
        env=train_env,
        weights=initial_weights,
        buffer_size=cfg["buffer_size"],
        gamma=cfg["gamma"],
        tau=cfg.get("tau", 0.005),
        batch_size=cfg["batch_size"],
        learning_starts=cfg["learning_starts"],
        policy_lr=cfg.get("policy_lr", 3e-4),
        q_lr=cfg.get("q_lr", 1e-3),
        alpha=cfg.get("sac_alpha", 0.2),
        autotune=cfg.get("sac_autotune", True),
        policy_freq=cfg.get("policy_freq", 2),
        log=False,
        seed=seed,
    )

    # ---- eval setup ----
    archive = ParetoArchive()
    num_eval_weights = cfg.get("num_eval_weights", 5)
    eval_weights = equally_spaced_weights(reward_dim, n=num_eval_weights)
    ref_point = np.full(reward_dim, -50.0)

    total_timesteps = cfg["total_timesteps"]
    eval_freq = cfg.get("eval_freq", 5000)

    # ---- training loop ----
    start_time = time.time()
    obs, _ = train_env.reset()
    current_weight = random_weights(reward_dim, dist="dirichlet")
    agent.set_weights(current_weight)

    logger.info("Initial weight: %s", current_weight,
                extra={"json_event": {
                    "event": "weight_vector",
                    "weight": current_weight.tolist(),
                    "global_step": 0,
                }})

    for global_step in range(1, total_timesteps + 1):
        # --- action selection ---
        if global_step < cfg["learning_starts"]:
            action = train_env.action_space.sample()
        else:
            action = agent.eval(obs)

        # --- environment step ---
        next_obs, reward, terminated, truncated, infos = train_env.step(action)

        # --- buffer add ---
        real_next_obs = next_obs
        if "final_observation" in infos:
            real_next_obs = infos["final_observation"]
        agent.buffer.add(
            obs=obs,
            next_obs=real_next_obs,
            action=action,
            reward=reward,
            done=terminated,
        )

        obs = next_obs

        # --- gradient update ---
        if global_step >= cfg["learning_starts"]:
            agent.global_step = global_step
            agent.update()

        # --- episode end ---
        if terminated or truncated:
            if "episode" in infos:
                log_episode_info(
                    infos["episode"], np.dot, current_weight, global_step,
                )

            obs, _ = train_env.reset()
            current_weight = random_weights(reward_dim, dist="dirichlet")
            agent.set_weights(current_weight)

            logger.info("Weight: %s", current_weight,
                        extra={"json_event": {
                            "event": "weight_vector",
                            "weight": current_weight.tolist(),
                            "global_step": global_step,
                        }})

        # --- periodic evaluation ---
        if global_step % eval_freq == 0:
            front = []
            for w in eval_weights:
                _, _, vec_return, _ = policy_evaluation_mo(
                    agent, eval_env, w, rep=1,
                )
                front.append(vec_return)
                archive.add(None, vec_return)

            log_all_multi_policy_metrics(
                current_front=front,
                hv_ref_point=ref_point,
                reward_dim=reward_dim,
                global_step=global_step,
                n_sample_weights=num_eval_weights,
            )

        # --- periodic SPS ---
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            logger.info("Training: SPS=%d global_step=%d", sps, global_step,
                        extra={"json_event": {
                            "event": "training_metrics",
                            "SPS": sps,
                            "global_step": global_step,
                        }})

    # ---- final: save Pareto front ----
    final_front = archive.evaluations
    logger.info("Training complete. Final Pareto front size: %d", len(final_front),
                extra={"json_event": {
                    "event": "training_complete",
                    "pareto_front": [f.tolist() for f in final_front],
                    "pareto_front_size": len(final_front),
                    "global_step": total_timesteps,
                }})

    # ---- cleanup ----
    train_env.close()
    eval_env.close()

    return agent, archive


if __name__ == "__main__":
    main()
