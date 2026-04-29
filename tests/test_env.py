"""Basic environment interface test."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "morl-baselines"))
import numpy as np
from env import MorlNgspiceEnv


YAML_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_gf180.yaml",
)


def test_env_creation():
    env = MorlNgspiceEnv(
        env_config={
            "yaml_path": YAML_PATH,
            "lookup_style": "normd",
            "corner_sim": False,
        }
    )
    assert env.observation_space.shape == (10,), f"obs shape: {env.observation_space.shape}"
    assert env.action_space.shape == (6,), f"action shape: {env.action_space.shape}"
    assert env.reward_space.shape == (2,), f"reward shape: {env.reward_space.shape}"
    assert env.unwrapped.reward_space.shape == (2,)
    assert env.spec.id == "comparator_gf180"
    print("  test_env_creation PASSED")


def test_reset():
    env = MorlNgspiceEnv(
        env_config={
            "yaml_path": YAML_PATH,
            "lookup_style": "normd",
            "corner_sim": False,
        }
    )
    obs, info = env.reset()
    assert obs.shape == (10,), f"obs shape: {obs.shape}"
    assert "cur_specs" in info
    assert "params" in info
    print("  test_reset PASSED")


def test_step():
    env = MorlNgspiceEnv(
        env_config={
            "yaml_path": YAML_PATH,
            "lookup_style": "normd",
            "corner_sim": False,
        }
    )
    env.reset()
    action = np.array([0.1, -0.05, 0.0, 0.2, -0.1, 0.05])
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward.shape == (2,), f"reward shape: {reward.shape}"
    assert not terminated, "should not terminate early"
    assert isinstance(truncated, bool)
    print("  test_step PASSED")


def test_truncation():
    env = MorlNgspiceEnv(
        env_config={
            "yaml_path": YAML_PATH,
            "lookup_style": "normd",
            "corner_sim": False,
            "episode_len": 3,
        }
    )
    env.reset()
    for i in range(3):
        action = np.zeros(6)
        _, _, _, truncated, _ = env.step(action)
    assert truncated, f"should truncate after episode_len steps"
    print("  test_truncation PASSED")


if __name__ == "__main__":
    print("Running env interface tests...")
    test_env_creation()
    test_reset()
    test_step()
    test_truncation()
    print("All env tests passed!")
