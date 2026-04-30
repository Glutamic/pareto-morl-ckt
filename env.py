"""MORL environment for analog circuit parameter optimization."""

import os
from collections import OrderedDict
from types import SimpleNamespace

import gymnasium
import numpy as np
from gymnasium import spaces

from eval_engines.ngspice.CircuitClass import CircuitClass
from utils import compute_vector_reward, extract_global_goal, load_yaml, lookup


class MorlNgspiceEnv(gymnasium.Env):
    """Multi-objective RL env for ngspice-based circuit sizing.

    Action space: continuous Box(low=-1, high=1) — normalized parameter deltas.
    Observation: [cur_spec_norm(N), cur_params(M)] — flat Box.
      - cur_spec_norm: lookup(cur_specs, global_goal), max-specs sign-flipped
      - cur_params: normalized in [-1, 1]
    Reward: vector of per-spec tanh-normalized deltas (positive = goal met).

    Both observation and reward use global_goal as the normalization reference.
    """

    ACT_LOW = -1.0
    ACT_HIGH = 1.0

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}
        super().__init__()

        self._root = os.getcwd()
        self.yaml_path = env_config.get("yaml_path")
        if self.yaml_path is None:
            raise ValueError("env_config must contain 'yaml_path'")

        yaml_data = load_yaml(self.yaml_path)
        self.yaml_data = yaml_data

        # --- parameters ---
        params = yaml_data["params"]
        self.params_id = list(params.keys())
        self.params_val = list(params.values())  # [(min, max), ...]
        self.num_params = len(self.params_id)

        # --- specs ---
        target_specs = yaml_data["target_specs"]
        self.global_goal, self.specs_id = extract_global_goal(target_specs)
        self.num_specs = len(self.specs_id)

        # --- simulation environments ---
        tt_netlist = [yaml_data["dsn_netlist"][0]]
        all_netlists = list(yaml_data["dsn_netlist"])
        self.num_corners = len(all_netlists)

        self.tt_sim_env = CircuitClass(
            yaml_path=self.yaml_path, path=self._root, design_netlists=tt_netlist,
        )
        self.full_sim_env = CircuitClass(
            yaml_path=self.yaml_path, path=self._root, design_netlists=all_netlists,
        )

        # --- settings ---
        self.episode_len = env_config.get("episode_len", 30)
        self.corner_sim = env_config.get("corner_sim", False)
        self.prec_params = yaml_data.get("prec_params", [9] * self.num_params)
        self.yaml_init_params = yaml_data.get("init_params")

        # --- action space ---
        self.action_space = spaces.Box(
            low=np.full(self.num_params, self.ACT_LOW),
            high=np.full(self.num_params, self.ACT_HIGH),
            dtype=np.float64,
        )

        # --- observation space: [cur_spec_norm(N), cur_params(M)] ---
        obs_dim = self.num_specs + self.num_params
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64,
        )

        # --- reward space (required by GPIPD) ---
        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_specs,), dtype=np.float64,
        )

        basename = os.path.splitext(os.path.basename(self.yaml_path))[0]
        self.spec = SimpleNamespace(id=basename)

        # --- state ---
        self.cur_params = self._init_params()
        self.cur_step = 0
        self.episode_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_step = 0
        self.episode_count += 1
        self.cur_params = self._init_params()

        cur_specs_raw, _ = self._simulate(self.cur_params)
        vector_reward = compute_vector_reward(
            cur_specs_raw, self.global_goal, self.specs_id,
        )
        obs = self._build_obs(cur_specs_raw)

        info = {
            "cur_specs": cur_specs_raw,
            "params": self._translate_params(self.cur_params),
        }

        print(f"[ep {self.episode_count} reset] "
              f"specs={cur_specs_raw.round(1)} "
              f"reward={vector_reward.round(3)} "
              f"params_norm={self.cur_params.round(3)}")
        return obs, info

    def step(self, action):
        self.cur_params = self._update_params(action)
        cur_specs_raw, corner_done = self._simulate(self.cur_params)
        vector_reward = compute_vector_reward(
            cur_specs_raw, self.global_goal, self.specs_id,
        )

        self.cur_step += 1
        terminated = False
        truncated = self.cur_step >= self.episode_len

        obs = self._build_obs(cur_specs_raw)

        print(f"[ep {self.episode_count} step {self.cur_step}] "
              f"reward={vector_reward.round(3)} "
              f"obs_spec={obs[:self.num_specs].round(3)} "
              f"obs_params={obs[self.num_specs:].round(3)} "
              f"raw_specs={cur_specs_raw.round(1)} "
              f"{'corner' if corner_done else 'TT'}")

        info = {
            "cur_specs": cur_specs_raw,
            "params": self._translate_params(self.cur_params),
            "corner_sim_done": corner_done,
        }
        return obs, vector_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal: observation (uses global_goal as reference)
    # ------------------------------------------------------------------

    def _build_obs(self, raw_specs):
        norm = compute_vector_reward(raw_specs, self.global_goal, self.specs_id)
        return np.concatenate([norm, self.cur_params])

    # ------------------------------------------------------------------
    # Internal: simulation
    # ------------------------------------------------------------------

    def _simulate(self, norm_params):
        physical = self._translate_params(norm_params)
        param_dict = OrderedDict(zip(self.params_id, physical))

        if self.corner_sim:
            _states, specs_list, _infos = self.full_sim_env.run(param_dict)
            corner_done = True
        else:
            _states, specs_list, _infos = self.tt_sim_env.run(param_dict)
            corner_done = False

        spec_arrays = []
        for spec in specs_list:
            sorted_spec = OrderedDict(sorted(spec.items(), key=lambda k: k[0]))
            spec_arrays.append(np.array(list(sorted_spec.values())))

        all_specs = np.array(spec_arrays)

        if self.corner_sim and len(spec_arrays) > 1:
            reverse_indices = [
                i for i, sid in enumerate(self.specs_id) if sid.endswith("_max")
            ]
            worst = all_specs.copy()
            worst[:, reverse_indices] *= -1.0
            worst_idx = np.argmin(worst, axis=0)
            worst_specs = all_specs[worst_idx, np.arange(self.num_specs)]
        else:
            worst_specs = all_specs[0]

        return worst_specs, corner_done

    # ------------------------------------------------------------------
    # Internal: parameter helpers
    # ------------------------------------------------------------------

    def _init_params(self):
        """Initialize normalized params by inverse-mapping from YAML init_params."""
        if self.yaml_init_params is None:
            return np.zeros(self.num_params, dtype=np.float64)
        norm = []
        for i, phys in enumerate(self.yaml_init_params):
            lo, hi = self.params_val[i]
            # Invert: phys = lo + (hi - lo) * (norm + 1) / 2
            norm.append(2.0 * (phys - lo) / (hi - lo) - 1.0)
        return np.clip(np.array(norm, dtype=np.float64), self.ACT_LOW, self.ACT_HIGH)

    def _update_params(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        new = self.cur_params + action
        return np.clip(new, self.ACT_LOW, self.ACT_HIGH)

    def _translate_params(self, norm_params):
        """Convert normalized params [-1, 1] to physical values [min, max]."""
        physical = []
        for i, p in enumerate(norm_params):
            lo, hi = self.params_val[i]
            val = lo + (hi - lo) * (p - self.ACT_LOW) / (self.ACT_HIGH - self.ACT_LOW)
            val = np.round(val, decimals=self.prec_params[i])
            physical.append(val)
        return physical
