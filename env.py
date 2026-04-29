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
    Observation: [cur_spec_norm, global_goal_norm, cur_params] — flat Box.
    Reward: vector of per-spec normalized deltas (positive = satisfies goal).

    Two simulation modes:
      - TT-only (default): only simulate the typical-typical corner.
      - Corner worst-case: simulate all PVT corners and take worst per spec.
    """

    ACT_LOW = -1.0
    ACT_HIGH = 1.0

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}
        super().__init__()

        # --- paths ---
        self._root = os.getcwd()
        self.yaml_path = env_config.get("yaml_path")
        if self.yaml_path is None:
            raise ValueError("env_config must contain 'yaml_path'")

        # --- load YAML ---
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
        self.g_star = np.array(yaml_data["normalize"])  # normalization reference

        # --- sim environments ---
        tt_netlist = [yaml_data["dsn_netlist"][0]]
        corner_netlists = list(yaml_data["dsn_netlist"][1:])
        all_netlists = list(yaml_data["dsn_netlist"])
        self.num_corners = len(all_netlists)

        self.tt_sim_env = CircuitClass(
            yaml_path=self.yaml_path,
            path=self._root,
            design_netlists=tt_netlist,
        )
        self.corner_sim_env = CircuitClass(
            yaml_path=self.yaml_path,
            path=self._root,
            design_netlists=corner_netlists,
        )
        self.full_sim_env = CircuitClass(
            yaml_path=self.yaml_path,
            path=self._root,
            design_netlists=all_netlists,
        )

        # --- settings ---
        self.episode_len = env_config.get("episode_len", 30)
        self.lookup_style = env_config.get("lookup_style", "normd")
        self.corner_sim = env_config.get("corner_sim", False)
        self.prec_params = yaml_data.get("prec_params", [9] * self.num_params)

        # --- action space: normalized delta per param ---
        self.action_space = spaces.Box(
            low=np.full(self.num_params, self.ACT_LOW),
            high=np.full(self.num_params, self.ACT_HIGH),
            dtype=np.float64,
        )

        # --- observation space ---
        obs_dim = 2 * self.num_specs + self.num_params
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # --- reward space (required by MO-Gymnasium / GPIPD) ---
        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_specs,), dtype=np.float64
        )

        # --- spec id for wandb logging ---
        basename = os.path.splitext(os.path.basename(self.yaml_path))[0]
        self.spec = SimpleNamespace(id=basename)

        # --- state ---
        self.cur_params = np.zeros(self.num_params, dtype=np.float64)
        self.cur_step = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_step = 0
        self.cur_params = self._init_params()

        cur_specs_raw, _ = self._simulate(self.cur_params)
        vector_reward = compute_vector_reward(
            cur_specs_raw, self.global_goal, self.specs_id, self.lookup_style
        )
        global_goal_norm = self._norm_global_goal()
        info = {
            "cur_specs": cur_specs_raw,
            "params": self._translate_params(self.cur_params),
        }
        obs = self._build_obs(
            self._norm_specs(cur_specs_raw), global_goal_norm, self.cur_params
        )
        return obs, info

    def step(self, action):
        self.cur_params = self._update_params(action)
        cur_specs_raw, corner_done = self._simulate(self.cur_params)
        vector_reward = compute_vector_reward(
            cur_specs_raw, self.global_goal, self.specs_id, self.lookup_style
        )
        global_goal_norm = self._norm_global_goal()

        self.cur_step += 1
        terminated = False
        truncated = self.cur_step >= self.episode_len

        info = {
            "cur_specs": cur_specs_raw,
            "params": self._translate_params(self.cur_params),
            "corner_sim_done": corner_done,
        }
        obs = self._build_obs(
            self._norm_specs(cur_specs_raw), global_goal_norm, self.cur_params
        )
        return obs, vector_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal: observation
    # ------------------------------------------------------------------

    def _build_obs(self, cur_spec_norm, global_goal_norm, cur_params):
        return np.concatenate([cur_spec_norm, global_goal_norm, cur_params])

    def _norm_specs(self, raw_specs):
        norm = lookup(raw_specs, self.g_star, style=self.lookup_style)
        for i, sid in enumerate(self.specs_id):
            if sid.endswith("_max"):
                norm[i] *= -1.0
        return norm

    def _norm_global_goal(self):
        norm = lookup(self.global_goal, self.g_star, style=self.lookup_style)
        for i, sid in enumerate(self.specs_id):
            if sid.endswith("_max"):
                norm[i] *= -1.0
        return norm

    # ------------------------------------------------------------------
    # Internal: simulation
    # ------------------------------------------------------------------

    def _simulate(self, norm_params):
        """Run ngspice simulation and return worst-case raw spec values.

        TT-only mode: run only TT corner.
        Corner mode: run all corners, take element-wise worst.
        """
        physical = self._translate_params(norm_params)
        param_dict = OrderedDict(zip(self.params_id, physical))

        if self.corner_sim:
            states, specs_list, infos = self.full_sim_env.run(param_dict)
            corner_done = True
        else:
            states, specs_list, infos = self.tt_sim_env.run(param_dict)
            corner_done = False

        # Sort specs by key for consistent ordering
        spec_arrays = []
        for spec in specs_list:
            sorted_spec = OrderedDict(sorted(spec.items(), key=lambda k: k[0]))
            spec_arrays.append(np.array(list(sorted_spec.values())))

        all_specs = np.array(spec_arrays)  # (num_corners, num_specs)

        if self.corner_sim and len(spec_arrays) > 1:
            # Pick worst across corners per spec
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
        """Initialize parameters to center of normalized range."""
        return np.zeros(self.num_params, dtype=np.float64)

    def _update_params(self, action):
        """Apply action as a delta in normalized parameter space."""
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
