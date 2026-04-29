"""Utility functions for MORL circuit sizing."""

from collections import OrderedDict

import numpy as np
import yaml


def load_yaml(path):
    """Load a YAML file with FullLoader (supports !!python/tuple, ordered dicts)."""
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def extract_global_goal(target_specs):
    """Extract the most ideal target from each spec range.

    For 'xxx_max' (minimize): takes the minimum of the range.
    For 'xxx_min' (maximize): takes the maximum of the range.
    """
    global_goal = []
    specs_id = list(target_specs.keys())
    for sid in specs_id:
        lo, hi = target_specs[sid]
        if sid.endswith("_max"):
            global_goal.append(float(lo))
        elif sid.endswith("_min"):
            global_goal.append(float(hi))
        else:
            raise ValueError(f"Spec '{sid}' must end with '_min' or '_max'")
    return np.array(global_goal), specs_id


def lookup(spec, goal_spec, style="normd"):
    """Normalize spec values relative to a goal reference.

    Returns signed normalized deltas. Negative = below goal; positive = above goal.
    """
    spec = np.asarray(spec, dtype=np.float32)
    goal_spec = np.asarray(goal_spec, dtype=np.float32)
    epsilon = 1e-9
    goal_safe = np.where(goal_spec == 0, epsilon, goal_spec)

    if style == "normd":
        delta = spec - goal_safe
        abs_delta = np.abs(delta) + epsilon
        denom = goal_safe + np.abs(spec) + epsilon
        norm_spec = np.sign(delta) * np.exp(np.log(abs_delta) - np.log(denom))
    elif style == "tanh":
        scale = 10.0
        norm_spec = np.tanh((spec - goal_safe) / (scale * goal_safe)) / np.tanh(
            1.0 / scale
        )
    else:
        raise ValueError(f"Unknown lookup style: {style}")
    return norm_spec


def compute_vector_reward(cur_specs, global_goal, specs_id, lookup_style="normd"):
    """Compute per-dimension vector reward.

    Each dimension = lookup(cur_specs[i], global_goal[i]).
    'max' specs (minimization) are sign-flipped so that positive = better.
    """
    rel = lookup(cur_specs, global_goal, style=lookup_style)
    for i, sid in enumerate(specs_id):
        if sid.endswith("_max"):
            rel[i] *= -1.0
    return rel
