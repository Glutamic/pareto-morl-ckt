"""Utility functions for MORL circuit sizing."""

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


def lookup(spec, reference):
    """Normalize spec values relative to a reference using tanh.

    Returns signed normalized deltas. Negative = worse than reference;
    positive = better than reference. Range approximately (-5, 5).
    """
    spec = np.asarray(spec, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    epsilon = 1e-9
    ref_safe = np.where(ref == 0, epsilon, ref)

    scale = 10.0
    norm_spec = np.tanh((spec - ref_safe) / (scale * ref_safe)) / np.tanh(1.0 / scale)
    return norm_spec


def compute_vector_reward(cur_specs, global_goal, specs_id):
    """Compute per-dimension vector reward.

    Each dimension = lookup(cur_specs[i], global_goal[i]).
    'max' specs (minimization) are sign-flipped so that positive = better.
    """
    rel = lookup(cur_specs, global_goal)
    for i, sid in enumerate(specs_id):
        if sid.endswith("_max"):
            rel[i] *= -1.0
    return rel
