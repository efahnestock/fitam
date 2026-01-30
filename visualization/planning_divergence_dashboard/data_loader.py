"""Data loading utilities for the planning approach comparison dashboard."""

import pickle
from pathlib import Path
from typing import Any

DATA_BASE_PATH = Path("/data/diffusion/fitam_data/evaluations/all_test")


def get_approaches() -> list[str]:
    """List available approaches (directories in evaluations folder)."""
    if not DATA_BASE_PATH.exists():
        return []
    return sorted([
        d.name for d in DATA_BASE_PATH.iterdir()
        if d.is_dir()
    ])


def get_trials(approach: str) -> list[str]:
    """List all trials for a given approach."""
    approach_path = DATA_BASE_PATH / approach
    if not approach_path.exists():
        return []
    return sorted([
        d.name for d in approach_path.iterdir()
        if d.is_dir() and d.name.startswith("map_")
    ])


def get_common_trials(approach1: str, approach2: str) -> list[str]:
    """Find trials present in both approaches."""
    trials1 = set(get_trials(approach1))
    trials2 = set(get_trials(approach2))
    return sorted(trials1 & trials2)


def _get_trial_run_path(approach: str, trial: str) -> Path | None:
    """Get the path to the trial run directory (e.g., 0000005)."""
    trial_path = DATA_BASE_PATH / approach / trial
    if not trial_path.exists():
        return None

    # Find the run directory (typically named like 0000005)
    run_dirs = [d for d in trial_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not run_dirs:
        return None

    # Return the first (or only) run directory
    return sorted(run_dirs)[0]


def load_trial_data(approach: str, trial: str) -> dict[str, Any] | None:
    """Load and parse debug_info.pkl for a trial."""
    run_path = _get_trial_run_path(approach, trial)
    if run_path is None:
        return None

    debug_info_path = run_path / "debug_info.pkl"
    if not debug_info_path.exists():
        return None

    with open(debug_info_path, "rb") as f:
        data = pickle.load(f)

    # Get raw data
    cost_history = data.get("cost_history", [])
    state_transition_indexes = data.get("state_transition_indexes", [])

    # Compute accumulated cost at each planning iteration
    # cost_history contains per-step costs, we need cumulative sum
    accumulated_costs = []
    cumsum = 0.0
    for cost in cost_history:
        cumsum += float(cost)
        accumulated_costs.append(cumsum)

    # Sample accumulated costs at planning iterations (matching costmap indices)
    sampled_costs = []
    for idx in state_transition_indexes:
        if idx < len(accumulated_costs):
            sampled_costs.append(accumulated_costs[idx])

    return {
        "accumulated_cost": sampled_costs,
        "state_transition_indexes": state_transition_indexes,
    }


def get_costmap_count(approach: str, trial: str) -> int:
    """Count PNG files in local_costmap folder."""
    run_path = _get_trial_run_path(approach, trial)
    if run_path is None:
        return 0

    costmap_path = run_path / "local_costmap"
    if not costmap_path.exists():
        return 0

    return len(list(costmap_path.glob("*.png")))


def get_costmap_path(approach: str, trial: str, index: int) -> Path | None:
    """Get the path to a specific costmap image."""
    run_path = _get_trial_run_path(approach, trial)
    if run_path is None:
        return None

    costmap_path = run_path / "local_costmap" / f"{index:07d}.png"
    if not costmap_path.exists():
        return None

    return costmap_path
