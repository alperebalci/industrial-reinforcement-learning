"""Matched-seed evaluation for manufacturing policies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .environment import ManufacturingConfig, ManufacturingEnv

Policy = Callable[[np.ndarray], int]


def evaluate_policy(
    policy: Policy,
    *,
    episodes: int = 100,
    seed: int = 10_000,
    config: ManufacturingConfig | None = None,
) -> dict[str, float]:
    rows = []
    for episode in range(episodes):
        env = ManufacturingEnv(config)
        state = env.reset(seed=seed + episode)
        done = False
        totals = {
            "profit": 0.0,
            "throughput": 0.0,
            "escaped_defects": 0.0,
            "detected_defects": 0.0,
            "inspection_cost": 0.0,
            "rework_cost": 0.0,
            "maintenance_cost": 0.0,
            "maintenance_count": 0.0,
            "inspection_count": 0.0,
        }
        max_wear = env.machine_wear
        min_quality = env.process_quality
        failed = 0.0
        steps = 0

        while not done:
            action = int(policy(state))
            state, reward, done, info = env.step(action)
            totals["profit"] += reward
            totals["throughput"] += info["units_produced"]
            totals["escaped_defects"] += info["escaped_defects"]
            totals["detected_defects"] += info["detected_defects"]
            totals["inspection_cost"] += info["inspection_cost"]
            totals["rework_cost"] += info["rework_cost"]
            totals["maintenance_cost"] += info["maintenance_cost"]
            totals["maintenance_count"] += float(info["maintenance"])
            totals["inspection_count"] += float(info["intensive_inspection"])
            max_wear = max(max_wear, info["wear"])
            min_quality = min(min_quality, info["process_quality"])
            failed = max(failed, float(info["failure"]))
            steps += 1

        totals["max_wear"] = float(max_wear)
        totals["min_quality"] = float(min_quality)
        totals["failure"] = float(failed)
        totals["steps"] = float(steps)
        rows.append(totals)

    keys = rows[0].keys()
    result = {
        f"mean_{key}": float(np.mean([row[key] for row in rows]))
        for key in keys
    }
    result["p10_profit"] = float(
        np.percentile([row["profit"] for row in rows], 10)
    )
    result["p90_escaped_defects"] = float(
        np.percentile([row["escaped_defects"] for row in rows], 90)
    )
    return result
