"""Transparent non-learning policies for manufacturing control."""

from __future__ import annotations

import numpy as np

from .environment import Action


def fixed_speed_policy(state: np.ndarray) -> int:
    """Hold nominal speed except for an emergency maintenance threshold."""
    wear = float(state[1])
    temperature = float(state[0])
    if wear >= 0.88 or temperature >= 0.94:
        return int(Action.MAINTENANCE)
    return int(Action.HOLD_SPEED)


def threshold_policy(state: np.ndarray) -> int:
    """Reactive engineering rule using wear, temperature, quality and workload."""
    temperature, wear, workload, quality, rate = map(float, state)
    if wear >= 0.72 or temperature >= 0.84:
        return int(Action.MAINTENANCE)
    if quality <= 0.88 and workload <= 0.78:
        return int(Action.INTENSIVE_INSPECTION)
    if temperature >= 0.72 or wear >= 0.58:
        return int(Action.DECREASE_SPEED)
    if (
        workload >= 0.66
        and rate <= 0.82
        and wear <= 0.48
        and temperature <= 0.66
    ):
        return int(Action.INCREASE_SPEED)
    return int(Action.HOLD_SPEED)
