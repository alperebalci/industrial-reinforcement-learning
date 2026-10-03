"""Stochastic manufacturing environment for joint production, maintenance and inspection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    DECREASE_SPEED = 0
    HOLD_SPEED = 1
    INCREASE_SPEED = 2
    MAINTENANCE = 3
    INTENSIVE_INSPECTION = 4


@dataclass(frozen=True)
class ManufacturingConfig:
    horizon: int = 100
    min_rate: float = 5.0
    max_rate: float = 20.0
    initial_rate: float = 10.0
    nominal_temperature: float = 25.0
    max_temperature: float = 110.0
    production_revenue: float = 100.0
    escaped_defect_cost: float = 500.0
    detected_defect_rework_cost: float = 80.0
    maintenance_cost: float = 2000.0
    operating_cost_per_unit: float = 5.0
    intensive_inspection_cost_per_unit: float = 10.0
    default_detection_probability: float = 0.25
    intensive_detection_probability: float = 0.92
    inspection_throughput_factor: float = 0.85
    maintenance_wear_reduction: float = 55.0
    failure_wear: float = 100.0
    failure_temperature: float = 108.0


class ManufacturingEnv:
    """Small auditable simulator with five observable state variables.

    Observation, normalized to [0, 1]:
    [temperature, wear, workload, intrinsic process quality, production rate].

    Intensive inspection changes defect detection, not intrinsic process quality.
    Maintenance creates one full period of downtime.
    """

    n_actions = len(Action)

    def __init__(self, config: ManufacturingConfig | None = None):
        self.config = config or ManufacturingConfig()
        self.rng = np.random.default_rng()
        self.history: list[dict] = []
        self.t = 0
        self.machine_temp = 0.0
        self.machine_wear = 0.0
        self.workload = 0.0
        self.process_quality = 0.0
        self.production_rate = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        cfg = self.config
        self.t = 0
        self.machine_temp = float(
            np.clip(cfg.nominal_temperature + self.rng.normal(0.0, 3.0), 15.0, 45.0)
        )
        self.machine_wear = float(self.rng.uniform(2.0, 10.0))
        self.workload = float(self.rng.uniform(35.0, 65.0))
        self.process_quality = float(self.rng.uniform(0.93, 0.97))
        self.production_rate = cfg.initial_rate
        self.history = []
        return self._observation()

    def _observation(self) -> np.ndarray:
        cfg = self.config
        obs = np.array(
            [
                self.machine_temp / cfg.max_temperature,
                self.machine_wear / cfg.failure_wear,
                self.workload / 100.0,
                self.process_quality,
                (self.production_rate - cfg.min_rate) / (cfg.max_rate - cfg.min_rate),
            ],
            dtype=np.float64,
        )
        return np.clip(obs, 0.0, 1.0)

    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, dict]:
        action = Action(int(action))
        cfg = self.config
        self.t += 1

        # Draw a fixed shock vector every period so policies evaluated with the
        # same seed face matched exogenous randomness.
        temp_noise, wear_noise, quality_noise, workload_noise = self.rng.normal(
            loc=[0.0, 0.0, 0.0, 0.0], scale=[1.5, 0.12, 0.0025, 4.0]
        )

        maintenance = action == Action.MAINTENANCE
        intensive_inspection = action == Action.INTENSIVE_INSPECTION

        if action == Action.DECREASE_SPEED:
            self.production_rate = max(cfg.min_rate, self.production_rate - 2.0)
        elif action == Action.INCREASE_SPEED:
            self.production_rate = min(cfg.max_rate, self.production_rate + 2.0)
        elif maintenance:
            self.machine_wear = max(
                0.0, self.machine_wear - cfg.maintenance_wear_reduction
            )
            self.machine_temp = max(
                cfg.nominal_temperature, self.machine_temp - 18.0
            )

        scheduled_rate = 0.0 if maintenance else self.production_rate
        throughput_factor = (
            cfg.inspection_throughput_factor if intensive_inspection else 1.0
        )
        units_produced = scheduled_rate * throughput_factor

        detection_probability = (
            cfg.intensive_detection_probability
            if intensive_inspection
            else cfg.default_detection_probability
        )
        intrinsic_defective_units = units_produced * (1.0 - self.process_quality)
        detected_defects = intrinsic_defective_units * detection_probability
        escaped_defects = intrinsic_defective_units - detected_defects

        revenue = units_produced * cfg.production_revenue
        operating_cost = units_produced * cfg.operating_cost_per_unit
        escaped_defect_cost = escaped_defects * cfg.escaped_defect_cost
        rework_cost = detected_defects * cfg.detected_defect_rework_cost
        inspection_cost = (
            units_produced * cfg.intensive_inspection_cost_per_unit
            if intensive_inspection
            else 0.0
        )
        maintenance_cost = cfg.maintenance_cost if maintenance else 0.0
        profit = (
            revenue
            - operating_cost
            - escaped_defect_cost
            - rework_cost
            - inspection_cost
            - maintenance_cost
        )

        production_load = units_produced / cfg.max_rate
        self.machine_temp += (
            -0.12 * (self.machine_temp - cfg.nominal_temperature)
            + 8.0 * production_load
            + 0.018 * self.machine_wear
            + temp_noise
        )
        self.machine_temp = float(
            np.clip(self.machine_temp, 15.0, cfg.max_temperature + 5.0)
        )

        if not maintenance:
            self.machine_wear += (
                0.25
                + 1.35 * production_load
                + 0.009 * max(0.0, self.machine_temp - 45.0)
                + 0.003 * self.workload
                + wear_noise
            )
        self.machine_wear = float(
            np.clip(self.machine_wear, 0.0, cfg.failure_wear + 5.0)
        )

        quality_recovery = 0.035 * (0.97 - self.process_quality)
        quality_damage = (
            0.00020 * max(0.0, self.machine_temp - 50.0)
            + 0.000075 * self.machine_wear
            + 0.0025 * production_load
        )
        self.process_quality = float(
            np.clip(
                self.process_quality
                + quality_recovery
                - quality_damage
                + quality_noise,
                0.50,
                0.99,
            )
        )

        self.workload += 0.10 * (55.0 - self.workload) + workload_noise
        self.workload = float(np.clip(self.workload, 10.0, 100.0))

        failure = (
            self.machine_wear >= cfg.failure_wear
            or self.machine_temp >= cfg.failure_temperature
        )
        done = self.t >= cfg.horizon or failure

        info = {
            "step": self.t,
            "action": int(action),
            "profit": float(profit),
            "units_produced": float(units_produced),
            "intrinsic_defective_units": float(intrinsic_defective_units),
            "detected_defects": float(detected_defects),
            "escaped_defects": float(escaped_defects),
            "inspection_cost": float(inspection_cost),
            "rework_cost": float(rework_cost),
            "maintenance_cost": float(maintenance_cost),
            "maintenance": maintenance,
            "intensive_inspection": intensive_inspection,
            "failure": bool(failure),
            "temperature": float(self.machine_temp),
            "wear": float(self.machine_wear),
            "workload": float(self.workload),
            "process_quality": float(self.process_quality),
            "production_rate": float(self.production_rate),
        }
        self.history.append(info.copy())
        return self._observation(), float(profit), bool(done), info
