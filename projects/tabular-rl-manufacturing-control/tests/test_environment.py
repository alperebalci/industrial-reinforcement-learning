import numpy as np

from manufacturing_rl.environment import Action, ManufacturingEnv


def test_reset_and_step_are_seed_reproducible():
    env1 = ManufacturingEnv()
    env2 = ManufacturingEnv()
    state1 = env1.reset(seed=7)
    state2 = env2.reset(seed=7)
    assert np.allclose(state1, state2)

    out1 = env1.step(Action.HOLD_SPEED)
    out2 = env2.step(Action.HOLD_SPEED)
    assert np.allclose(out1[0], out2[0])
    assert out1[1:] == out2[1:]


def test_maintenance_has_full_period_downtime():
    env = ManufacturingEnv()
    env.reset(seed=3)
    _, reward, _, info = env.step(Action.MAINTENANCE)
    assert info["units_produced"] == 0.0
    assert info["maintenance_cost"] == env.config.maintenance_cost
    assert reward == -env.config.maintenance_cost


def test_intensive_inspection_reduces_escape_fraction():
    normal = ManufacturingEnv()
    inspect = ManufacturingEnv()
    normal.reset(seed=11)
    inspect.reset(seed=11)

    _, _, _, info_normal = normal.step(Action.HOLD_SPEED)
    _, _, _, info_inspect = inspect.step(Action.INTENSIVE_INSPECTION)

    normal_fraction = info_normal["escaped_defects"] / max(
        info_normal["intrinsic_defective_units"], 1e-12
    )
    inspect_fraction = info_inspect["escaped_defects"] / max(
        info_inspect["intrinsic_defective_units"], 1e-12
    )
    assert inspect_fraction < normal_fraction
