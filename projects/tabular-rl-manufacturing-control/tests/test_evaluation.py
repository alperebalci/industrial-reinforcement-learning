from manufacturing_rl.baselines import fixed_speed_policy, threshold_policy
from manufacturing_rl.evaluation import evaluate_policy


def test_evaluation_reports_operational_kpis():
    result = evaluate_policy(threshold_policy, episodes=3, seed=41)
    for key in [
        "mean_profit",
        "mean_throughput",
        "mean_escaped_defects",
        "mean_maintenance_count",
        "mean_inspection_count",
        "mean_failure",
        "p10_profit",
    ]:
        assert key in result


def test_matched_seed_evaluation_is_reproducible():
    first = evaluate_policy(fixed_speed_policy, episodes=4, seed=55)
    second = evaluate_policy(fixed_speed_policy, episodes=4, seed=55)
    assert first == second
