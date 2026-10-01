# OR-AI research extensions

- [Learned Dynamics and Uncertainty-Aware MPC](projects/learned-dynamics-uncertainty-mpc/): implemented bounded baseline, 7 local checks passed, 2026-10-01.

Adds a logged-data probabilistic ensemble and matched continuous CEM planning comparison. True plant access is excluded from learned planning. Observed distribution-shift failures are retained. Existing flagship implementations are unchanged; full existing regressions were not run. New checks and benchmarks execute in an isolated workflow.
