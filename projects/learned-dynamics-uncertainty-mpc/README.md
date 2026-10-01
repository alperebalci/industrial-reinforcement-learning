# Learned Dynamics and Uncertainty-Aware MPC

Implemented bounded research baseline, v0.1, 2026-10-01.

A bootstrap ensemble of three probabilistic neural transition models is fitted to 400 fixed logged transitions. Each model predicts a mean state increment and log variance using Gaussian negative log likelihood. CEM planning propagates trajectory particles with a fixed ensemble member per particle and shared noise across candidate plans.

Comparators are a base-stock rule, learned mean-cost MPC, learned mean-plus-standard-deviation MPC and a known-dynamics MPC. Both MPC model classes use the same continuous CEM search budget (48 candidates, 9 particles, 3 rounds, horizon 3). The known-dynamics reference has privileged plant knowledge, explicitly including the shifted efficiency. All evaluation controllers receive the same demand lookahead and matched realized demand/noise paths.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python study.py --output local-results.json
```

Core API: train_ensemble(logged_features,delta_targets), plan(models,state,forecast), evaluate(models). A test replaces the true plant function with an exception during learned planning to detect information leakage. Seven local checks passed. The executed benchmark uses three training seeds and eight disjoint ten-period evaluation episodes per nominal/efficiency-shift regime.

## Interpretation and limitations

Nominal learned-controller costs are close to the known-model reference on this small fixture. Under an unseen efficiency decrease, learned controllers degrade and produce substantially more stockouts; these negative results are retained. A mean-plus-standard-deviation objective is not a chance constraint, calibration guarantee or safety certificate. Action bounds alone do not guarantee service.

This is a PETS-inspired ensemble-MPC study, not a full PETS reproduction. There is no SAC comparator, pretrained world model, real-plant deployment, or model-free superiority claim. Model fitting uses synthetic logs, not online data collection. Committed results summarize the executed run; the script additionally writes per-episode costs.

Primary reference: https://arxiv.org/abs/1805.12114

Independent implementation. Existing repository license applies; existing flagship code is unchanged. See VALIDATION.json for environment and source fingerprints. checks.py is explicitly invoked rather than joining root pytest discovery.
