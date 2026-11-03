# Optional iteration extension

`scripts/run_iteration_loop.py` is independent of main pipeline launch. It
supports candidate selection, training manifests, persistent train/export/eval
jobs and reflection. `orchestrator/iteration/` is canonical; obsolete duplicate
top-level implementations are not part of this release.

`configs/iteration_smoke_simulator.yaml` uses included synthetic label rows.
Its metrics and checkpoints are simulated and explicitly report zero real
optimizer steps. This smoke checks job state/resume contracts only.

Real use selects `loop.trainer_backend: bagel`, `evaluator_backend: benchmark`,
and an actual candidate pool. In addition to the protocol YAML, provide:

```yaml
bagel:
  vendor_root: vendor/bagel
  model_path: /path/to/BAGEL-7B-MoT
  python_executable: /path/to/bagel-environment/bin/python
  visible_devices: '0,1'
  nproc_per_node: 2
  total_steps: 1000
  gradient_accumulation_steps: 8
  expected_num_tokens: 4096
  max_num_tokens: 8192
  max_num_tokens_per_sample: 8192
  timeout_s: 86400
  poll_interval_s: 2
```

These values illustrate the backend contract, not a recommended training
recipe or a fixed effective batch. BAGEL uses token packing; inspect optimizer
coverage and consumed examples, not GPU count alone. Use
`scripts/prepare_bagel_episodes.py`, `audit_training_sources.py`, and
`plan_bagel_coverage.py` before a long job. `require_nine_tasks` can enforce
coverage of every task in real optimizer-consumed episodes.

`benchmark` requires explicit fixed subsets/protocols and command adapters.
See `benchmark_backends.py:benchmark_protocol` and tests for the validated
result contract: model/checkpoint identity, subset identity, real-vs-simulated
status, metrics and units. Missing commands or metrics are errors. Configure
private judge credentials outside the checkout. Private one-off GEdit judge
and proxy scripts are not installation dependencies in this public tree.

`configs/iteration_paper.yaml` is retained as the paper protocol target, not
advertised as an executable training deployment. No real training or paid
evaluation is started by setup, doctor, package installation, or unit tests.
