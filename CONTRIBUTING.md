# Contributing

Preserve the existing stage/dataclass/JSONL contracts and source provenance.
Keep runtime tool adapters in `executor/tools/`, implementation code in
`executor/tasks/`, and orchestration in `orchestrator/`. Make focused changes;
avoid moving all packages while changing algorithms.

Run focused regressions, then the full offline suite in a pipeline environment
with pytest installed. Mock network/model calls. A real model run requires an
explicit output root and deployment validation. Document GPU mapping, budgets,
schema changes and validation limits in pull requests. Never commit keys,
weights, videos, temp records or generated experiment outputs. Do not delete
source data or intermediates as part of setup, start or stop.
