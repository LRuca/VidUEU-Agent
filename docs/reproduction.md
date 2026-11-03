# Paper recipes

The four `configs/train_*.yaml` recipes record per-capability InternVL3/BAGEL
settings. `configs/eval_paper.yaml` records judges and passes. `vidueu reproduce
validate/plan --config ...` works offline without Torch, weights or API access.
`ok` means the protocol contract is valid; `ready_for_execution` is separate.
Templates intentionally list unresolved deployment fields.

InternVL3-8B understanding freezes the vision encoder, uses LoRA, two epochs,
AdamW lr 5e-5, global batch 32, cosine, warmup 0.03, bf16, gradient checkpointing,
ZeRO2 and two A100-40G reference GPUs. Planning checks GPU count x microbatch x
accumulation equals 32. The official trainer/data converter is not vendored.
The paper does not specify its LoRA rank or exact preprocessing here; supply
those explicitly rather than treating a guessed default as a paper setting.

BAGEL-7B-MoT freezes the VAE encoder/decoder and ViT; LoRA r16/alpha32,
one epoch, AdamW lr 5e-5, cosine, eight A100-40G reference GPUs. Understanding
uses text CE. Editing/unified also use image MSE and fully train `llm2vae`,
`vae2llm`, `time_embedder`. Token packing means GPU count x accumulation does
not establish equivalent batches/epochs. Supply measured steps-per-epoch, image
preprocessing and packing; audit consumed samples/tokens and trainable coverage.

`scripts/run_bagel_train.py` is a real short training/audit helper, defaulting
to two steps and 256px. It is not a full-paper trainer. A deployment adapter must
translate this recipe into a pinned trainer and audited approved-data conversion.
The optional native implementation and its identities remain in `vendor/bagel`
and `manifests/vendor_patches.json`; independent review stays external.

Execution requires `reproduce train/eval --execute`, complete deployment fields,
explicit GPUs, a full source commit, existing adapter/model/manifest and a new
output. The external adapter accepts `--recipe <absolute-config>` and records
exact arguments, immutable input identities, consumption and completion results.
No adapter is inferred and no GPU process is started by plan/validate. CPU smoke
acceptance performs no actual training or benchmark execution.

| Benchmark | Judge | Report |
| --- | --- | --- |
| LLaVA-Bench | Qwen3.5-35B-A3B | official protocol |
| MMDU | Qwen3-VL-235B-A22B-Instruct | dimensions and aggregate |
| GEdit-Bench-En | GPT-4.1 | G_SC, G_PQ, G_O, Average |
| WEAVEbench | GPT-4o | Science, Creation, Logic, Game, Average |

The native BAGEL evaluator covers MMDU/GEdit/WEAVE. Other understanding and
held-out benchmarks need official adapters. Preserve official splits, frozen
case IDs, input/target images, prompt/decoding/seed, base/trained hashes and judge
identity/date. Report failed/refused/skipped counts and denominators; do not
silently replace a judge or score a refusal. GEdit scores and WEAVE fractions
have different scales.

One generation and one judge pass is the default. To estimate paper judge
variance, use three judge repeats on the same fixed generated outputs, not
three generation runs. Iteration uses four 30K rounds, each from the same
initialization, then scales the fourth-round distribution to 100K; see iteration.md.
Secrets belong in ignored deployment files/environment variables.
