# Installation and asset preparation

For the normal one-command deployment, run `bash deploy.sh`. By default it
selects the `pipeline` and `omniparser` profiles, installs their isolated
Python environments, downloads and verifies the public example's assets, runs
both local runtime checks, and writes a path-aware config and launcher under
`.deploy/launch/`. It does not launch production inference, training, or
evaluation. Pass `--config path/to/pipeline.yaml` to derive profiles and assets
from its `tools` list. Add `--profile dev` for the lightweight release/smoke
environment, or explicitly request `vllm` and `bagel` when those optional
environments are needed. Use `--profile all` to request every profile.

| Command | Installed scope |
| --- | --- |
| `bash deploy.sh` | Pipeline and OmniParser; existing VLM API and CPU tools |
| `bash deploy.sh --profile bagel` | BAGEL training environment and weights; add this for auto-iteration after default deployment |
| `bash deploy.sh --profile all` | Pipeline, OmniParser, local vLLM, BAGEL and dev environments, plus all managed assets |

An extra `--profile` without `--config` selects exactly those profiles. With
`--config`, profiles inferred from tools are retained and requested profiles
are added. Installation of BAGEL does not provide benchmark datasets, fixed
subsets, judge credentials or user-specific evaluation adapters.

The host must provide the Python minor versions required by the selected
profiles, with venv, git, ffmpeg/ffprobe, and tmux. An NVIDIA driver is needed
for GPU runtime checks; `nvcc` and a compatible C++ compiler are required only
by the BAGEL profile.
Python discovery probes PATH, the running interpreter, and Conda installations
under the user's home or `/opt/conda`. CUDA discovery probes PATH, `CUDA_HOME`,
and `/usr/local/cuda*`. Selected executables are printed and actually executed;
no Conda activation or global PATH edits are required.

Use `bash deploy.sh --dry-run` to inspect the complete plan without downloads
or writes. `--model-root`, `--env-prefix`, `--python310`, `--python312`,
`--index-url`, and `--endpoint` override locations/interpreters/mirrors. The
Hugging Face endpoint does not redirect official Git/URL assets. If changing
the model/environment roots, use `--config` and the generated launcher; it
records the actual interpreters and model root, so `VIDUEU_PYTHON` is not
required.

Successful environments receive a deployment receipt; reruns check their
dependency recipe, runtime imports, and `pip check` before reusing them. Interrupted environments created
by this installer have a separate setup receipt and resume completed command
stages when the exact dependency recipe matches. Updating only adapter/runtime
checks reuses installed packages and repeats validation, without rebuilding a
venv or downloading CUDA wheels again. Setup can also recognize the exact older
receipt format from before runtime checks were added; it verifies the old
commands and requirement hashes and upgrades the receipt as validation succeeds.
`setup --dry-run` reports `resume_reason` and `pending_commands`, so the remaining
work is visible before execution. Unmanaged or changed-dependency environments
are refused; no environments are deleted. Rerun the same deploy command to resume.
For direct low-level environment setup without model download, use
`bash setup.sh --profile pipeline --resume`; most users should use
`bash deploy.sh` so installation, asset download, verification, and runtime
checks stay in one resumable command.
Caches and build temporaries default to `<checkout>/.deploy`; `--work-dir`
relocates them. A free-space reserve of 20 GiB is checked on each target
filesystem (`--min-free-gib` overrides it). This reserve does not estimate the
total environment/model size; allow additional space for all weights and builds.
Model downloads use the existing SHA-256 receipts and refuse changed files.
HF files are downloaded serially with verified HTTP Range resume and a
600-second per-request timeout by default. This bypasses the Hub snapshot
cache layer, which is fragile for large LFS/Xet files behind a proxy. Override
the timeout with `--download-timeout` on `scripts/download_models.py`, or with
`VIDUEU_DOWNLOAD_TIMEOUT`. Partial files stay beside their target and are never
published as complete assets; size and SHA-256/Git blob checks must pass before
the receipt is marked complete.
URL transfers retry transient connection failures and resume partial bytes when
the server confirms the requested range; the full SHA-256 is checked before
publishing the file. Servers ignoring Range restart the managed partial file.
URL downloads retain hostname/certificate validation. If Python's default CA
store is empty, they load the trusted certifi bundle already supplied by
requests. Explicit `SSL_CERT_FILE`/`SSL_CERT_DIR` settings are respected.
Benchmark datasets, fixed subsets, and paid judge access remain user inputs;
their paths/commands are documented in `configs/iteration.example.yaml`.

Run from the source checkout. `setup.sh` delegates to `scripts/setup_env.py`;
`--dry-run` prints the exact interpreter/pip commands without creating a venv.
`--python /path/to/python3.10` selects a base interpreter; `--index-url` selects
a package mirror. PyTorch uses its explicit matching CUDA wheel index.
`--wheelhouse /path/to/wheels` selects a fully offline installation, including
bootstrap and editable-build dependencies. The directory must contain wheels
for the target Linux/Python version. Private/global pip indexes are not inherited.

| Profile | Python | Purpose |
| --- | --- | --- |
| pipeline | 3.10 | Torch 2.6/cu124, Transformers 4.57, CPU OCR and local tools |
| dev | 3.10 | pytest, package building and lightweight release tests |
| vllm | 3.12 | vLLM 0.19 with its independently resolved Torch requirements |
| omniparser | 3.12 | Torch 2.6 and Transformers 4.47, isolated bridge |
| bagel | 3.10 | Customized BAGEL, Torch 2.5.1, Transformers 4.49 and flash-attn |

Do not combine these requirements into one environment. A CUDA toolkit and
compiler may be needed to build BAGEL's flash-attn. Setup does not edit conda
environments, CUDA drivers, system shell files, proxies, or crontabs. If an
installation fails, the partial new environment is left intact for inspection
and matching-recipe resume. No automatic removal occurs.

Setup checks runtime imports before the final `pip check`. The selected profile's
runtime check runs after installation; the OmniParser inference check runs only
when the `omniparser` profile is selected. Package metadata alone is
insufficient for that optional bridge.
Requirements are tested-family constraints;
they are not hash-locked dependency sets for every operating system. CPU Paddle
is the default. Optional legacy MUSIQ/BlurMap dependencies are not installed
by the main profile; the public paper template disables them.

CPU OCR is intentional, including inside the isolated OmniParser process. Its
YOLO/Florence models may independently use CUDA. Requesting `cuda:N` for Paddle
with the default CPU wheel now fails explicitly; it never silently downgrades.
`doctor --check-gpu` reports the configured OCR backend as well as Torch CUDA.
To use GPU Paddle, prepare a separate compatible GPU Paddle environment and
validate its CUDA runtime before changing the OCR device; installing CPU and
GPU Paddle wheels together is unsupported. The default one-command deployment
does not make that replacement or alter an existing production environment.

OmniParser uses the same downloaded PP-OCRv5 assets as the pipeline. Its
compatibility loader removes the pinned upstream demo's eager EasyOCR/PaddleOCR
2.x initialization and unused cloud SDK imports in memory; downloaded source
and weight receipts stay unchanged. It checks the expected source layout and
fails if the pinned upstream structure changes. No OpenAI key, EasyOCR model
download, or network connection is needed for the default bridge. Optional
legacy `use_paddleocr: false` requires separately cached EasyOCR weights;
automatic downloads are disabled.

Repeat the actual adapter checks after deployment:

```bash
./environments/pipeline/bin/python scripts/check_runtime.py \
  --profile pipeline --model-root ./model --inference
CUDA_VISIBLE_DEVICES=5 ./environments/omniparser/bin/python scripts/check_runtime.py \
  --profile omniparser --model-root ./model --inference --device cuda:0
```

The second command uses physical GPU5 for YOLO/Florence and CPU for OCR. Choose
your own authorized physical card. Checks use synthetic fixtures and no paid
API; production videos and outputs are not read. Changes to installed dependency
recipes still require a separate environment prefix. A runtime-check-only update
does not require reinstalling the environments or models.

BAGEL's default image/JSONL training does not install Decord. Encoded-video
datasets load it on demand and require a separately prepared, platform-compatible
Decord build. The upstream 0.6.0 wheel carries a Python 3.6 tag and fails modern
`pip check` on Python 3.10; it is not relabeled or ignored by this installer.

Model manifest entries record fixed revisions, official origins, required files
and licensing links. Downloads require `--only`. Fetch code before weights when
both share a directory (`raft_code raft_weights`, `omni_code omni_weights`).
Do not manually mix managed/unmanaged assets in the same target directory.

Receipts under `<model-root>/.vidueu-assets/` bind each asset definition to its
files' SHA-256. `--offline`/`--verify` never contacts a provider and fails on
missing or changed bytes. `doctor --check-assets` checks paths, basic structure
and weight shards, not a replacement for receipt verification. Snapshot code
is downloaded without executing it. Large model downloads and model inference
are separate actions. Keep enough disk space for checkpoints and video intermediates.
