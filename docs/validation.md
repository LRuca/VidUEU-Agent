# CPU/offline acceptance (2026-10-21)

## Release fixes checked on 2026-11-03

The current full offline regression passed 523 tests, with 11 platform-dependent
skips. Doctor endpoint regressions use a real loopback HTTP
server: unauthenticated local VLM, server-required authentication, key-file
authentication, served-model mismatch, and explicit/direct proxy routing.
Project licensing is Apache-2.0 for original code; component declarations and
the pinned OmniParser detector/caption licenses were checked against upstream.
Private provider acceptance configs are outside the public tree and rejected
by the release check.

Fresh real-video attempts with a remote vision API reached T1/T2, but all
selected frames were rejected by the existing strict quality gate. These are
valid filtering results, not a successful T1-T4 output/export acceptance.
Local Qwen/vLLM acceptance is pending available GPU memory: all eight A100s
were occupied by unrelated jobs. No third-party jobs were stopped. The
prepared launcher refused to start with insufficient free memory. A passing
default-quality full pipeline is not claimed by this release note.

### Temporary gate-bypass acceptance

At the user's explicit request, a separate private one-video configuration set
`overrides.tasks.T2_gate_b.enable_model_stage: false`. The fast quality profile,
real tool inference, T3, metadata admission and T4 remained active. Public
algorithm defaults and T3 code were unchanged. The run used real 1920x1080
OpenVidHD frames, a remote vision API in OpenAI wire mode, the existing text
provider, and CPU tools with CUDA hidden. It completed with exit code 0.

The writer produced 5 instruction rows (4 understanding and 1 unify). Portable
export dry-run and actual export passed: 4 understanding, 1 unify, 1 metadata
record and 9 distinct image files. Hash and full-image decoding verification
passed both in the original package and after copying it to a different
directory. Source JSONL hashes were unchanged. There were no edit rows, so
real edit generation is not covered by this test.

This is a software-path and portable-export check with a disabled VLM quality
filter, not a default-quality data acceptance or a local Qwen/vLLM test. The
private acceptance manifest explicitly records `test_only: true` and
`publish_ready: false`. No private provider configs, credentials, production
data or acceptance media are part of the public source tree.

## Historical offline acceptance

The public candidate was checked without GPU inference or paid API calls.
Production jobs, environments, datasets and temp/log/image stores were untouched.
Independent post-generation review remains outside this source tree.

| Check | Result |
| --- | --- |
| Full offline regression in the existing Linux pipeline interpreter, CUDA mask empty | 472 passed; one existing pytest collection warning |
| Lightweight public CLI and release tests in a fresh dev environment without Torch | 47 passed |
| Windows full offline regression before the final two added cases | 460 passed, 10 Linux/symlink-dependent skips |
| Windows focused final public CLI/T4 checks | 25 passed, 2 platform-dependent skips |
| Source closure, syntax, credential-pattern and private-workflow exclusion | Passed; 293 source files checked |
| Wheel build and installation in another new Python3.10 venv | Passed; dependencies installed from an offline wheelhouse |
| Installed wheel outside the source checkout, without Torch | CPU smoke passed; pip check passed |
| Shell entry points | bash -n passed |

Smoke sets CUDA_VISIBLE_DEVICES to empty, rejects model-library imports and
socket creation, encodes a one-second fixture using CPU FFmpeg, and uses a fixed
planner fixture in place of model/tool inference. It runs the actual entrypoint and
JSONL writer twice, checks completed-video resume does not append duplicates,
exports understanding/edit/unify/metadata plus two images, and verifies a copied
package in a different location. Source video bytes are unchanged. The report
records simulated=true, zero model/API calls and zero optimizer steps.

Additional regressions cover small-config compilation, CLI override ordering,
dotenv precedence/cycles, secret redaction, explicit GPU mappings, child-only
proxy modes, owned process/tmux lifecycle, selective tools, export dry-run,
missing/corrupt/out-of-root images, symlink escape, duplicate keys, producer
provenance, nested/typed unified image paths, package tampering and write-failure
propagation. Export images must verify and fully decode; no rows are silently
dropped to make a package pass.

The fresh environment was created by the dev setup profile with offline wheels.
A separately built wheel was then installed in another new venv and exercised
from outside the checkout. No full pipeline/vLLM/OmniParser/BAGEL environment
matrix, real model inference, paid request, optimizer or genuine benchmark score
is claimed by this acceptance.

Training/evaluation recipes were validated offline. They report missing
deployment adapters, weights, inputs and preprocessing/accounting choices;
they are not a claim that all paper experiments execute out of the box.
The default evaluation contract is one generation and one judge pass.

Original code is now licensed under Apache-2.0. Third-party code/model/media
rights remain separate; see NOTICE.md. Citation metadata awaits the arXiv release.
