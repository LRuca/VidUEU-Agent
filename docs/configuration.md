# Configuration

Create a small config with `vidueu init --output config.local.yaml`.
Common fields:

```yaml
version: 1
preset: paper
input: ./videos
output: ./outputs/demo
model_root: ./model
capabilities: [understanding, editing, unified]
tools: [ocr, asr, yolo]
hardware:
  profile: external_vlm_cpu_tools
  gpus: []
llm:
  base_url: https://api.deepseek.com
  model: deepseek-flash
  api_key_env: DEEPSEEK_API_KEY
vlm:
  base_url: http://127.0.0.1:8110/v1
  model: Qwen3.5-9B-local
budget:
  max_api_attempts: 100
network:
  proxy_mode: direct
```

The preset uses paper_v2 generation prompts; the deployment text model is
configurable, while the paper reports DeepSeek-V3. Config validation does not
validate provider availability or a key.

Precedence is packaged defaults, public fields, advanced `overrides`, explicit
CLI input/output. Lists replace lists. Variables are substituted inside parsed
YAML scalars, not YAML syntax. CLI values replace unresolved input/output first.
Small-config paths use the file's directory; CLI paths use the caller's cwd.
Output must be outside the source-video directory. `python` defaults to the
running interpreter and also accepts executable names. Optional OmniParser uses
a separate `tool_python` environment.

Adjacent .env is read as data, never executed. Process environment wins.
NAME=value, optional export, quotes and variable references are supported.
Missing variables, cycles, duplicate YAML keys and shell substitution fail.
Use ignored key files or environment variables. `config show` always redacts
keys and proxy URLs.

The text-client attempt budget persists at <output>/deepseek_budget.sqlite3.
Retries/failed requests consume reservations. Restarts do not reset it and
changing an existing ledger's limit is refused. Doctor reads an existing ledger
without creating/modifying it. This budget covers the text client; external VLM
billing is separate. New work needs an explicitly budgeted new output.
Call records live at <output>/api_calls.jsonl.

Managed child networking: direct clears six HTTP/SOCKS proxy variables, inherit
keeps launch-shell proxies, explicit uses network.proxy_url. Use a placeholder
such as ${PIPELINE_PROXY_URL} from .env; the resolved URL is not written to run
state. No global network configuration changes. Foreground legacy execution
inherits its shell environment. Doctor's VLM model probe uses direct networking.

hardware.gpus lists physical indices and defaults to empty. A tool's cuda:1 is
the second listed GPU. Default tools use CPU. A shared VLM is never implicitly
started/stopped; optional local vLLM has its own managed lifecycle. Dashboard and
managed vLLM bind to loopback.

Advanced overrides expose fields such as models.yolo_device, optical_flow and
tasks.T4. Inspect the resolved contract with config show. The large
configs/paper.yaml remains a legacy compatibility template requiring its named
variables; new users should use small config.

Managed run refuses force_overwrite. Resume skips completed videos; incomplete
videos may rerun and replace their matching task rows through the existing resume
logic. Use a fresh output to isolate an experiment. Start/stop never cleans
source videos, images, temp or logs. See [outputs](outputs.md).
## Vision API authentication and protocol

The default local Qwen/vLLM endpoint may run without authentication. Doctor
queries `/v1/models` without requiring an artificial key and reports the real
HTTP or model-name error. When using a protected endpoint, set `vlm.api_key_env`
or `vlm.api_key_file`. Doctor uses the same configured credential and networking
mode as the managed run. `direct` bypasses shell proxies; `inherit` uses them;
`explicit` uses `network.proxy_url`. No generation request is made by doctor.

Keep the default `wire_profile: vllm` for local vLLM. For a remote reasoning
vision API, set `vlm.wire_profile: openai` and, if supported by the provider,
`vlm.reasoning_effort: low`. This sends `max_completion_tokens` and the chosen
reasoning effort without vLLM-only sampling/template parameters. Explicit
LLM reasoning effort no longer inherits the default `thinking: disabled`.

For a small local GPU smoke, the managed vLLM launcher accepts explicit
`--max-model-len 16384 --max-num-seqs 1 --enforce-eager`. Context and sequence
limits must be positive. These options affect the owned vLLM service only;
ordinary launches retain vLLM defaults when the options are omitted.

