# Model selection

Every model is declared once in `mewm/llm/registry.py`; the CLI, the config, the
transport dispatch and the connectivity check all read that table.

```bash
python -m mewm.cli.main models          # what exists and whether it is reachable
python -m mewm.cli.main test-models     # one text round-trip each
python -m mewm.cli.main test-models --vision   # the check that matters for P/A/R
```

## Choosing per role

```bash
python -m mewm.cli.main run --dataset casme_sq \
  --reasoning-model claude-sonnet-5 --reasoning-effort max \
  --critic-model grok-4.5
```

or in `configs/mewm_agent.yaml`:

```yaml
llm:
  reasoning_model: claude-sonnet-5
  perception_model: gemini-3-pro
  structure_model: gemini-3-pro
  critic_model: gpt-5.6-sol
  reasoning_effort: xhigh
```

**`critic_model` must come from a different provider than `reasoning_model`.** The check
is at provider level, not id level: two tiers of one family share training and therefore
share blind spots, and a critic with the reasoner's blind spots is not an independent
check. `mewm doctor` reports this, and `run` warns.

## Hosted models — verified 2025-08-24

| Model | Provider | Transport | Efforts | Text | Vision |
| --- | --- | --- | --- | --- | --- |
| `claude-sonnet-5` | anthropic | Anthropic Messages | high, xhigh, max | ok | ok |
| `gpt-5.6-sol` | openai | OpenAI responses/chat | high, xhigh | ok | ok |
| `grok-4.5` | grok | Anthropic Messages | — | ok | ok |
| `gemini-3-pro` | gemini | native `generateContent` | — | ok | ok |
| `gemini-3.1-pro` | gemini | native `generateContent` | — | ok | ok |
| `deepseek-v4-flash-vision-exp` | deepseek | Anthropic Messages | — | ok | intermittent |

Three things were discovered by probing rather than assumed, and each would otherwise be
a silent failure:

- **`deepseekV4-Flash-Vision-Exp` is not a served id.** The endpoint accepts only
  `deepseek-v4-pro`, `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp`. The
  mixed-case spelling is aliased onto the served id. Related: asking that relay for
  `deepseek-chat` returns `deepseek-v4-flash` *without an error*, which is why the
  client verifies the served model id and refuses a mismatch.
- **`gemini-3-pro-preview` is listed by the relay but 500s on every call.** It is
  aliased onto `gemini-3-pro`, which works. The gemini relay is also native-Gemini
  shaped — both the OpenAI and Anthropic paths 404 against it.
- **The newcli relays sit behind Cloudflare** and answer an unrecognised user agent with
  403 / error 1010. Those providers send an explicit `User-Agent`.

**DeepSeek vision is intermittent.** Text is reliable. Single image requests returned an
empty completion most of the time in measurement (2/6 and 0/18 in two bursts against the
raw endpoint, so it is not a client bug). A `min_retries=6` floor usually recovers it at
roughly 13 s per call against ~5 s elsewhere. Fine for text-only R/C roles; prefer
claude / grok / gemini / gpt for any phase that reads frames.

## Open-weight models

| Model | Repo | VRAM | Vision |
| --- | --- | --- | --- |
| `Qwen3-VL-8B` | `Qwen/Qwen3-VL-8B-Instruct` | ~18 GB | yes |
| `Qwen3.6-VL-27B` | `Qwen/Qwen3.6-VL-27B-Instruct` | ~58 GB | yes |
| `Qwen3.8-27B` | `Qwen/Qwen3.8-27B-Instruct` | ~58 GB | no |
| `Gemma-4-31B` | `google/gemma-4-31b-it` | ~66 GB | yes |

Weights download on first use. Check the box first, or pre-fetch:

```bash
python -m mewm.cli.main local-models                       # disk, VRAM, what is cached
python -m mewm.cli.main local-models --download Qwen3-VL-8B
```

`Qwen3.8-27B` is text-only, so it can serve R or C but not the frame-reading phases; the
client drops images for a text-only model and says so rather than failing obscurely.

There is deliberately **no silent fallback** to a hosted model when weights are missing.
A run reporting `Qwen3-VL-8B` must have used it, or any comparison built on it is
meaningless — so a missing model raises `WeightsUnavailableError` carrying the
instructions below.

### Manual download

Automatic download fails for four usual reasons; the error names the one it hit.

```bash
# 1. huggingface-cli (resumable, preferred)
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./weights/Qwen3-VL-8B

# behind a mirror (common in mainland China)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./weights/Qwen3-VL-8B

# 2. ModelScope (Qwen weights are mirrored there)
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ./weights/Qwen3-VL-8B

# 3. git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ./weights/Qwen3-VL-8B
```

Then point the framework at the directory:

```bash
export MEWM_LOCAL_WEIGHTS_QWEN3_VL_8B=/absolute/path/to/weights/Qwen3-VL-8B
```

The variable is `MEWM_LOCAL_WEIGHTS_` plus the model id upper-cased with `-` and `.`
replaced by `_` (`Qwen3.6-VL-27B` → `MEWM_LOCAL_WEIGHTS_QWEN3_6_VL_27B`).

Causes worth checking, in order: a **gated repo** (accept the licence, then
`huggingface-cli login`); **no network or a proxy** (set `HF_ENDPOINT`); **disk** (budget
roughly the VRAM figure again on disk); **transformers too old** for the architecture
(`pip install -U transformers`).

## Adding a model

One entry in `_MODEL_LIST`:

```python
ModelSpec(
    "my-model-id", "provider-name", "Display Name",
    aliases=("other-spelling",), efforts=("high", "xhigh"), default_effort="high",
    vision=True, context=128_000, notes="anything a user would otherwise trip over",
)
```

For a new provider add a `ProviderSpec` naming its transport and environment variables.
Then confirm it: `python -m mewm.cli.main test-models --vision --models my-model-id`.

## Explicit backends and the audit manifest (修改方案 §5)

Every role declares `hosted` or `local` (yaml `llm.*_backend`, or CLI
`--backend` / `--reasoning-backend` ...). The registry only validates:
`hosted` + an open-weight id, or `local` + a hosted id, is a `BackendMismatch`
error at startup (exit code 2), never a silent re-route.

```bash
python -m mewm.cli.main run --dataset casme_sq --backend hosted      # pure API
python -m mewm.cli.main run --dataset casme_sq --backend local \
       --critic-backend hosted                                       # mixed
```

Missing local weights are a hard error unless `--fallback-to-api` is passed; each
fallback is WARNING-logged and flagged in the per-video `backend_manifest.json`
(role, model, backend, route, latency per call), so mixed runs stay auditable.
Open-weight entries carry `quantization: 4bit` (nf4 double quantisation,
bitsandbytes >= 0.45 on RTX 50-series); weights prefetch resumably with
`python -m mewm.cli.main local-models --prefetch Qwen3-VL-8B`.
