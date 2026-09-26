# LLM-ROUTER

Routes a task to the cheapest model that can answer it.

The router runs in two steps:

1. A cheap **classifier** decides between two routes — `fast` and `powerful` —
   using the criteria in `router.py` (`_CRITERIA`).
2. An [Agno](https://docs.agno.com) agent answers the task with the model of
   that route, over that model's own OpenAI-compatible endpoint.

Every model has its own base URL and its own API key, so a route can sit behind
a different gateway or credential than the others. LLM-ROUTER speaks nothing
but `POST {base_url}/chat/completions` — unless you switch the classifier to
`laya`, which is the single documented exception.

## Install

```bash
uv sync
```

## Configure

```bash
cp .env.example .env
```

| Variable | Required | Endpoint | Meaning |
| --- | --- | --- | --- |
| `OPENAI_BASE_URL` | yes | classifier | Base URL of the classifier endpoint |
| `OPENAI_API_KEY` | yes | classifier | Bearer token for the classifier |
| `ROUTER_MODEL` | yes | classifier | Model that performs the classification |
| `FAST_BASE_URL` | yes | `fast` | Base URL of the `fast` route |
| `FAST_API_KEY` | yes | `fast` | Bearer token for the `fast` route |
| `FAST_MODEL` | yes | `fast` | Model for the `fast` route |
| `POWERFUL_BASE_URL` | yes | `powerful` | Base URL of the `powerful` route |
| `POWERFUL_API_KEY` | yes | `powerful` | Bearer token for the `powerful` route |
| `POWERFUL_MODEL` | yes | `powerful` | Model for the `powerful` route |
| `CLASSIFIER` | no | — | `openai` (default) or `laya` |
| `LAYA_MODEL` | no | — | Laya checkpoint: `multilingual` (default), `english`, `typed-decisions` |
| `LAYA_MAX_LEN` | no | — | Tokens the Laya classifier reads: `1024` (default), up to `8192` |
| `LAYA_PRELOAD` | no | — | Load Laya checkpoints eagerly (`1`/`true`/`yes`/`on`) |

In code the same shape is a `Settings` object with one `Endpoint` per model:

```python
Endpoint(base_url="https://fast-gateway.internal/v1", api_key="...", model="luna")
settings.endpoint_for("fast")  # -> the fast endpoint
```

Missing or blank variables raise `ConfigError` at startup instead of failing
later against the endpoint. There is no fallback to a shared base URL or key: a
route that is not configured does not silently borrow another route's
credentials.

## Run

```bash
uv run python router.py
```

```python
from router import ModelRouter, Settings

router = ModelRouter(Settings.from_env())
result = router.run("Extract the city name from: I landed in Lisbon last night.")
# {"route": "fast", "model": "luna", "content": "Lisbon",
#  "confidence": 0.98, "source": "openai"}
# the request went to FAST_BASE_URL with FAST_API_KEY
```

`route_and_run(task)` is the one-liner that builds the router from the
environment and runs a single task.

## Classifiers

### `openai` (default)

Asks the endpoint for a **single token** (`max_tokens=1`, `logprobs=true`) and
compares the logprobs of the labels `A` and `B`. Nothing is generated, so there
is no answer to parse. If the endpoint rejects the `logprobs` parameter, the
router retries once without it and reads the label from the returned text.

`confidence` is the softmax over the two label logprobs. It is not a calibrated
probability, and when the endpoint returns no logprobs it is `None`.

### `laya`

A local [Laya](https://huggingface.co/convaiinnovations/laya) decision model
answers the same question as a typed `choice` and returns a probability for
each option. **This is a local model runtime** and therefore the one deliberate
exception to the OpenAI-compatible-only rule; the answering models stay
OpenAI-compatible either way.

```bash
uv sync --extra laya
```

```bash
CLASSIFIER=laya uv run python router.py
```

The backend uses **`convaiinnovations/laya-multilingual`** by default, covering
100+ languages, and reads 1024 tokens. `LAYA_MODEL=english` or
`LAYA_MODEL=typed-decisions` switch the checkpoint; `LAYA_MAX_LEN=8192` lifts
the input limit for long documents.

The checkpoint is **not** loaded unless `CLASSIFIER=laya`. It is loaded when the
`ModelRouter` is constructed, not on the first `run()`; set `LAYA_PRELOAD=true`
to load all checkpoints into memory up front instead of lazily on first use,
trading memory for first-request latency.

### Both

Both backends fail closed: a task the classifier cannot decide goes to the
`fast` route, the cheaper model.

## Known limits

- The `laya` classifier is **unproven for this task**. On Laya's own
  typed-decisions benchmark the shipped checkpoints score near chance without
  fine-tuning (0.342 multilingual against 0.318 random), and the probabilities
  are overconfident until you fit a temperature on your own data. Treat the
  `laya` backend as an experiment until you have measured it on your own tasks;
  the `openai` backend is the default for that reason.
- `confidence` is uncalibrated in both backends and says nothing about whether
  the chosen *route* is correct.
- No request timeout is configured; a hung endpoint blocks the caller.
- The router is synchronous only, and an endpoint error on the classifier is not
  retried — it surfaces as `APIStatusError`.

## Development

```bash
uv run bash scripts/gauntlet.sh
```

Reruns every check: lint, format, types, tests with coverage in randomized
order, mutation testing (`scripts/mutate.py`) and a real end-to-end run against
three separate local OpenAI-compatible servers
(`scripts/fake_openai_server.py`), one per endpoint, so a route that ignores
its own configuration cannot pass unnoticed.
