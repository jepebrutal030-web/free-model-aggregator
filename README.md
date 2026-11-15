# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/jepebrutal030-web/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/jepebrutal030-web/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server (`main.ts`, Node 22, stdlib only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and server-sent events —
over the ten free providers the trial gives it, answering every request with
the best free model that meets the request's requirements.

## How it works

- Reads the brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the sealed
  quality table and the workload manifest.
- Probes every provider's catalog route in the provider's own wire and builds
  one catalog (`node main.ts --list` prints it, byte for byte as registered).
- Registers port + catalog with the env, serves the client's 20 requests one
  at a time, polls `GET /work` and exits 0 when the env says done.
- Routing: free models meeting the requirements, ranked by the quality
  table's `rating_milli`, best first. A 402/429/5xx is retried once on the
  same model (a failure on the provider's own schedule is transient), then the
  request moves to the next candidate. A provider that fails twice in a row is
  treated as spent for the rest of the trial. Only one provider call may
  succeed per request, so a 200 is always the answer.
- Wires: each provider is called in its own wire (openai, anthropic or
  gemini) and the answer is translated back into the client's wire, with the
  provider's text whole, its usage counts as that wire reports them, and its
  stop reason mapped. A JSON-mode OpenAI request only goes to a wire that can
  carry JSON mode (openai or gemini). Streamed OpenAI calls carry
  `stream_options.include_usage` so the usage chunk arrives.
- Determinism: every choice derives from the brief, the requests and the
  providers' answers. No clock, no unseeded randomness, no timing-dependent
  retry cap (a fixed 30-attempt cap per request).

## Files

- `main.ts` — the server: brief, catalog, routing, failover, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe.
- `lib/wire.ts` — client wires and the three upstream wires, request and
  answer translation (JSON and SSE), model lists.
- `run-trial.sh` — local trial runner (env + bot + scorer) for practice trials.

## Local results (practice trials, env + scorer from the kit)

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |

Quality 99 is the ceiling on these trials: the best model for most requests
(OpenRouter `z-ai/glm-5.3-flash:free`, 1475) has a per-trial budget of 10–12
calls, so the remaining requests must fall back to the 1470 models. Every
request is served validly; the state export is byte-identical across repeated
runs of the same trial.
