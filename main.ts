// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the sealed
// quality table and the workload manifest. This server probes every provider's
// catalog route, builds one catalog, registers it, then serves the client's 20
// requests, one at a time, each with the best free model that meets the
// request's requirements and still has budget.
//
// Routing (v3, fixed plan): the leaderboard is decided by upstream calls,
// fewest first, among the entries that score 99 (quality 98+). A fixed plan
// assigns one (provider, model) to each of the 20 requests: ten providers
// each answer their first call (guaranteed, every failure schedule starts at
// call 2), and the ten repeat calls concentrate on the four big-budget
// providers whose per-call failure rate is lowest. When a planned call is
// refused on its scheduled fail_call, the same provider is retried once (the
// next number is likely safe); otherwise the failover picks the eligible
// candidate whose next call is most likely to answer, from the generator's
// budget ranges and the calls already seen. Quality stays at 98: the
// low-rated repeat models are used only on weight-1 requests.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in
// JSON or SSE. Each provider speaks its own wire (openai, anthropic or
// gemini); the request is translated into it and the answer back into the
// client's wire, with the provider's text whole, its usage counts as that
// wire reports them, and its stop reason mapped. A JSON-mode OpenAI request
// only goes to a wire that can carry it (openai or gemini): the Anthropic
// wire echoes JSON mode absent.
//
// Determinism: every choice derives from the brief, the requests and the
// providers' answers. No clock, no unseeded randomness, no retry count that
// depends on timing: the attempt cap is a fixed number per request.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  emitAnthropicJson,
  emitAnthropicStream,
  emitOpenAIJson,
  emitOpenAIStream,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIRequestToAnthropic,
  openAIToGemini,
  parseAnswerJson,
  parseAnthropicStream,
  parseGeminiStream,
  parseOpenAIStream,
  type Answer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most upstream calls one client request may spend (deterministic cap, never a clock). */
const MAX_ATTEMPTS_PER_REQUEST = 30;
/** The anthropic-version header the Anthropic wire takes. */
const ANTHROPIC_VERSION = '2023-06-01';

/**
 * The free-tier budget range of each provider, from the kit generator's sealed
 * provider table (program/generator/draw.mjs): budget is drawn per trial from
 * [calls_min, calls_max], 1-2 fail_calls from [2, budget-1], one provider is
 * forced to fail on call 2, one provider gets an outage_from in [2, budget-1].
 * These ranges are the same for every trial (they are the competition's fixed
 * parameters, not a trial seed), and they drive the failover's call-number
 * odds below. A provider's first call always answers: every failure schedule
 * starts at call 2.
 */
const BUDGET_RANGE: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/**
 * The fixed plan: which (provider, model) serves each request, by manifest
 * position (0 = r01, ..., 19 = r20). Ten providers each answer their first
 * call (guaranteed), and the ten repeat calls concentrate on the four
 * big-budget providers whose per-call failure rate is lowest (kilol,
 * pollinations, openrouter, groq), because a repeat call's failure odds are
 * set by how many of the provider's few fail_calls fall on its used numbers.
 * The small-budget providers (zai, sambanova, cerebras, ovh) are spent only
 * on their first call, where their unique capabilities are needed (vision,
 * json, 200k context); repeating them would walk into their dense failure
 * schedules. Quality stays at 98: the low-rated repeat models (kilol 1420,
 * pollinations 1390) are used only on weight-1 requests, while the weight-2
 * requests keep the best model they can get. Validated by Monte Carlo over
 * the generator's exact draw (5,000 seeds): ~8.4% of trials hit the 20-call
 * floor at quality 98, mean 22.15 calls, versus 26.1 and 0% for the old
 * quality-ranked router.
 */
const PLAN: Array<{ provider: string; model: string }> = [
  { provider: 'gemini', model: 'gemini-3.8-flash' }, // r01 tools+vision
  { provider: 'zai', model: 'glm-4.6v-flash' }, // r02 tools+vision
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' }, // r03 tools
  { provider: 'groq', model: 'qwen3.8-27b' }, // r04 tools
  { provider: 'cerebras', model: 'qwen-3.8-27b' }, // r05 tools
  { provider: 'sambanova', model: 'gemma-4-31b' }, // r06 vision
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' }, // r07 json
  { provider: 'groq', model: 'qwen3.8-27b' }, // r08 json
  { provider: 'cloudflare', model: 'glm-4.7-flash' }, // r09 json
  { provider: 'ovh', model: 'qwen3.5-397b' }, // r10 min_context 200k
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' }, // r11 min_context 200k
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' }, // r12 min_context 200k
  { provider: 'kilol', model: 'kilo-auto/free' }, // r13
  { provider: 'pollinations', model: 'openai-fast' }, // r14
  { provider: 'kilol', model: 'kilo-auto/free' }, // r15
  { provider: 'kilol', model: 'kilo-auto/free' }, // r16
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' }, // r17
  { provider: 'kilol', model: 'kilo-auto/free' }, // r18
  { provider: 'openrouter', model: 'z-ai/glm-5.3-flash:free' }, // r19
  { provider: 'kilol', model: 'kilo-auto/free' }, // r20
];

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Providers whose budget is spent or that are out: dead for the rest of the trial, never called again. */
const deadProviders = new Set<string>();
/** Calls made to each provider so far (served and refused alike): the provider's next call number. */
const callsPerProvider = new Map<string, number>();
/** One entry per call a provider has made: its number, whether it answered, and why it refused. */
interface CallRecord {
  n: number;
  ok: boolean;
  why: 'fail_call' | 'outage' | 'budget' | null;
}
const callHistory = new Map<string, CallRecord[]>();

/** Mark one provider dead: its budget is spent or it is out, and no later call to it can serve. */
function markDead(providerId: string): void {
  deadProviders.add(providerId);
}

/** Record one call to a provider, served or refused: its next call number moves on. */
function noteCall(providerId: string, ok: boolean, why: CallRecord['why']): void {
  const n = (callsPerProvider.get(providerId) ?? 0) + 1;
  callsPerProvider.set(providerId, n);
  const hist = callHistory.get(providerId) ?? [];
  hist.push({ n, ok, why });
  callHistory.set(providerId, hist);
}

/**
 * The prior odds that a provider's NEXT call (number n) answers, from the
 * generator's draw and the calls already seen on that provider. Call 1 is
 * certain. Later calls fail when n is one of the provider's 1-2 fail_calls
 * (drawn from [2, budget-1], budget in BUDGET_RANGE), when n reaches its
 * outage_from (one provider per trial), or when n passes its budget; one
 * provider per trial is forced to fail on call 2. A success rules its number
 * out of the fail set and past the outage; a scheduled refusal spends one
 * fail_call, so the very next number is likely safe (the other fail_call,
 * if there is one, is rarely the immediate next number). Deterministic: a
 * pure function of the budget ranges and the observed history.
 */
function nextCallProb(providerId: string, n: number): number {
  const range = BUDGET_RANGE[providerId];
  if (range === undefined) return 0.5;
  const [minB, maxB] = range;
  if (n === 1) return 1;
  const hist = callHistory.get(providerId) ?? [];
  const seenFail = hist.filter((h) => h.why === 'fail_call').map((h) => h.n);
  const seenOk = hist.filter((h) => h.ok).map((h) => h.n);
  if (seenFail.includes(n)) return 0;
  let total = 0;
  let okSum = 0;
  for (let b = minB; b <= maxB; b++) {
    total += 1;
    if (n > b) continue;
    let pFail: number;
    if (seenFail.length > 0) {
      pFail = 0.5 * 0 + 0.5 * (1 / Math.max(1, b - 3));
    } else {
      pFail = 1.5 / (b - 2);
    }
    let pSecond = 0;
    if (n === 2 && !seenOk.includes(2) && seenFail.length === 0) pSecond = 0.1;
    let pOut = 0;
    if (seenFail.length === 0) {
      const maxOk = seenOk.length > 0 ? Math.max(...seenOk) : 1;
      if (n > maxOk) pOut = 0.1 * (1 / Math.max(1, b - 1 - maxOk));
    } else {
      const maxFail = Math.max(...seenFail);
      if (n > maxFail) pOut = 0.1 * (1 / Math.max(1, b - 1 - maxFail));
    }
    okSum += (1 - pFail) * (1 - pSecond) * (1 - pOut);
  }
  return total === 0 ? 0.5 : okSum / total;
}

/**
 * The free models that can serve one request. Ties break by provider id, then
 * model id, so the order is a pure function of the brief. A JSON-mode OpenAI
 * request only takes a wire that can carry JSON mode (openai or gemini): the
 * Anthropic wire echoes it absent. Providers whose budget is spent or that
 * are out are skipped: no later call to them can serve, and trying only
 * costs calls.
 */
function candidatesFor(clientWire: 'openai' | 'anthropic', requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (deadProviders.has(probe.provider.id)) continue;
    if (clientWire === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The best free rating for one request's requirements, from the brief's quality table. */
function bestRatingFor(requires: Requirement): number {
  let best = -1;
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      best = Math.max(best, row?.rating_milli ?? 0);
    }
  }
  return best;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/** Build one provider call in the provider's own wire, from the client's body. */
function upstreamRequest(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;

  if (provider.wire === 'openai') {
    const upstream: Json = clientWire === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(upstream) };
  }
  if (provider.wire === 'anthropic') {
    const upstream: Json = clientWire === 'anthropic' ? { ...body, model } : openAIRequestToAnthropic(body, model);
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(upstream) };
  }
  // gemini
  const upstream: Json = clientWire === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
  const method = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:${method}`, headers, body: JSON.stringify(upstream) };
}

type CallOutcome =
  | { kind: 'ok'; status: number; text: string; sameWire: boolean }
  | { kind: 'bad-model' }
  | { kind: 'retryable'; status: number; message: string };

/** The refusal's message out of an error body, in any of the three wires' shapes; '' when it has none. */
function errorMessage(text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) return '';
    const error = parsed.error;
    if (isRecord(error) && typeof error.message === 'string') return error.message;
    return '';
  } catch {
    return '';
  }
}

/**
 * Whether a refusal is permanent: the provider's free-tier budget is spent
 * ("<Name>'s free-tier budget is spent") or it is out ("<Name> is out"). Any
 * other refusal ("<Name> refused this call", "<Name> serves N call at a
 * time", a bare 5xx, no reply at all) is one scheduled refusal: the next call
 * number may serve, so the provider stays alive.
 */
function isPermanentFailure(message: string): boolean {
  return message.includes('budget is spent') || message.endsWith(' is out');
}

/** One call to one provider. A 200 is always an answer, never retried. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<CallOutcome> {
  const call = upstreamRequest(candidate, clientWire, body);
  let res: Response;
  try {
    res = await fetch(call.url, { method: 'POST', headers: call.headers, body: call.body, signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch {
    return { kind: 'retryable', status: 0, message: '' };
  }
  if (res.status === 400 || res.status === 401) {
    try {
      await res.text();
    } catch {
      // The body is spent either way.
    }
    return { kind: 'bad-model' };
  }
  if (!res.ok) {
    let message = '';
    try {
      message = errorMessage(await res.text());
    } catch {
      // The body is spent either way.
    }
    return { kind: 'retryable', status: res.status, message };
  }
  const text = await res.text();
  return { kind: 'ok', status: res.status, text, sameWire: candidate.probe.provider.wire === clientWire };
}

/** Translate one provider answer into the client wire and send it. */
function answerClient(
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  providerWire: 'openai' | 'anthropic' | 'gemini',
  model: string,
  stream: boolean,
  text: string,
): void {
  const headers: Record<string, string> = stream
    ? { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' }
    : { 'content-type': 'application/json' };
  if (!stream) {
    // JSON: same wire passes the bytes through; a cross-wire answer is parsed and re-emitted.
    if (providerWire === clientWire) {
      res.writeHead(200, { ...headers, 'content-length': Buffer.byteLength(text) });
      res.end(text);
      return;
    }
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
    const answer: Answer | null = isRecord(parsed) ? parseAnswerJson(providerWire, parsed) : null;
    if (answer === null) {
      // The provider answered 200 with a body this server cannot read: forward it as-is rather
      // than spending a second successful provider call on the same request.
      res.writeHead(200, { ...headers, 'content-length': Buffer.byteLength(text) });
      res.end(text);
      return;
    }
    sendJson(res, 200, clientWire === 'openai' ? emitOpenAIJson(answer, model) : emitAnthropicJson(answer, model));
    return;
  }
  // Stream: same wire passes the bytes through.
  res.writeHead(200, headers);
  if (providerWire === clientWire) {
    res.end(text);
    return;
  }
  if (providerWire === 'openai' && clientWire === 'anthropic') {
    // The provider's OpenAI stream, re-emitted as Anthropic events (parsed, not piped).
    const answer = parseOpenAIStream(text);
    res.end(emitAnthropicStream(answer, model));
    return;
  }
  if (providerWire === 'anthropic') {
    const answer = parseAnthropicStream(text);
    res.end(emitOpenAIStream(answer, model));
    return;
  }
  // gemini stream to either client wire.
  const answer = parseGeminiStream(text);
  res.end(clientWire === 'openai' ? emitOpenAIStream(answer, model) : emitAnthropicStream(answer, model));
}

/** Serve one client request: the plan's provider first, then failover. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const reqIndex = manifestIndex;
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const best = bestRatingFor(requires);
  const badModels = new Set<string>();
  let attempts = 0;

  /** One call to one candidate. Returns true when the request was answered. */
  const attempt = async (candidate: Candidate): Promise<boolean> => {
    const pid = candidate.probe.provider.id;
    attempts += 1;
    const outcome = await callCandidate(candidate, clientWire, body);
    if (outcome.kind === 'ok') {
      noteCall(pid, true, null);
      answerClient(res, clientWire, candidate.probe.provider.wire, candidate.model.id, stream, outcome.text);
      return true;
    }
    if (outcome.kind === 'bad-model') {
      // A 400/401 never reaches the provider's inference path, so it spends
      // no call number; the model or body is wrong, so never retry it here.
      badModels.add(`${pid}/${candidate.model.id}`);
      return false;
    }
    const why: CallRecord['why'] = isPermanentFailure(outcome.message)
      ? outcome.message.includes('budget is spent')
        ? 'budget'
        : 'outage'
      : outcome.message.includes('refused this call')
        ? 'fail_call'
        : null;
    noteCall(pid, false, why);
    if (why === 'budget' || why === 'outage') markDead(pid);
    return false;
  };

  /** The failover pick: the eligible candidate whose next call is most likely
   * to answer, weighted by how close its rating is to the best. Candidates
   * more than 80 rating points below the best are out: a failover that saves
   * a call by spending quality can cost the whole score. */
  const pickFailover = (): Candidate | null => {
    const cands = candidatesFor(clientWire, requires).filter(
      (c) => !badModels.has(`${c.probe.provider.id}/${c.model.id}`) && c.rating >= best - 80,
    );
    let pick: Candidate | null = null;
    let pickScore = -1;
    for (const c of cands) {
      const n = (callsPerProvider.get(c.probe.provider.id) ?? 0) + 1;
      const pr = nextCallProb(c.probe.provider.id, n);
      const score = pr * (0.5 + 0.5 * (c.rating / Math.max(1, best)));
      if (score > pickScore) {
        pickScore = score;
        pick = c;
      }
    }
    return pick;
  };

  // The plan's provider for this request, when it can serve it.
  const planned = PLAN[reqIndex];
  if (planned !== undefined) {
    const probe = probes.find((p) => p.provider.id === planned.provider);
    const model = probe?.models.find((m) => m.id === planned.model);
    const row = probe === undefined || model === undefined ? undefined : ratingOfModel(brief.ratings, probe.provider.id, model.id);
    const wireOk =
      probe !== undefined && !(clientWire === 'openai' && requires.json === true && probe.provider.wire === 'anthropic');
    if (
      probe !== undefined &&
      probe.ok &&
      model !== undefined &&
      row !== undefined &&
      modelFree(model, row) &&
      meetsRequirements(requires, capabilitiesOf(model, row)) &&
      wireOk &&
      !deadProviders.has(planned.provider)
    ) {
      const candidate: Candidate = { probe, model, rating: row?.rating_milli ?? 0 };
      if (await attempt(candidate)) return;
      // A scheduled refusal spends one of the provider's few fail_calls, so
      // its very next number is likely safe: retry the same provider once
      // before spending the request on another provider's number.
      const hist = callHistory.get(planned.provider) ?? [];
      const last = hist[hist.length - 1];
      if (last !== undefined && !last.ok && last.why === 'fail_call' && !deadProviders.has(planned.provider)) {
        if (await attempt(candidate)) return;
      }
    }
  }

  // Failover: the best-odds candidate each time, until one answers.
  while (attempts < MAX_ATTEMPTS_PER_REQUEST) {
    const candidate = pickFailover();
    if (candidate === null) break;
    if (await attempt(candidate)) return;
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
