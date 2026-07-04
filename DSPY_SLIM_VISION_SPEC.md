# dspy-slim-ts: Vision / Multimodal Support Spec for Fondraft

> **Audience**: maintainer of `dspy-slim-ts` (you).
> **Why**: Fondraft's whole reason for being is sending construction-drawing
> images to a vision LLM and getting back structured findings. Today
> `dspy-slim-ts` has no first-class image support. This document is a precise
> spec for what to add so the Fondraft refactor can light up end-to-end.
> **Target consumer model**: `openrouter/google/gemini-3-flash-preview`
> ([model card](https://openrouter.ai/google/gemini-3-flash-preview)).
> **Status**: scoped from a working probe of `dspy-slim-ts@0.1.0` against
> Fondraft's `convex/_shared/analyze.ts`; bugs surfaced by that probe are
> annotated below.

---

## 0. Executive summary

Fondraft needs to call a vision LLM with a single page image plus structured
metadata, and get back a typed list of findings. Today this is impossible
through the dspy-slim-ts signature/predict/adapter pipeline because:

1. There is no `Image` (or any other multimodal) field type.
2. The `ChatAdapter` collapses every input field into one **string** content
   block before handing it to `LM.acall`, so even if the caller smuggled a
   base64 image into a `str` field, it would arrive at the model as plain
   text — not as a real image input the model can see.
3. There is no Google/Gemini provider profile. Today `model: 'google/...'`
   silently routes to `api.openai.com` and 401s. Fondraft will use OpenRouter
   (`openrouter/google/...`), which already works at the transport layer —
   but this constrains the wire format choice (see §3).

The smallest set of changes that unblocks Fondraft is **Phase 1**:

- **A.** Add an `Image` value type (§2).
- **B.** Teach `ChatAdapter` / `JSONAdapter` to emit OpenAI-style content
  parts (`text` + `image_url`) when an input field is an `Image`,
  instead of stringifying everything (§3).
- **C.** Add a `supportsVision` capability flag on `LM` keyed off the model
  string, and reject vision signatures on non-vision models early (§4).
- **D.** A small "raw multimodal call" escape hatch for advanced users who
  want to bypass signatures entirely (§5). Optional but cheap.
- **E.** Tests, including one optional integration test against OpenRouter
  (§7).

Beyond that, the larger opportunity is a four-phase path:

1. **Phase 1 — Predict vision support**: first-class `Image` inputs through
   the existing signature/adapter/LM pipeline. Confidence: **90-95%**.
   This unblocks Fondraft's current page-level review flow.
2. **Phase 2 — RLM orchestrating vision tools**: RLM does not receive image
   tensors directly; it plans over page IDs, crop handles, sheet metadata, and
   structured observations, then calls custom vision tools implemented with
   `Predict` or `LM.acompletion`. Confidence today: **75-85%**. Do not
   one-shot this; run the experiments in §6.2 until we can justify ≥90%.
3. **Phase 3 — GEPA over vision/RLM traces**: GEPA optimizes review prompts,
   tool policies, aggregation/verifier instructions, and crop-selection
   prompts using labeled CD-set examples and RLM traces. Confidence today:
   **70-80%**. Treat this as an optimization research phase with the
   confidence gates in §6.3.
4. **Phase 4 — Native RLM image fields**: redesign RLM's prompt/evaluator
   substrate so image-bearing signatures can flow through oracle leaves
   without collapsing pixels into text. Confidence today: **45-55%**. This is
   a speculative platform phase; the goal is to experiment our way to ≥90%
   before any broad implementation.

Native RLM image passthrough remains **explicitly out of scope for Phase 1**.
Fondraft's immediate production need is `Predict`; the revolutionary path is
tracked, but phased behind evidence.

---

## 1. Concrete Fondraft call shape

This is the call site we need to make work — see
[`convex/_shared/analyze.ts`](convex/_shared/analyze.ts):

```ts
import {
  settings, LM, Image, Predict
} from 'dspy-slim-ts';

settings.configure({
  lm: new LM({
    model: 'openrouter/google/gemini-3-flash-preview',
    apiKey: process.env.OPENROUTER_API_KEY!,
    // Optional but recommended (OpenRouter rankings):
    headers: {
      'HTTP-Referer': 'https://fondraft.app',
      'X-Title': 'Fondraft'
    }
  })
});

const analyze = new Predict(
  'page_image: Image, page_number: int, document_title: str, ' +
    'page_width: int, page_height: int, system_prompt: str ' +
    '-> findings: list[dict]'
);

const prediction = await analyze.aforward({
  page_image: Image.fromBuffer(pngBuffer, 'image/png'),
  page_number: 20,
  document_title: 'Sanibel Fire Station',
  page_width: 3600,
  page_height: 2400,
  system_prompt: SYSTEM_PROMPT
});

const findings = prediction.getOr<unknown[]>('findings', []);
```

Output of that flow on the wire (what hits OpenRouter):

```json
{
  "model": "google/gemini-3-flash-preview",
  "messages": [
    { "role": "system", "content": "<adapter-built signature instructions>" },
    {
      "role": "user",
      "content": [
        { "type": "text",
          "text": "[[ ## page_number ## ]]\n20\n\n[[ ## document_title ## ]]\nSanibel Fire Station\n\n[[ ## page_width ## ]]\n3600\n\n[[ ## page_height ## ]]\n2400\n\n[[ ## system_prompt ## ]]\nYou are an expert construction document reviewer…\n\n[[ ## page_image ## ]]" },
        { "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBORw0KGgo…" } },
        { "type": "text",
          "text": "Respond with the corresponding output fields, starting with `[[ ## findings ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`." }
      ]
    }
  ],
  "response_format": { "type": "json_schema", "json_schema": { /* findings schema */ } }
}
```

Key wire-format facts (verified from
[OpenRouter docs](https://openrouter.ai/docs/features/multimodal/images)):

- `image_url.url` accepts both **HTTPS URLs** and **`data:<mime>;base64,…`** URIs.
- Supported MIME types: `image/png`, `image/jpeg`, `image/webp`, `image/gif`.
- OpenRouter recommendation: **send text first, then images**. (If image must
  come first, put it in the system prompt — we won't.)
- `dspy-slim-ts` already declares `ContentPart` with `type: 'image_url'` in
  [`chat_message.d.ts`](../dspy-slim-project/dspy-slim-ts/dist/chat_message.d.ts) — the type is in place; only the adapter is missing.

---

## 2. New value type: `Image`

### 2.1 Public surface

```ts
// dspy-slim-ts/src/image.ts (new file)

export type SupportedImageMime =
  | 'image/png'
  | 'image/jpeg'
  | 'image/webp'
  | 'image/gif';

export class Image {
  readonly mimeType: SupportedImageMime;
  /** Either `url` or `bytes` is set; never both empty. */
  readonly url?: string;          // 'https://…' or 'data:<mime>;base64,…'
  readonly bytes?: Uint8Array;    // raw, will be base64-encoded at send time

  private constructor(args: { mimeType: SupportedImageMime; url?: string; bytes?: Uint8Array });

  static fromBuffer(bytes: Uint8Array | ArrayBuffer | Buffer, mimeType: SupportedImageMime): Image;
  static fromBase64(base64: string, mimeType: SupportedImageMime): Image;   // bare base64, no data: prefix
  static fromDataUri(dataUri: string): Image;                               // parses & validates
  static fromUrl(url: string, mimeType?: SupportedImageMime): Image;        // public https URL

  /** Always returns a `data:<mime>;base64,…` URI ready for `image_url.url`. */
  toDataUri(): string;
  /** Tagging helper used by the adapter. */
  readonly _isDspyImage: true;
}

export function isImage(v: unknown): v is Image;
```

### 2.2 Validation rules

- `mimeType` must be one of the four supported MIMEs. Reject others with
  `ConfigurationError("Unsupported image mime type: …")`.
- For `fromBuffer` / `fromBase64`: payload must be non-empty. We do **not**
  parse the magic bytes — trust the caller.
- For `fromDataUri`: must match `^data:(image\/(?:png|jpeg|webp|gif));base64,(.+)$`.
- Soft warning (single `console.warn`, not throw) if the resulting base64
  payload exceeds 20 MB; OpenRouter accepts large payloads but Convex
  serialization gets unhappy past ~32 MB.

### 2.3 Why not also `Audio` / `File` / `PDF` v1?

- Fondraft only ships images today. Phase 2 may want PDFs (Gemini natively
  accepts PDFs via `inline_data`), but OpenRouter's chat completions
  surface doesn't expose PDF input — only `image_url`.
- Scope-controlled v1 = Image only. Add `Audio`/`File` later by mirroring
  the same shape.

### 2.4 Export from `index.ts`

```ts
export { Image, isImage, type SupportedImageMime } from './image.js';
```

---

## 3. Adapter changes

### 3.1 What's wrong today

`ChatAdapter.formatUserMessageContent`
([adapter.js:306](../dspy-slim-project/dspy-slim-ts/dist/adapter.js#L306))
returns a single `string`. Every input field — including images smuggled in
as `str` — gets `JSON.stringify`'d into that string. The downstream `Message`
ends up as `{role:'user', content:'…3 MB of base64…'}`, not as a content-part
array.

### 3.2 New behaviour

When at least one input field is an `Image`:

1. Build the user message as `Message<{role:'user', content: ContentPart[]}>`.
2. Emit content parts in this order (per OpenRouter "text first" guidance):
   - **Part A — non-image text block**: a single `{type:'text'}` containing
     the rendered `[[ ## name ## ]] / value` blocks for every non-Image
     input field, in declaration order, plus the image-field markers like
     `[[ ## page_image ## ]]` (no inline value — value is the next part).
   - **Part B — image content parts**: one
     `{type:'image_url', image_url:{url: image.toDataUri()}}` per Image
     field, in declaration order. **Each immediately follows its marker** —
     so if you have two Image fields, interleave the marker text + image as
     two text/image pairs rather than dumping all images at the end.
   - **Part C — output requirements text**: the existing
     `userMessageOutputRequirements(signature)` string as a final
     `{type:'text'}` block.

This preserves `[[ ## name ## ]]` markers so the existing structured-output
parser (`JSONAdapter.parse`) keeps working unchanged.

### 3.3 Worked example

Inputs: `{ page_image: Image, page_number: 20, system_prompt: "…" }`.
Signature: `page_image: Image, page_number: int, system_prompt: str -> findings: list[dict]`.

Resulting `content`:

```ts
[
  { type: 'text',  text: '[[ ## page_number ## ]]\n20\n\n[[ ## system_prompt ## ]]\n…' },
  { type: 'text',  text: '[[ ## page_image ## ]]' },
  { type: 'image_url', image_url: { url: 'data:image/png;base64,…' } },
  { type: 'text',  text: 'Respond with the corresponding output fields, starting with `[[ ## findings ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.' }
]
```

### 3.4 Demos / few-shot

Few-shot demos sometimes carry image inputs too. **v1 simplification**: if a
demo input contains an `Image`, replace it with a placeholder text
`'<image elided in demo>'` in the demo's user-message content, and emit no
image part for demos. Real models tolerate this; it keeps adapter logic
linear.

### 3.5 Conversation history

Same rule as demos — historical user messages with images get the placeholder
treatment. Fondraft doesn't use `history` today, so this doesn't need to be
beautiful.

### 3.6 Detection helper

```ts
// adapter.ts
function fieldHasImage(value: unknown): boolean {
  return isImage(value);
}

function inputsContainImage(inputs: Record<string, unknown>): boolean {
  return Object.values(inputs).some(fieldHasImage);
}
```

If `inputsContainImage(inputs)` is `false`, take the existing string-only
path verbatim — zero behaviour change for non-vision callers.

### 3.7 Signature parser

The string DSL (`'a: str, b: int -> c: list[dict]'`) parser must accept
`Image` as a recognized type token. Map it to a runtime descriptor that
`format` can detect (in concert with `isImage` on the value). Other than
parsing, the type doesn't change Predict's coercion logic — Image fields
are input-only; `prediction.get('page_image')` is irrelevant.

If the parser is purely string-based and doesn't enforce types, simply
adding `Image` to its known type list is enough. Otherwise, declare the
type token alongside `str`/`int`/`bool`/`list`/`dict`.

---

## 4. LM capability flag

### 4.1 New getter

```ts
// lm.ts
get supportsVision(): boolean { /* … */ }
```

### 4.2 Detection table

Keyed off the **fully-qualified** model string; check via substring match
after `providerNameFromModel` parsing:

| Model substring                        | `supportsVision` |
|----------------------------------------|------------------|
| `openrouter/google/gemini-3`           | ✅ |
| `openrouter/google/gemini-2.5`         | ✅ |
| `openrouter/google/gemini-2.0`         | ✅ |
| `openrouter/google/gemini-1.5`         | ✅ |
| `openrouter/anthropic/claude-3`        | ✅ |
| `openrouter/anthropic/claude-sonnet`   | ✅ |
| `openrouter/anthropic/claude-opus`     | ✅ |
| `openrouter/openai/gpt-4o`             | ✅ |
| `openrouter/openai/gpt-4-turbo`        | ✅ |
| `openrouter/openai/gpt-4-vision`       | ✅ |
| `openai/gpt-4o`                        | ✅ |
| `openai/gpt-4-turbo`                   | ✅ |
| `openai/gpt-4-vision`                  | ✅ |
| (anything else)                        | ❌ |

Implementation: a small `VISION_MODEL_PATTERNS: readonly RegExp[]` array;
`supportsVision` returns `patterns.some(p => p.test(this.model))`.

Don't be too clever: this list is human-maintained. Add a one-liner
JSDoc on the array pointing at this spec.

### 4.3 Enforcement

In `Predict.aforward` (or in the adapter, before calling `lm.acall`):

```ts
if (inputsContainImage(inputs) && !lm.supportsVision) {
  throw new ConfigurationError(
    `Signature includes Image input(s) but model "${lm.model}" is not on the vision allowlist. ` +
    `Use a vision-capable model (e.g. openrouter/google/gemini-3-flash-preview) or remove the Image field.`
  );
}
```

Throw **before** the HTTP call — fondraft would otherwise pay the LLM
round-trip cost just to find out.

### 4.4 Override knob

Power users may want to opt in for an experimental model not yet on the
list. Add an `LMOptions.forceVisionCapable?: boolean` (default `false`)
that short-circuits the check.

---

## 5. Raw escape hatch (optional, ~30 LOC)

Some Fondraft tools (Phase 2 RLM tools) want full control over the message
payload — same as the Python POC's `lm(messages=[...])` pattern. The
existing `BaseLM.acall(undefined, messages, kwargs)` already supports this,
but the typing is awkward (`prompt` arg ignored, `Message` type allows only
`'user' | 'assistant' | 'system'`).

Add a small convenience method:

```ts
// lm.ts
async acompletion(args: {
  messages: readonly Message[];
  kwargs?: Record<string, unknown>;
}): Promise<string> {
  const outs = await this.acall(undefined, args.messages, args.kwargs);
  return outs[0]?.text ?? '';
}
```

This is what Fondraft's `convex/_shared/analyze.ts` will fall back to if
the signature route hits a snag in production. Keep both options open.

---

## 6. RLM / GEPA roadmap

### 6.1 Phase 1 rule: no native RLM image fields

`RLM` currently collapses input fields to a single text prompt before calling
its internal `Predict` leaves. The evaluator's oracle signatures are
text-only (`prompt: str -> answer: str` and the effect-loop variant), so an
`Image` value passed through the top-level RLM signature would be serialized
or dropped instead of reaching the model as a real multimodal content part.

**Phase 1 decision**: Image fields on an `RLM` signature throw at construction
time or first call:

```ts
throw new ConfigurationError(
  'RLM does not support native Image input fields yet. Use Predict for one-shot ' +
  'vision tasks, or wrap vision calls in RLM tools.'
);
```

This keeps Phase 1 honest: Fondraft can ship page-level review through
`Predict`, and users get a clear error instead of a false sense that RLM saw
the image.

### 6.2 Phase 2: RLM orchestrates vision tools

**Hypothesis**: RLM does not need native image tensors to be valuable for CD
set review. It can operate as the planning and verification layer over a set
of typed vision tools:

- `InspectPage({pageId, prompt}) -> findings[]`
- `InspectCrop({pageId, bbox, prompt}) -> observations[]`
- `CompareRegions({leftPageId, leftBbox, rightPageId, rightBbox, prompt}) -> differences[]`
- `VerifyFinding({pageId, finding, prompt}) -> verdict`
- `ExtractCallouts({pageId, prompt}) -> callouts[]`

Each tool owns the multimodal call, using either `Predict('image: Image, ...')`
or the raw `LM.acompletion({messages})` escape hatch. RLM receives only JSON
observations, bounded text excerpts, page/crop handles, and prior findings.
This aligns with the current effect system: `Custom` effects already provide
an open-world handler boundary.

Confidence today: **75-85%**. We should not one-shot a production
implementation. The goal is to experiment to ≥90% confidence with these
gates:

1. **Tool boundary spike**: implement one local custom RLM effect named
   `InspectPage` that calls a replayed/stubbed vision `Predict` and returns
   structured JSON. Success: an RLM plan can request the tool, ingest the
   result, and return a finding without touching RLM core.
2. **Live single-page smoke**: run the same handler against
   `openrouter/google/gemini-3-flash-preview` on a tiny image fixture and one
   real Fondraft sheet. Success: tool latency, request body, and parsed output
   are stable across at least 10 repeated runs.
3. **Multi-tool orchestration**: add `InspectCrop` and `VerifyFinding` on a
   small CD-set fixture. Success: RLM chooses at least one targeted follow-up
   crop/check after an initial page pass and improves precision or recall
   versus one-shot `Predict`.
4. **Budget and trace audit**: confirm the run respects `RLMBudget`,
   records `_rlm_trace`, and never stores base64 payloads in trace/memory.
   Success: traces contain page/crop references and tool outputs only.
5. **Acceptance threshold**: Phase 2 reaches ≥90% confidence when a 10-20 page
   Fondraft fixture shows better review quality than one-shot `Predict` under
   a bounded call budget, with deterministic replay tests for the handler
   protocol.

Implementation after confidence gate:

- Add examples/docs for vision custom effects rather than changing RLM core.
- Add tests for handler dispatch, JSON validation, trace redaction, and budget
  accounting.
- Keep image storage/cropping outside RLM; RLM receives stable opaque handles.

### 6.3 Phase 3: GEPA optimizes vision/RLM programs

**Hypothesis**: GEPA can optimize multimodal review systems without becoming
multimodal itself. It only needs instruction-bearing targets, reflective
traces, metric feedback, and candidate artifacts. Images stay represented as
fixture IDs, page IDs, crop boxes, and content hashes.

Confidence today: **70-80%** because the current GEPA facade is substrate-level
and gated behind an external engine. The unknowns are not wire format; they
are whether the traces and metrics carry enough signal for useful prompt and
policy evolution.

Experiment path to ≥90% confidence:

1. **Trace shape audit**: run Phase 2 vision-tool traces through
   `capturePredictorTraces` and `materializeReflectiveDataset`. Success:
   reflective data is JSON-serializable, compact, and free of base64/image
   bytes.
2. **Metric definition**: build a small labeled Fondraft eval set with
   finding-level precision/recall, severity correctness, bbox/page reference
   correctness, and duplicate penalty. Success: metric feedback explains
   failures in text that an optimizer can use.
3. **Static engine rehearsal**: use `createStaticGEPAEngine` to attach a known
   better instruction artifact to the vision/RLM program. Success: GEPA can
   project the right instruction cells and attach optimized instructions
   without corrupting the program.
4. **Manual candidate loop**: before trusting an automated optimizer, run a
   small human-in-the-loop candidate sweep over 5-10 instruction variants.
   Success: the GEPA target surfaces correspond to prompts whose changes move
   the metric.
5. **Engine-backed pilot**: connect an approved GEPA engine for a tiny budget
   run. Success: selected candidate improves held-out score over baseline and
   artifacts are reproducible enough to check into test fixtures.

Phase 3 reaches ≥90% confidence when GEPA produces a measurable held-out
improvement on a fixed Fondraft eval set and the saved optimization artifact
can be reapplied deterministically.

### 6.4 Phase 4: native RLM image fields

**Hypothesis**: Native RLM image signatures would let users write
`new RLM('page_image: Image, prompt: str -> findings: list[dict]')` and have
oracle leaves receive multimodal content parts directly. This is the most
framework-revolutionary path, but it cuts across RLM's current design.

Confidence today: **45-55%**. Do not implement broadly until experiments
settle the contract. Key design questions:

- Should `split`, `map`, `vote`, and `ensemble` operate over image fields, text
  fields, or structured multimodal bundles?
- Does an oracle leaf receive all images every time, image handles plus a tool
  for fetching crops, or only the image subset selected by the planner?
- How do traces stay useful without persisting large binary payloads?
- How do demos/history with images work without exploding prompt cost?
- Should the image value be part of `CombinatorValue`, or should it remain an
  external handle resolved only at LM-call time?

Experiment path to ≥90% confidence:

1. **Design prototype behind a flag**: create a private prototype that extends
   `CombinatorValue` or introduces a `MultimodalPrompt` carrier. Success:
   one oracle leaf can call a vision model without changing public RLM APIs.
2. **Single-leaf equivalence**: prove native RLM image input can reproduce
   Phase 1 `Predict` output on the same fixtures. Success: identical request
   shape or intentionally documented differences.
3. **Combinator semantics tests**: define what `split`, `map`, `vote`, and
   `ensemble` do with image-bearing inputs. Success: deterministic unit tests
   cover each legal behavior and reject ambiguous cases.
4. **Cost/trace guardrails**: ensure images are deduped across oracle calls,
   represented by handles in traces, and subject to a multimodal budget.
   Success: repeated self-consistency runs do not repeatedly serialize the
   same large image unless explicitly requested.
5. **Compare against Phase 2**: native image RLM must beat or simplify the
   tool-orchestration design on at least one real Fondraft workflow. Success:
   either higher quality at equal budget or a materially simpler user API
   without hiding critical control.

Only after those gates should native RLM image fields graduate from prototype
to public API. Until then, Phase 2 is the recommended architecture for
multistep vision reasoning.

---

## 7. Tests

### 7.1 Unit tests (must pass deterministically)

```
tests/image.test.ts
  - Image.fromBuffer(…, 'image/png').toDataUri() round-trips
  - Image.fromDataUri('data:image/png;base64,…') parses mime+bytes
  - Image.fromBuffer(…, 'image/bmp') throws ConfigurationError
  - isImage(new Image(…)) === true; isImage('foo') === false
  - 25 MB buffer triggers a single console.warn (use a spy)

tests/signature_image.test.ts
  - Parser accepts 'page_image: Image, n: int -> a: str'
  - Parser still accepts existing 'a: str -> b: int' verbatim

tests/adapter_image.test.ts
  - format({page_image: Image, n: int -> answer: str}, …, {page_image, n: 5})
    returns Message[] whose user message has content = ContentPart[] with:
      [text(non-image markers + n=5), text('[[ ## page_image ## ]]'),
       image_url(data:image/png;base64,…), text(output requirements)]
  - format with NO image fields returns content: string (unchanged)
  - Two Image fields produce interleaved [marker, image_url, marker, image_url]
    in declaration order
  - Demo with Image input: image elided, placeholder text in its place

tests/lm_vision.test.ts
  - new LM({model:'openrouter/google/gemini-3-flash-preview'}).supportsVision === true
  - new LM({model:'openai/gpt-4.1-mini'}).supportsVision === false
  - forceVisionCapable: true overrides

tests/predict_vision.test.ts (uses ReplayLM)
  - Signature with Image but stubbed non-vision LM throws ConfigurationError
  - Signature with Image + vision LM (replayed) returns parsed prediction
```

### 7.2 Integration test (gated)

```
tests/integration_openrouter_vision.test.ts
  - Skipped unless process.env.OPENROUTER_API_KEY is set.
  - Loads tests/fixtures/tiny-checkerboard.png (~2 KB).
  - Calls Predict('image: Image, prompt: str -> answer: str') with
    model='openrouter/google/gemini-3-flash-preview' and prompt
    'Reply with exactly one word: the dominant pattern.'
  - Asserts answer.length > 0 and answer.toLowerCase() includes
    'check' or 'grid' or 'pattern'.
  - Budget: < $0.001 per run (flash is cheap).
```

### 7.3 Snapshot the OpenRouter request body

In the adapter test, mock `fetch` and snapshot the JSON body sent. Helps
catch regressions where the adapter rebuilds the content shape incorrectly
(e.g. nests `image_url` wrong, swaps order).

---

## 8. Non-goals / explicitly punted

- Streaming vision responses in Phase 1.
- Audio (`Audio` field type).
- PDF input (`PDF` field type) — Gemini supports it natively but OpenRouter
  doesn't surface it; revisit if/when we move off OpenRouter.
- Native RLM image fields in Phase 1. Phase 2 uses tools; Phase 4 researches
  native image passthrough.
- Automatic image resizing/compression — caller's responsibility.
- Image generation (output-side multimodal).

---

## 9. Concrete bugs surfaced by today's probe

These are independent of the vision spec but were uncovered by trying to
exercise the current Fondraft refactor and should be tracked alongside the
work above:

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `convex/_shared/analyze.ts` | `taskType: 'extraction'` is not a valid `TaskType` (valid: `search` \| `classify` \| `aggregate` \| `pairwise` \| `summarise` \| `multi_hop` \| `unknown`). | Switch to `Predict` (no `taskType`); RLM is wrong primitive for one-shot vision extraction anyway. |
| 2 | `convex/_shared/analyze.ts` | `model: 'google/gemini-2.5-pro'` falls through to OpenAI's base URL → 401. | Switch to `model: 'openrouter/google/gemini-3-flash-preview'` and read `OPENROUTER_API_KEY` from Convex env. |
| 3 | `convex/_shared/analyze.ts` | Sends `imageBase64: str` as a normal text field — model never sees the image. | Resolved by this spec (§§2–3). Field becomes `page_image: Image`. |

---

## 10. Rollout order (suggested)

### Phase 1 — Predict vision support

1. Land §2 (`Image` type) + §7.1 unit tests — pure, no external deps.
2. Land §4 (`supportsVision`) + tests.
3. Land §3 (adapter changes) + adapter unit tests + snapshot test.
4. Land §5 (`acompletion`) — trivial, ship alongside §3.
5. Wire §7.2 integration test (gated).
6. Bump dspy-slim-ts version, update Fondraft's `convex/_shared/analyze.ts`
   to the §1 call shape, run a real Sanibel page through it, compare
   findings count to Phase 0/1 baselines (~10 findings/page).

Each step ships independently and leaves the library in a working state.

### Phase 2 — RLM vision tools

Do not start with a broad RLM-core rewrite. Start with the §6.2 experiments:

1. Build a replayed `InspectPage` custom effect handler.
2. Run a live single-page smoke against OpenRouter.
3. Add `InspectCrop` and `VerifyFinding` on a small Fondraft fixture.
4. Audit trace redaction and budget behavior.
5. Promote to public examples/docs only after the fixture beats one-shot
   `Predict` quality under a bounded call budget.

### Phase 3 — GEPA for vision/RLM

Do not assume GEPA works just because the abstractions line up. Run the §6.3
confidence ladder:

1. Verify trace materialization with image handles, not bytes.
2. Define finding-level metrics and feedback text.
3. Rehearse artifact attach with `createStaticGEPAEngine`.
4. Run a manual candidate sweep to validate that target instructions move the
   metric.
5. Connect an approved engine for a tiny pilot and require held-out
   improvement before broad use.

### Phase 4 — Native RLM images

Treat this as a platform prototype until confidence reaches ≥90%:

1. Prototype a multimodal prompt carrier behind a private flag.
2. Prove single-leaf equivalence with Phase 1 `Predict`.
3. Specify and test image semantics for `split`, `map`, `vote`, and
   `ensemble`.
4. Add multimodal budget, dedupe, and trace guardrails.
5. Graduate only if native RLM images beat or materially simplify the Phase 2
   tool architecture on a real Fondraft workflow.

---

## 11. References

- Current Fondraft analyze boundary: [`convex/_shared/analyze.ts`](convex/_shared/analyze.ts)
- Current dspy-slim-ts adapter: [`adapter.js:271-340`](../dspy-slim-project/dspy-slim-ts/dist/adapter.js#L271-L340)
- Current dspy-slim-ts message types: [`chat_message.d.ts`](../dspy-slim-project/dspy-slim-ts/dist/chat_message.d.ts) (already declares `ContentPart` with `image_url`)
- OpenRouter vision API: <https://openrouter.ai/docs/features/multimodal/images>
- Target model: <https://openrouter.ai/google/gemini-3-flash-preview>
- Python POC reference (vision call shape we're cloning): [`poc/src/fondraft_poc/analyze.py`](poc/src/fondraft_poc/analyze.py) lines ~95–125
- Fondraft refactor master plan: [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md)
