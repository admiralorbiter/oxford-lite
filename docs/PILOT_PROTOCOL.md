# OXFORD Lite — Pilot Protocol

## Purpose

This pilot validates the plumbing and presentation for a later preregistered
black-box model-lineage study. It is **not** a confirmatory attribution study.

## Pilot question

Do Ox Alpha's *differential prompt-token counts* look more like GLM-5.2 than a
distant negative control on a small set of fresh probes?

## Models

- Target: `stealth/ox-alpha`
- Candidate: `z-ai/glm-5.2:free`
- Negative control: `google/gemma-4-26b-a4b-it:free`

All three are queried through the same OpenRouter API surface. The pilot does
not pin inference providers; provider controls belong in the full protocol.

## Experimental unit

One `(model, probe)` API request. Six probes × three models = 18 requests.
Request order is deterministically shuffled to avoid a simple all-of-model-A,
then-all-of-model-B collection pattern.

## Measurement

The primary field is `usage.prompt_tokens` from the raw API response.
Absolute counts are not treated as a fingerprint because wrappers and chat
templates can add overhead. For models A and B we inspect whether:

`prompt_tokens_A(x) - prompt_tokens_B(x)`

is constant across probes, and whether their vectors after subtracting the
first probe are identical.

## Interpretation

- A constant offset across six probes is an **interesting structural signal**.
- It is not proof of shared weights, exact checkpoint, or operator/provider.
- Six probes are intentionally too small for a publication-grade claim.
- Existing public Ox/GLM probes are not copied into this pilot corpus.

## Build gate after pilot

If the runner works and the report is useful, freeze the full candidate
registry, generate a much larger candidate-only probe pool, preselect
high-information probes, preregister the scoring/decision rules, and only then
run the confirmatory target study.
