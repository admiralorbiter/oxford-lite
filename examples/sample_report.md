# OXFORD Lite — DEMO / SYNTHETIC DATA

> **Pilot only.** This report measures a tiny structural fingerprint. It does not identify an exact model, checkpoint, or provider.

Run: `20260821-220815-demo`  
Generated: `2026-08-21T22:08:15+00:00`  
Successful requests: **18/18**

## Pairwise structural comparison

| Ox Alpha vs. | Common probes | Exact normalized deltas | Shape MAE | Constant offset | Offset span |
|---|---:|---:|---:|---:|---:|
| GLM-5.2 (free) | 6 | 5/5 (100%) | 0.00 | Yes (+75) | 0 |
| Gemma 4 26B A4B (free) | 6 | 0/5 (0%) | 5.80 | No | 10 |

## Prompt-token counts

| Probe | Ox Alpha | GLM-5.2 | Gemma 4 |
|---|---:|---:|---:|
| Mixed boundaries | 109 | 34 | 29 |
| Multiscript | 116 | 41 | 44 |
| Emoji + joiners | 122 | 47 | 52 |
| Code syntax | 125 | 50 | 46 |
| Structured identifiers | 120 | 45 | 49 |
| Repetition + whitespace | 117 | 42 | 38 |

## Interpretation

Synthetic demo only. No inference about Ox Alpha is permitted from these values.

## Scientific boundary

- Existing public Ox/GLM fingerprint strings are not in this six-probe corpus.
- Absolute token counts can include wrapper/chat-template overhead; the main comparison uses differential shape.
- This pilot does not pin providers and contains no behavior/tool/vision assays.
- A successful-looking pilot is a build/plumbing green light, not confirmatory evidence.
