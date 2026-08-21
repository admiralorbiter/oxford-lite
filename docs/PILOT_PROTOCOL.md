# OXFORD Lite — Scientific Pilot Protocol

## Purpose

This pilot formalizes the mathematical and experimental methodology for black-box model-lineage and tokenizer-partitioning assays. It is an exploratory structural assay and plumbing validation, not a confirmatory attribution claim.

---

## Core Assay: Differential Token Geometry

### Problem: Wrapper & Chat Template Invariance
Raw prompt-token counts $T(x)$ reported by black-box APIs include constant overhead from system prompts, hidden headers, and chat templates:
$$T_{\text{target}}(x) = T_{\text{tokenizer}}(x) + k_{\text{wrapper}}$$

Because $k_{\text{wrapper}}$ is unknown *a priori*, absolute token counts cannot serve as direct fingerprints.

### Solution: Differential Baseline Normalization
For any probe corpus $\{x_0, x_1, \dots, x_n\}$, choose an anchor probe $x_0$. Define the normalized differential vector:
$$\Delta T(x_i) = T(x_i) - T(x_0) \quad \text{for } i \in \{1, \dots, n\}$$

Substituting the wrapper equation:
$$\Delta T_{\text{target}}(x_i) = [T_{\text{tokenizer}}(x_i) + k] - [T_{\text{tokenizer}}(x_0) + k] = T_{\text{tokenizer}}(x_i) - T_{\text{tokenizer}}(x_0)$$

Thus, the unknown wrapper overhead $k$ cancels out identically. If a target model shares the tokenizer and vocabulary partitioning of a candidate model, its normalized differential vector must match:
$$\Delta T_{\text{target}}(x_i) \equiv \Delta T_{\text{candidate\_local}}(x_i) \quad \forall i$$

And the estimated wrapper overhead is:
$$\hat{k} = T_{\text{target}}(x_i) - T_{\text{candidate\_local}}(x_i)$$

---

## Three Experimental Surfaces

| Assay Tier | Target | Candidates / Controls | Computation / Transport | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **A. Structural Lineage** (`structural`) | `stealth/ox-alpha` (Remote) | Local tokenizers (GLM, Gemma, Qwen, Tiktoken) | **In-memory local tokenization** + 6 remote target queries | Isolates tokenizer boundaries with zero candidate API calls; immune to rate limits. |
| **B. Remote API Assays** (`remote`) | `stealth/ox-alpha` (Remote) | Remote GLM-5.2, Gemma 4, controls | OpenRouter API (resumable, provider pinned) | Confirmatory serving, latency, and provider attribution. |
| **C. Local Behavioral Assays** (`local`) | — | Local Ollama models (Gemma, Qwen, Llama) | Local GPU / CPU inference | Rapid behavioral perturbation assays and micro-world candidate ranking. |

---

## Error Handling & Resumability

1. **Cell-Level Atomicity**: Every cell is indexed by $(M, P)$ where $M$ is the model identifier and $P$ is the probe identifier.
2. **Loss-Minimized Observations**: Responses (including HTTP 429 and provider error metadata) are preserved in `raw/observations.jsonl`.
3. **Resumable Work Queues**: Invoking `--resume latest` reads existing observation logs, skips completed cells, and executes only missing/failed cells.
4. **Adaptive Backoff**: HTTP 429 responses inspect upstream `Retry-After` headers and apply exponential backoff with random jitter up to `--max-retries`.

---

## Attributability Boundaries

- A perfect shape match ($\Delta T_{\text{target}} \equiv \Delta T_{\text{candidate}}$ with $\text{MAE} = 0.0$) across synthetic probes confirms identical tokenizer segmentation for those boundary conditions.
- It does **not** identify an exact fine-tuned checkpoint, operator identity, or inference host without secondary behavioral and serving assay validation.
