# OXFORD Lite

A black-box model-lineage and differential tokenization geometry assay suite for **Ox Alpha**.

OXFORD Lite isolates tokenization boundaries and structural signatures by comparing differential prompt-token counts:
$$[T_{\text{target}}(x_i) - T_{\text{target}}(x_0)] \quad \text{vs} \quad [T_{\text{candidate}}(x_i) - T_{\text{candidate}}(x_0)]$$

Under this formulation, any constant wrapper overhead $k$ cancels completely:
$$[T_{\text{candidate}}(x_i) + k] - [T_{\text{candidate}}(x_0) + k] = T_{\text{candidate}}(x_i) - T_{\text{candidate}}(x_0)$$

---

## Assay Architecture

```
                    ┌───────────────────────────────┐
                    │     OXFORD Exploration 1      │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
1. STRUCTURAL VALIDITY      2. LOCAL EXPLORATION        3. REMOTE ASSAYS
- Real fail-closed Rust     - Empirical Tokenizer       - Positive control on
  tokenizers (`zai-org`,      Collision Monte Carlo       known specimens
  `Qwen2.5`, `gemma-2`,       (1M trials null test)     - Resumable cells
  `cl100k`, `o200k`)        - High-Information Probe      (`--resume latest`)
- Remote Ox Alpha ONLY        Synthesizer (variance-    - Multi-Envelope wrapper
  (6 fast requests total)     ranked probe selector)      invariance assay
- Immune to candidate       - Offline Ollama assays     - Model-aware backoff &
  API rate limits                                         provider pinning
```

---

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API Key**:
   - Copy `.env.example` to `.env` (e.g. `copy .env.example .env` on Windows).
   - Set `OPENROUTER_API_KEY=sk-or-v1-...` in `.env`.
3. **Verify Environment**:
   ```bash
   python oxford.py doctor
   ```
4. **Run the Structural Assay (Recommended - 6 Ox Alpha calls only)**:
   ```bash
   python oxford.py structural --open
   ```

---

## Zero-Dollar Laptop Experiments ($0 API Spend)

### 1. Empirical Tokenizer Collision Simulation (`collision`)
Evaluates candidate tokenizers across synthetic probes and runs a 100,000+ trial Monte Carlo simulation measuring the empirical collision rate of unrelated tokenizers under the null hypothesis:
```bash
python oxford.py collision --trials 100000 --probes-pool 2000
```
*Outputs empirical odds (e.g. $P(k=4 \text{ collisions}) \approx 0.16\%$, $P(k=6) \approx 0.004\%$).*

### 2. High-Information Probe Synthesizer (`synthesize-probes`)
Generates 5,000+ diverse candidate strings locally, measures token length variance across candidate tokenizers, filters out uninformative strings, and exports the top discriminatory probes:
```bash
python oxford.py synthesize-probes --count 5000 --top-k 16
```
*Saves selected probes to `probes/high_information_probes.json`.*

### 3. Environment & Tokenizer Auditor (`doctor`)
Performs a fail-closed integrity audit of local Rust tokenizer backends and API connectivity:
```bash
python oxford.py doctor
```

---

## Target & Validation Assays

### 4. Structural Assay (`structural`)
Runs real in-memory tokenizers (`zai-org/GLM-5.2`, `Qwen/Qwen2.5-7B-Instruct`, `alpindale/gemma-2b`, `cl100k`, `o200k`), queries **only Ox Alpha** remotely, and tests normalized shape matching:
```bash
python oxford.py structural --open
```

### 5. Multi-Envelope Wrapper Invariance Assay (`envelope`)
Tests whether content delta geometry remains invariant ($\Delta T_i \equiv \Delta T_{\text{candidate}}$) while the wrapper intercept $k_e$ shifts across 3 frozen request envelopes (minimal, standard instruction, system+user):
```bash
python oxford.py envelope --open
```

### 6. Known-Specimen Positive Control (`positive-control`)
Validates OXFORD against a known specimen (e.g. `z-ai/glm-5.2:free` or `qwen/qwen-2.5-7b-instruct`):
```bash
python oxford.py positive-control --open
```

### 7. Resumable Remote Assay (`remote` or `pilot`)
Runs remote models with model-aware scheduling, provider pinning, and jittered 429 retry backoff:
```bash
# Standard run
python oxford.py remote --open

# Resume a previous/interrupted run (executes only missing/failed cells)
python oxford.py remote --resume latest --open

# Use paid OpenRouter endpoints (bypasses free shared-pool congestion)
python oxford.py remote --paid --open
```

### 8. Unit Tests
```bash
python -m unittest discover -s tests
```

---

## Run Artifacts

Each assay produces structured run artifacts under `runs/<timestamp>-<mode>/`:
- `manifest.json` — Frozen models, local tokenizers, probes, hashes, and configuration
- `raw/attempts.jsonl` — Immutable append-only record of all request attempts, status codes, and latencies
- `raw/observations.jsonl` — Active deduplicated cell observations
- `summary.json` — Count matrices, normalized deltas, exact matches, and MAE rankings
- `report.md` — Markdown summary report
- `report.html` — Visual dashboard with interactive tables and tier badges

---

## Epistemic Ledger & Scientific Boundary

- **Offset Invariance**: Constant wrapper overhead $k$ drops out under differential baseline subtraction:
  $$[T_{\text{target}}(x_i) - T_{\text{target}}(x_0)] = T_{\text{tokenizer}}(x_i) - T_{\text{tokenizer}}(x_0)$$
- **Attribution Boundary**: A matching differential tokenization geometry indicates a shared tokenizer/vocab family, but does not identify a specific checkpoint, server operator, or provider without confirmatory behavioral and serving assays.

See [`docs/PILOT_PROTOCOL.md`](file:///c:/Users/admir/Github/oxford-lite/docs/PILOT_PROTOCOL.md) for the complete scientific protocol.
