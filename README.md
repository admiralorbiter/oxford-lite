# OXFORD Lite

A black-box model-lineage and differential tokenization geometry assay suite for **Ox Alpha**.

OXFORD Lite isolates tokenization boundaries and structural signatures by comparing differential prompt-token counts:
$$[T_{\text{target}}(x_i) - T_{\text{target}}(x_0)] \quad \text{vs} \quad [T_{\text{candidate}}(x_i) - T_{\text{candidate}}(x_0)]$$

Under this formulation, any constant wrapper overhead $k$ (e.g. `$T_{\text{Ox}}(x) = T_{\text{GLM}}(x) + 75$`) cancels completely:
$$[T_{\text{GLM}}(x_i) + k] - [T_{\text{GLM}}(x_0) + k] = T_{\text{GLM}}(x_i) - T_{\text{GLM}}(x_0)$$

---

## Assay Architecture

```
                    ┌───────────────────────────────┐
                    │          OXFORD Lite          │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
1. STRUCTURAL ASSAY         2. REMOTE ASSAY             3. LOCAL ASSAY
- Local candidate           - Remote OpenRouter         - Local Ollama
  tokenizers (GLM, Gemma,     endpoints                   models (Gemma,
  Qwen, Tiktoken)           - Model-aware dispatch        Qwen)
- Remote Ox Alpha ONLY        (no fixed delay penalty)  - Negative controls
  (6 fast requests total)   - Resumable cells           - Behavioral micro-worlds
- Immune to candidate         (`--resume latest`)       - Fast offline iteration
  API rate limits           - `--paid` / `--free`
                            - 429 jittered backoff
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
3. **Run Doctor Check**:
   ```bash
   python oxford.py doctor
   ```
4. **Run the Structural Assay (Recommended - 6 Ox Alpha calls only)**:
   ```bash
   python oxford.py structural --open
   ```

---

## Commands & Modes

### 1. Structural Assay (`structural`)
Runs local candidate tokenizers instantly in-memory, queries **only Ox Alpha** remotely across the probe corpus, and computes normalized differential shapes:
```bash
python oxford.py structural --open
```

### 2. Remote API Assay (`remote` or `pilot`)
Runs remote target and candidate models with model-aware scheduling, provider pinning, and jittered 429 retry backoff:
```bash
# Standard run
python oxford.py remote --open

# Resume a previous/interrupted run (executes only missing/failed cells)
python oxford.py remote --resume latest --open

# Use paid OpenRouter endpoints (bypasses free shared-pool congestion)
python oxford.py remote --paid --open
```

### 3. Local Assay (`local`)
Probes local Ollama models for prompt evaluation token counts:
```bash
python oxford.py local --models gemma2:9b qwen2.5:7b --open
```

### 4. Synthetic Demo (`demo`)
Generates sample multi-tier reports with zero external calls:
```bash
python oxford.py demo --open
```

### 5. Unit Tests
```bash
python -m unittest discover -s tests
```

---

## Run Artifacts

Each assay produces structured run artifacts under `runs/<timestamp>-<mode>/`:
- `manifest.json` — Frozen models, local tokenizers, probes, hashes, and configuration
- `raw/observations.jsonl` — Loss-minimized per-cell observation logs including raw response payloads and headers
- `summary.json` — Count matrices, normalized deltas, exact matches, and MAE rankings
- `report.md` — Markdown summary report
- `report.html` — Visual dashboard with interactive tables and tier badges

---

## Scientific Boundary

- **Offset Invariance**: Constant wrapper overhead $k$ drops out under differential baseline subtraction.
- **Probe Freshness**: The bundled probes are synthetic and do not copy public community fingerprint strings.
- **Attribution Boundary**: A matching differential tokenization geometry indicates a shared tokenizer/vocab family, but does not identify a specific checkpoint, server operator, or provider without confirmatory behavioral and serving assays.

See [`docs/PILOT_PROTOCOL.md`](file:///c:/Users/admir/Github/oxford-lite/docs/PILOT_PROTOCOL.md) for the complete scientific protocol.
