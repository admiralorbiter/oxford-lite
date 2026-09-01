# OXFORD Lite (`oxford-lite`) — Prospective Black-Box Model-Lineage Forensics

> **Status:** `[COMPLETED PROSPECTIVE NATURAL EXPERIMENT / LINEAGE FORENSICS STUDY]` (August 21–22, 2026: ~15+ commits) — Kept Public  
> **Science Book Status:** `[STUDY: 2026-08-22-oxford-ox-alpha-lineage-attribution]` (Indexed in Science Book Studies)  
> **Target Specimen:** Anonymous model `ox-alpha` (evaluated August 21–22, 2026)  
> **Ground-Truth Resolution:** Officially revealed by Z.ai on August 26, 2026 as **GLM-5.3-Flash**  
> **Portfolio Reference:** [`bigbraintime/projects/oxford-lite-model-forensics.md`](https://github.com/admiralorbiter/bigbraintime)  

---

## Retrospective: Prospective Validation & The GLM-5.3-Flash Reveal

### 1. The Natural Experiment Timeline
- **August 20, 2026:** Anonymous frontier model `ox-alpha` appears on OpenCode and OpenRouter.
- **August 21–22, 2026:** *OXFORD Lite* deploys differential tokenization geometry, multimodal probes, and refusal boundaries to conduct blind forensic attribution.
- **August 22, 2026 (Prediction Frozen):** Concluded `ox-alpha` was a distinct, unreleased Z.ai / GLM-lineage multimodal model (68% internal branch, 16% partner), distinct from public GLM-5.3. Withheld the "GLM-5.3 Flash" naming rumor due to insufficient grounding.
- **August 26, 2026 (Ground Truth):** Z.ai officially confirms `ox-alpha` was **GLM-5.3-Flash** (320B-parameter MoE, 18B active parameters, first natively multimodal GLM-5 model).

```text
                  [ THE PROSPECTIVE NATURAL EXPERIMENT ]
                  
     UNKNOWN SPECIMEN: "ox-alpha" (August 20, 2026)
                           │
                           ▼
          [ BLIND FORENSIC PROBING (Aug 21-22) ]
          • Differential tokenization geometry
          • Multimodal capability tests
          • Refusal & calibration bounds
                           │
                           ▼
          [ FROZEN PREDICTION (Aug 22) ]
          • Family: GLM (Z.ai) [P=0.84]
          • Status: Unreleased multimodal branch
          • Refused to overclaim "Flash" rumor
                           │
                           ▼
          [ OFFICIAL DISCLOSURE (Aug 26) ]
          • Ground Truth: GLM-5.3-Flash
```

### 2. Methodological Findings & Epistemic Calibration

1. **Successful Black-Box Attribution:** Correctly identified Z.ai origin, GLM family, unreleased status, and native multimodality prior to disclosure.
2. **The Partial Miss (Tokenizer vs. Base-Weight Lineage):** Oxford inferred GLM-5.2 base descent because token-count geometry matched GLM-5.2. In reality, GLM-5.3-Flash was trained from a *new base model* that inherited the GLM tokenizer.
   $$\mathbf{Tokenizer\ Lineage 
eq Base-Weight\ Ancestry 
eq Post-Training\ Genealogy}$$
3. **Calibrated Epistemic Discipline:** Having the correct rumor available but refusing to promote it without evidence represents sound scientific epistemology (*Justified belief $\gg$ lucky guessing*).

---

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

## Exploration 2A: Causal Support Dynamics ($F_M^{\text{dynamic}}$)

Rather than evaluating raw benchmark accuracy, Exploration 2A measures the **counterfactual response trajectory** under 8 paired causal interventions across minimal multi-path support environments $S(X) = \{\{A, B\}, \{C, D\}\}$:

$$F_M(W) = [y_{\text{base}}, y_{-A}, y_{-C}, y_{-AC}, y_{-AB}, y_{-ABC}, y_{\text{rescue}}, y_{\text{sham}}]$$

- **Isomorphic Twins**: Every base world is paired with an adversarially permuted twin $W'$ (randomized fact IDs, shuffled presentation, reversed rules, and inverted conjunctions) to test within-model stability.
- **Lexical vs Support Invariance (Sham)**: Retracts an unrelated distractor fact $E$ to separate true intervention-consistent support reasoning from superficial keyword sensitivity.
- **Permanent Calibration Fixture**: Frozen as `support-dynamics-elementary-v1` (`worlds/holdout/support_dynamics_holdout.json`, SHA-256 committed before target evaluation).

```bash
# Run causal support dynamics assay against target
python oxford.py dynamics-assay --open
```

---

## Exploration 3: Latent Support Acquisition & Four-Channel Lineage Forensics

Exploration 3 decomposes model similarity into four distinct, orthogonal evidence channels:
1. **Structural Channel ($\mathcal{F}^{\text{structural}}$)**: Tokenizer geometry, byte merges, and vocabulary serialization.
2. **Cognitive Channel ($\mathcal{F}^{\text{cognitive}}$)**: Minimal-support graph acquisition, independent root retention, and transition dynamics ($F_{\text{abandon}}^-, F_{A \to U}, F_{A \to R}, F^+, RIC(e)$).
3. **Calibration Channel ($\mathcal{F}^{\text{calibration}}$)**: Latent epistemic state mapping under paired label-decoupled codebooks ($F_{\text{cal}} = F_{\text{false}}^{\text{codebook}}$).
4. **Surface Channel ($\mathcal{F}^{\text{surface}}$)**: Strict schema compliance ($K_M$), localized renderer-flip distributions ($R_M$), and paired ontology label attraction ($L_M^{\text{flip}}$).

### Commands:

```bash
# 1. Run support acquisition assay on target model
python oxford.py acquisition-assay --target stealth/ox-alpha

# 2. Run paired label invariance assay (Exploration 3C)
python oxford.py label-invariance --target stealth/ox-alpha

# 3. Compile four-channel calibration ledger & distance matrices
python oxford.py lineage-calibrate
```

---

## Exploration 4: Normative Pressure & Post-Training Morphology

Adapted from the IMPACT research architecture, Exploration 4 evaluates post-training alignment morphology across standardized institutional pressures (Authority, Metric/KPI, Incentive, Social Consensus, and Corrective Evidence):
- **Morphology States**: Resistance ($R$), Assimilation ($A$), Compartmentalized Compliance ($C$), Judgment-Only Shift ($J$).
- **Estimands**: Pressure Selectivity ($|\Delta_{\text{Evidence}}| \gg |\Delta_{\text{Pressure}}|$) and Option Invariance ($O_M$).

---

## Epistemic Ledger & Scientific Boundaries

- **Structural Channel**: Constant wrapper overhead $k$ drops out under differential baseline subtraction. Matching tokenizer geometry indicates compatible tokenizer/vocab boundaries.
- **Cognitive Sibling Attribution**: Sibling-discordant cells ($\mathcal{S}_k^*$) and signed branch index ($B_k$) place candidate targets on known post-training transitions.
- **Attribution Boundary**: Structural, cognitive, and multimodal phenotypes narrow model lineage and post-training families without premature claims regarding hosting infrastructure or commercial product mappings.

See [`docs/PILOT_PROTOCOL.md`](file:///c:/Users/admir/Github/oxford-lite/docs/PILOT_PROTOCOL.md) for the complete scientific protocol.

