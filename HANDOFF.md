# OXFORD Lite: Project Handoff & Session Manifest

**Date & Time**: 2026-08-21 23:50 EST  
**Latest Git Commit**: `2f75537` (Pushed to `main`)  
**Active Test Suite**: **43 / 43 unit tests passing in 0.31s** (`python -m unittest discover -s tests`)

---

## 1. Executive Summary: Core Forensic Discoveries

1. **Pretraining Structural Trunk ($\mathcal{F}^{\text{structural}}$)**:
   - Ox Alpha exhibits exact tokenizer byte merges, vocabulary distribution, and special token serialization matching the **GLM tokenizer family** (Exploration 1).
2. **Sibling Branch Placement ($\mathcal{F}^{\text{cognitive}}$)**:
   - On the 17 cognitive coordinates independently proven to have shifted during the known $\text{GLM-5.2} \to \text{GLM-5.3}$ post-training update:
     - **Ox Alpha matches `GLM-5.2` (Base) on $14 / 17$ cells ($82.4\%$)**, matching `GLM-5.3` on only $3 / 17$ ($17.6\%$) and other on $0 / 17$ ($0.0\%$).
     - Replicated across **$7 \text{ vs } 1$ independent domain worlds** (Materials Science, Network Forensics, Astrophysics, Genomics).
     - **Signed Branch Index**: $B_{\text{acq}} = \mathbf{-0.65}$ (Strongly favors GLM-5.2 base developmental state).
     - Overall cognitive acquisition distance across all 144 shared holdout cells: $D_{\text{acq}}(\text{Ox}, \text{GLM-5.2}) = \mathbf{5.2\%}$ (5 / 96) vs $D_{\text{acq}}(\text{Ox}, \text{GLM-5.3}) = \mathbf{16.7\%}$ (16 / 96).
3. **Multimodal Vision Tokenization Geometry ($\mathcal{F}^{\text{vision}}$)**:
   - Empirical image scaling across 7 resolutions ($224\times224$ to $1344\times1344$) is an **exact 100% mathematical match (0 residual)** to the characteristic Z.ai GLM-V / CogVLM $28\times28$ patch grid formula:
     $$\text{Tokens}(W, H) = \left\lceil \frac{W}{28} \right\rceil \times \left\lceil \frac{H}{28} \right\rceil + 2$$
   - Both `stealth/ox-alpha` and `z-ai/glm-5v-turbo` enforce an identical **$112\times112$ spatial resolution floor ($16 + 2 = 18\text{ tokens}$)**, rejecting Qwen's $56\times56$ floor.
4. **Serving Layer Signature ($\mathcal{F}^{\text{serving}}$)**:
   - `stealth/ox-alpha` and `z-ai/glm-5.3` share the verbatim identical mandatory-reasoning validation error string:
     $$\texttt{\{"error":\{"message":"Reasoning is mandatory for this endpoint and cannot be disabled."\}\}}$$

---

## 2. Local Environment & Ollama Infrastructure

### Local Hardware Configuration
* **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM, CUDA 13.1, Driver 591.86).
* **Local Service**: Ollama daemon active on `http://127.0.0.1:11434` (OpenAI-compatible API at `http://127.0.0.1:11434/v1`).

### Downloaded Local Models
```bash
# Check local Ollama models:
python scratch/check_ollama_models.py
```
* **`qwen2.5:3b`** (1.80 GB) — Fast local reasoning control.
* **`qwen2.5-coder:3b`** (1.80 GB) — Specialized agent/code model.
* **`mistral:latest`** (4.07 GB) — General instruction baseline.
* **`gemma3:12b`** (7.59 GB) — Large local baseline.
* **`nomic-embed-text:latest`** (0.26 GB) — Embeddings.

### Useful Local Ollama Commands
```powershell
# Start Ollama service (if restarting machine):
ollama serve

# Pull additional models:
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b

# Test local inference:
python scratch/test_local_qwen.py
```

---

## 3. Background Tasks & Run Manifest

### Completed Golden Datasets
* `runs/20260821-204102-acquisition` — `stealth/ox-alpha` (144 decisions)
* `runs/20260821-232943-acquisition` — `z-ai/glm-5-turbo` (144 decisions, $D_{\text{acq}} = 4.2\%$ to Ox)
* `runs/20260821-223308-acquisition` — `z-ai/glm-5.2` (144 decisions, $D_{\text{acq}} = 5.2\%$ to Ox)
* `runs/20260821-225151-acquisition` — `z-ai/glm-5.3` (144 decisions, $D_{\text{acq}} = 16.7\%$ to Ox)
* `runs/20260821-214232-acquisition` — `poolside/laguna-s-2.1:free` (45 decisions, $D_{\text{acq}} = 10.7\%$ to Ox)
* `runs/20260821-211156-acquisition` — `openai/gpt-4o-mini` (127 decisions, $D_{\text{acq}} = 28.6\%$ to Ox)

---

## 4. Immediate Next Steps for Tomorrow

1. **Compile GLM-5-Turbo Lineage Output**:
   ```bash
   python oxford.py lineage-calibrate
   ```
   *Inspect whether GLM-5-Turbo shares Ox Alpha's cognitive acquisition logic ($D_{\text{acq}}$) and surface contract adherence.*

2. **Execute Exploration 4 (Normative Pressure Assay)**:
   * Run the 72-cell normative sweep on known candidates: `GLM-5.2`, `GLM-5.3`, `GLM-5-Turbo`, `GLM-5V-Turbo`, and local `Qwen-2.5-Coder`.
   * Screen discriminative cells using the four-part utility metric:
     $$\text{Utility}_i = \text{Separation}_i \times \text{OrderStability}_i \times \text{RepeatStability}_i \times \text{Interpretability}_i$$
   * Freeze the top 12–20 high-utility cells, then evaluate where `stealth/ox-alpha` falls on the frozen coordinates.

3. **Secondary Family-Recovery Control (`Laguna-XS 2.1`)**:
   * Once OpenRouter daily free rate limits reset, run `poolside/laguna-xs-2.1:free` to test whether Laguna S and Laguna XS cluster on family-associated traits where Ox Alpha does not.

---

## 5. Calibrated Epistemic Hierarchy & Attribution Status

| Evidence Tier | Confidence | Core Empirical Finding |
| :--- | :---: | :--- |
| **Tier 1: Pretraining Trunk** | **Definitive** | Exact GLM tokenizer byte merges, vocabulary distribution, and special token serialization (Exploration 1). |
| **Tier 2: Cognitive Basin** | **Strong** | Sits tightly inside the **`GLM-5.2 / GLM-5-Turbo` cognitive support basin** ($D_{\text{acq}} = 4.2\% - 5.2\%$), rejecting the public `GLM-5.3` post-training shift ($14/17 = 82.4\%$ match to 5.2 base on sibling-discordant cells, $B_{\text{acq}} = -0.65$). |
| **Tier 3: Downstream Turbo Traits** | **Suggestive** | Slightly nearer `GLM-5-Turbo` overall ($D_{\text{acq}} = 4.2\%$ vs $5.2\%$, 1-cell delta; $B_{\text{total}} = +0.27$), shared GLM-V $28\times28$ patch grid & $112\times112$ floor, and elevated surface lexical pull ($F_{\text{false}} = 45.8\% \leftrightarrow 29.2\%$). |
| **Tier 4: Builder / Lab Identity** | **Unresolved** | Unreleased internal Z.ai efficiency/multimodal sibling (`GLM-5V-Turbo/Flash`) vs. sophisticated third-party GLM-5.2 post-training derivative. |

---

## 6. Working Lineage Hypothesis Space

```
                                [GLM Pretraining Trunk]
                                (Byte Merges / Vocab Match)
                                           │
                                           ▼
                                [GLM-5.2 Base Checkpoint]
                                (Cognitive S(H) Core Logic)
                                           │
                    ┌──────────────────────┴──────────────────────────┐
                    │                                                 │
                    ▼                                                 ▼
          [Public GLM-5.2 Text API]              [Downstream Turbo / V Branches]
          • Dacq = 5.2% to Ox                                         │
          • Baseline serving stack               ┌────────────────────┼────────────────────┐
                    │                            │                    │                    │
                    ▼                            ▼                    ▼                    ▼
          [Public GLM-5.3 Text API]        [GLM-5-Turbo]      [GLM-5V-Turbo]     [Third-Party GLM-5.2]
          • Dacq = 16.7% to Ox             • Dacq = 4.2%      • 28x28 Vision     • Custom post-training
          • 14/17 discordant cells         • Agent tuned      • 112x112 Floor    • Custom serving
            favor 5.2 base                       │                    │                    │
                                                 └──────────────┬─────┴────────────────────┘
                                                                ▼
                                                       [stealth/ox-alpha ?]
```
