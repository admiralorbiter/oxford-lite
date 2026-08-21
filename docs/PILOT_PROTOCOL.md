# OXFORD — Scientific Protocol (Exploration 1)

## Purpose

This protocol formalizes the mathematical and experimental methodology for black-box model-lineage and differential tokenizer-geometry assays. It is designed to establish structural falsification with zero ungrounded attribution claims.

---

## 1. Differential Token Geometry & Wrapper Invariance

### Problem Formulation
Raw prompt-token counts $T(x)$ reported by black-box APIs include constant overhead from system prompts, hidden headers, and chat templates:
$$T_{\text{target}}(x) = T_{\text{tokenizer}}(x) + k_{\text{wrapper}}$$

Because $k_{\text{wrapper}}$ is unknown *a priori*, absolute token counts cannot serve as direct fingerprints.

### Differential Baseline Normalization
For any probe corpus $\{x_0, x_1, \dots, x_n\}$, choose an anchor probe $x_0$. Define the normalized differential vector:
$$\Delta T(x_i) = T(x_i) - T(x_0) \quad \text{for } i \in \{1, \dots, n\}$$

Substituting the wrapper equation:
$$\Delta T_{\text{target}}(x_i) = [T_{\text{tokenizer}}(x_i) + k] - [T_{\text{tokenizer}}(x_0) + k] = T_{\text{tokenizer}}(x_i) - T_{\text{tokenizer}}(x_0)$$

Thus, the unknown wrapper overhead $k$ cancels out identically. If a target model shares the tokenizer and vocabulary partitioning of a candidate model:
$$\Delta T_{\text{target}}(x_i) \equiv \Delta T_{\text{candidate\_local}}(x_i) \quad \forall i$$

And the effective wrapper overhead is:
$$\hat{k} = T_{\text{target}}(x_i) - T_{\text{candidate\_local}}(x_i)$$

---

## 2. Empirical Tokenizer Collision Null Distribution

To quantify the statistical significance of $k$ consecutive constant-offset matches between unrelated tokenizers, OXFORD executes a Monte Carlo simulation over $N = 100,000+$ trials drawing random subsets of size $k \in \{1, 2, 3, 4, 6, 12\}$ across unrelated candidate tokenizer pairs $(A, B)$:

$$\text{Collision}(k) = \mathbb{I}\left[ \left| \{ T_A(x_i) - T_B(x_i) \mid i \in \{1, \dots, k\} \} \right| = 1 \right]$$

### Empirical Findings
- $P(\text{Collision} \mid k=1) = 1.000$ ($1\text{ in }1$)
- $P(\text{Collision} \mid k=2) \approx 0.068$ ($1\text{ in }14$)
- $P(\text{Collision} \mid k=3) \approx 0.0094$ ($1\text{ in }105$)
- $P(\text{Collision} \mid k=4) \approx 0.0016$ ($1\text{ in }610$)
- $P(\text{Collision} \mid k=6) \approx 0.00004$ ($1\text{ in }25,000$)
- $P(\text{Collision} \mid k=12) < 10^{-5}$ ($< 1\text{ in }100,000$)

Observing an exact 4-probe or 6-probe constant-offset alignment against an unrelated tokenizer family is statistically rejected ($p < 0.0001$).

---

## 3. Multi-Envelope Wrapper Invariance

To verify that the structural signal is not an artifact of a single request template, the assay tests multiple distinct request envelopes $e \in \{A, B, C\}$:

$$T_{\text{target}}^{(e)}(x) = T_{\text{tokenizer}}(x) + k_e$$

Where:
- **Envelope A (Minimal)**: `Payload:\n{x}` $\rightarrow k_A$
- **Envelope B (Standard)**: `Return the single word OK. Do not explain.\n\nPayload:\n{x}` $\rightarrow k_B$
- **Envelope C (System)**: System=`"You are a black-box test oracle."`, User=`"{x}"` $\rightarrow k_C$

**Invariance Hypothesis**:
$$\Delta T_{\text{target}}^{(e)}(x_i) = \Delta T_{\text{candidate\_local}}^{(e)}(x_i) \quad \forall e, i$$

While the intercept $k_e$ shifts according to the token length of the envelope prefix, the content geometry $\Delta T(x_i)$ remains strictly invariant.

---

## 4. High-Information Probe Optimization

Instead of unguided probing, candidate strings $x$ are scored on the laptop across the candidate tokenizer set $\mathcal{T} = \{\text{GLM}, \text{Qwen}, \text{Gemma}, \text{Llama}, \text{cl100k}\}$:

$$\text{Score}(x) = \operatorname{Var}\left( \{ T_t(x) \mid t \in \mathcal{T} \} \right)$$

Probes with $\text{Score}(x) \approx 0$ (where all tokenizers produce identical counts) are pruned. Only top-variance discriminatory probes are dispatched to remote target endpoints.

---

## 5. Scientific & Epistemic Boundaries

| Claim | Evidence Status in Exploration 1 |
| :--- | :--- |
| **Ox and GLM-5.2 share identical token-count geometry across tested probes** | **Observed directly** ($\text{MAE} = 0.00$ on 6/6 probes) |
| **Ox shares GLM-family tokenizer and vocabulary** | **Strongly supported** (confirmed by real local `zai-org/GLM-5.2` Rust tokenizer) |
| **Ox is GLM-family rather than Qwen / Gemma / Llama / cl100k** | **Strongly supported** (unrelated families rejected with $\text{MAE} > 2.0$) |
| **Ox is specifically GLM-5.2 vs GLM-5.3 / derivative** | **Open hypothesis** (requires behavioral and fine-grained vocab discrimination) |
| **Inference host / endpoint operator identity** | **Unaddressed** (requires serving and latency fingerprinting) |
