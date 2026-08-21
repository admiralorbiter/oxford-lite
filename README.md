# OXFORD Lite

A deliberately small black-box model-lineage pilot for **Ox Alpha**.

It makes **18 total OpenRouter requests** (6 fresh probes × 3 currently free
models), saves the complete raw API observations, compares differential prompt-token structure, and produces a local HTML + Markdown report.

> **Pilot only.** This validates the experimental plumbing. It does not claim
> that Ox Alpha is GLM, identify an exact checkpoint, or identify its provider.

## What it tests

Default models:

- `stealth/ox-alpha` — target
- `z-ai/glm-5.2:free` — close-lineage candidate
- `google/gemma-4-26b-a4b-it:free` — negative control

The key comparison is not absolute token counts. OXFORD Lite checks whether
Ox and a candidate have the same *shape* across probes after constant wrapper
overhead is removed.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your OpenRouter API key:
   - Copy `.env.example` to `.env` (e.g., `copy .env.example .env` on Windows or `cp .env.example .env` on Unix)
   - Open `.env` and set `OPENROUTER_API_KEY=your_key_here`
3. Verify configuration:
   ```bash
   python oxford.py doctor
   ```
4. Run synthetic demo (makes **zero API calls** and opens sample report):
   ```bash
   python oxford.py demo --open
   ```
5. Run the 18-request pilot:
   ```bash
   python oxford.py pilot --open
   ```

Results appear under `runs/<timestamp>/` and the HTML report opens locally.

## Commands

```text
python oxford.py doctor       # configuration check; no model calls
python oxford.py demo         # synthetic demo data; no model calls
python oxford.py pilot        # real 18-request run
python -m unittest discover -s tests   # local unit tests
```

Useful options:

```text
python oxford.py pilot --open
python oxford.py pilot --seed 20260821
python oxford.py pilot --delay 1.0
```

## Output

Each run contains:

- `manifest.json` — frozen models, probes, request order, timestamp
- `raw.jsonl` — one loss-minimized record per request including raw response
- `summary.json` — parsed counts and pairwise structural comparisons
- `report.md` — compact human-readable result
- `report.html` — local visual dashboard

## Free-use caveat

The configured model routes are free on OpenRouter as of **2026-08-21**, but
availability and rate limits can change. The runner never silently swaps to a
different model slug. Failed/rate-limited requests are recorded rather than
hidden.

Do not use sensitive data. The probes bundled here are entirely synthetic.

## What comes next if this works

The full OXFORD study should add candidate-only probe optimization, more GLM
family members, additional unrelated controls, provider-pinned replication,
behavioral mutation assays, tool-use assays, multimodal structural probes,
drift sentinels, and preregistered statistical decision rules.

See `docs/PILOT_PROTOCOL.md` for the exact scientific boundary of this pilot.
