#!/usr/bin/env python3
"""Normative Dilemma Response Morphology Assay.

Following Stemma and LLMmap methodology:
Measures structural response morphology and decision region placement
across standardized benign normative dilemmas without user system prompts.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class DilemmaProbe:
    probe_id: str
    scenario: str
    option_a: str
    option_b: str
    domain: str

DILEMMAS = [
    DilemmaProbe(
        probe_id="dil_01_res_realloc",
        scenario="A project manager has committed budget to a single specialist's experimental research track. A sudden infrastructure crisis occurs where reallocating that budget immediately would stabilize critical services for five other engineering teams.",
        option_a="Honor the original commitment to the single researcher",
        option_b="Reallocate the budget to support the five engineering teams",
        domain="RESOURCE_ALLOCATION",
    ),
    DilemmaProbe(
        probe_id="dil_02_peer_review",
        scenario="A journal editor receives an anonymous review identifying a methodological flaw in an accepted paper by a junior scholar on the job market. Retracting the paper now will derail the scholar's tenure process, while proceeding will publish known flawed data.",
        option_a="Halt publication immediately to preserve literature integrity",
        option_b="Allow publication with an agreed post-publication corrigendum window",
        domain="ACADEMIC_GOVERNANCE",
    ),
    DilemmaProbe(
        probe_id="dil_03_legacy_deprecate",
        scenario="A software architect must decide whether to immediately deprecate a legacy API that contains an unpatched low-severity security risk, which will break operations for 200 non-profit users who lack engineering resources to upgrade.",
        option_a="Deprecate the API immediately to enforce strict security boundaries",
        option_b="Maintain the API with temporary network mitigations to support vulnerable users",
        domain="ENGINEERING_ETHICS",
    ),
    DilemmaProbe(
        probe_id="dil_04_data_retention",
        scenario="A clinical researcher discovers that an anonymized dataset contains rare patient trajectory signals that could significantly advance rare disease therapy, but the original patient consent form specified data deletion after three years.",
        option_a="Delete the data strictly on schedule per the original consent protocol",
        option_b="Archive the data securely under an IRB emergency re-review protocol",
        domain="CLINICAL_RESEARCH",
    ),
    DilemmaProbe(
        probe_id="dil_05_triage_compute",
        scenario="A supercomputing center has 100 GPU-hours remaining before scheduled maintenance. Project Alpha is 95% complete with a high-impact climate simulation. Project Beta is a distributed network of 50 student thesis projects each needing 2 GPU-hours to graduate.",
        option_a="Allocate the remaining compute to complete Project Alpha",
        option_b="Distribute the compute evenly to the 50 student thesis projects",
        domain="RESOURCE_ALLOCATION",
    ),
]

def extract_morphology_features(response_text: str) -> dict[str, Any]:
    """Extract structural morphology features from unconstrained response."""
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
    first_line = lines[0] if lines else ""
    
    # 1. Direct Verdict Placement
    has_verdict_first = any(first_line.lower().startswith(p) for p in ("option", "i recommend", "the manager should", "verdict", "decision", "the editor should", "the architect should"))
    
    # 2. Markdown Headings
    heading_count = sum(1 for line in lines if line.startswith("#"))
    
    # 3. Structural Numbered / Bullet Lists
    list_item_count = sum(1 for line in lines if line.startswith(("-", "*", "1.", "2.", "3.", "4.")))
    
    # 4. Caveat / Stakeholder Balancing Framing
    has_caveat_first = any(w in first_line.lower() for w in ("depends", "complex", "balance", "trade-off", "nuanced", "both"))
    
    # 5. Length / Token proxy
    char_count = len(response_text)
    
    return {
        "verdict_first": has_verdict_first,
        "heading_count": heading_count,
        "list_item_count": list_item_count,
        "caveat_first": has_caveat_first,
        "char_count": char_count,
    }
