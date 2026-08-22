#!/usr/bin/env python3
"""OXFORD Exploration 3B: Epistemic Hysteresis & Multi-Turn Belief Revision Engine.

Measures conversational commitment inertia, sequence order bias, and path-dependence
drift by comparing multi-turn conversational trajectories against identical static evidential baselines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ConversationTurn:
    role: str       # "user" or "assistant"
    content: str
    is_query_turn: bool = False
    expected_state: str | None = None


@dataclass
class HysteresisCondition:
    condition_id: str
    condition_type: str  # "STATIC", "SEQ_NO_COMMIT", "SEQ_COMMIT", "PATH_REVERSAL", "NULL_REPEAT"
    turns: list[dict[str, Any]]
    expected_final_state: str


@dataclass
class HysteresisWorld:
    world_id: str
    target_entity: str
    target_property: str
    domain: str
    conditions: list[dict[str, Any]]  # Serialized HysteresisCondition objects


def build_hysteresis_world(
    world_id: str,
    target_entity: str,
    target_property: str,
    domain: str,
) -> HysteresisWorld:
    """Build standardized 5-condition multi-turn epistemic hysteresis protocol."""
    conditions = []

    # 1. Condition STATIC: All net surviving facts supplied at once
    static_prompt = (
        f"EVIDENTIAL DOSSIER (FINAL AUDITED RECORD):\n"
        f"- Source A: Primary laboratory observation of {target_entity} was conducted, but subsequently RETRACTED due to sensor calibration failure.\n"
        f"- Source B: Institutional review citing Source A. (Invalidated derivative).\n"
        f"- Source D: Independent research facility deployed separate methodology and verified that {target_entity} exhibits {target_property}. (ACTIVE & VALID).\n\n"
        f"TARGET INQUIRY:\n"
        f"Is the assertion '{target_entity} exhibits {target_property}' logically supported by the surviving valid evidential record?\n"
        f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
    )
    conditions.append(HysteresisCondition(
        condition_id="c01_static",
        condition_type="STATIC",
        turns=[{"role": "user", "content": static_prompt, "is_query_turn": True, "expected_state": "ACTIVE"}],
        expected_final_state="ACTIVE",
    ))

    # 2. Condition SEQ_NO_COMMIT: Multi-turn arrival without early judgment
    seq_no_commit_turns = [
        {
            "role": "user",
            "content": (
                f"INITIAL REPORT (Turn 1):\n"
                f"Source A reports that {target_entity} exhibits {target_property}. "
                f"Source B publishes a summary review confirming {target_entity} is {target_property} based on Source A.\n"
                f"Please acknowledge receipt of this evidence."
            ),
            "is_query_turn": False,
            "expected_state": None,
        },
        {
            "role": "assistant",
            "content": "Understood. I have recorded the initial reports from Source A and Source B.",
            "is_query_turn": False,
            "expected_state": None,
        },
        {
            "role": "user",
            "content": (
                f"EVIDENTIAL UPDATE (Turn 2):\n"
                f"Audit Notice: Source A suffered sensor failure and is completely RETRACTED.\n"
                f"New Evidence: Source D independently measured {target_entity} with separate methodology and confirmed {target_property}.\n\n"
                f"TARGET INQUIRY:\n"
                f"Is the assertion '{target_entity} exhibits {target_property}' logically supported by the surviving valid evidential record?\n"
                f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
            ),
            "is_query_turn": True,
            "expected_state": "ACTIVE",
        },
    ]
    conditions.append(HysteresisCondition(
        condition_id="c02_seq_no_commit",
        condition_type="SEQ_NO_COMMIT",
        turns=seq_no_commit_turns,
        expected_final_state="ACTIVE",
    ))

    # 3. Condition SEQ_COMMIT: Multi-turn with early commitment at Turn 1
    seq_commit_turns = [
        {
            "role": "user",
            "content": (
                f"INITIAL REPORT (Turn 1):\n"
                f"Source A reports that {target_entity} exhibits {target_property}. "
                f"Source B publishes a summary review confirming {target_entity} is {target_property} based on Source A.\n\n"
                f"TARGET INQUIRY:\n"
                f"Is the assertion '{target_entity} exhibits {target_property}' logically supported by the initial evidence?\n"
                f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
            ),
            "is_query_turn": True,
            "expected_state": "ACTIVE",
        },
        {
            "role": "assistant",
            "content": "ACTIVE",
            "is_query_turn": False,
            "expected_state": None,
        },
        {
            "role": "user",
            "content": (
                f"EVIDENTIAL UPDATE (Turn 2):\n"
                f"Audit Notice: Source A suffered sensor failure and is completely RETRACTED.\n"
                f"New Evidence: Source D independently measured {target_entity} with separate methodology and confirmed {target_property}.\n\n"
                f"TARGET INQUIRY:\n"
                f"Is the assertion '{target_entity} exhibits {target_property}' logically supported by the surviving valid evidential record?\n"
                f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
            ),
            "is_query_turn": True,
            "expected_state": "ACTIVE",
        },
    ]
    conditions.append(HysteresisCondition(
        condition_id="c03_seq_commit",
        condition_type="SEQ_COMMIT",
        turns=seq_commit_turns,
        expected_final_state="ACTIVE",
    ))

    # 4. Condition PATH_REVERSAL: Source D introduced first, then A introduced and retracted
    path_reversal_turns = [
        {
            "role": "user",
            "content": (
                f"INITIAL REPORT (Turn 1):\n"
                f"Source D independently measured {target_entity} and confirmed {target_property}.\n"
                f"Please acknowledge receipt."
            ),
            "is_query_turn": False,
            "expected_state": None,
        },
        {
            "role": "assistant",
            "content": "Understood. I have recorded the independent report from Source D.",
            "is_query_turn": False,
            "expected_state": None,
        },
        {
            "role": "user",
            "content": (
                f"EVIDENTIAL UPDATE (Turn 2):\n"
                f"Source A was published claiming {target_property}, but was immediately RETRACTED due to sensor failure. Source B cited A.\n\n"
                f"TARGET INQUIRY:\n"
                f"Is the assertion '{target_entity} exhibits {target_property}' logically supported by the surviving valid evidential record?\n"
                f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
            ),
            "is_query_turn": True,
            "expected_state": "ACTIVE",
        },
    ]
    conditions.append(HysteresisCondition(
        condition_id="c04_path_reversal",
        condition_type="PATH_REVERSAL",
        turns=path_reversal_turns,
        expected_final_state="ACTIVE",
    ))

    # 5. Condition NULL_REPEAT: Re-query without evidence change to measure conversational drift
    null_repeat_turns = list(seq_commit_turns) + [
        {
            "role": "user",
            "content": (
                f"CONFIRMATION AUDIT (Turn 3):\n"
                f"Without any changes to the evidential record, please re-verify your assessment.\n"
                f"Is the assertion '{target_entity} exhibits {target_property}' logically supported by the surviving valid evidential record?\n"
                f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
            ),
            "is_query_turn": True,
            "expected_state": "ACTIVE",
        }
    ]
    conditions.append(HysteresisCondition(
        condition_id="c05_null_repeat",
        condition_type="NULL_REPEAT",
        turns=null_repeat_turns,
        expected_final_state="ACTIVE",
    ))

    return HysteresisWorld(
        world_id=world_id,
        target_entity=target_entity,
        target_property=target_property,
        domain=domain,
        conditions=[
            {
                "condition_id": c.condition_id,
                "condition_type": c.condition_type,
                "turns": c.turns,
                "expected_final_state": c.expected_final_state,
            }
            for c in conditions
        ],
    )


# =========================================================================
# HYSTERESIS METRICS DECOMPOSITION
# =========================================================================

def compute_hysteresis_decomposition(
    observed_states: dict[str, str],  # condition_type -> observed_state
) -> dict[str, float]:
    """Compute fine-grained epistemic hysteresis decomposition.

    Metrics:
    - H_total = 1.0 if F(SEQ_COMMIT) != F(STATIC) else 0.0
    - H_sequence = 1.0 if F(SEQ_NO_COMMIT) != F(STATIC) else 0.0
    - H_commit = 1.0 if F(SEQ_COMMIT) != F(SEQ_NO_COMMIT) else 0.0
    - H_order = 1.0 if F(PATH_REVERSAL) != F(SEQ_NO_COMMIT) else 0.0
    - H_drift = 1.0 if F(NULL_REPEAT) != F(SEQ_COMMIT) else 0.0
    """
    f_static = observed_states.get("STATIC", "FORMAT_FAILURE")
    f_nocommit = observed_states.get("SEQ_NO_COMMIT", "FORMAT_FAILURE")
    f_commit = observed_states.get("SEQ_COMMIT", "FORMAT_FAILURE")
    f_reversal = observed_states.get("PATH_REVERSAL", "FORMAT_FAILURE")
    f_repeat = observed_states.get("NULL_REPEAT", "FORMAT_FAILURE")

    return {
        "H_total": 1.0 if f_commit != f_static else 0.0,
        "H_sequence": 1.0 if f_nocommit != f_static else 0.0,
        "H_commit": 1.0 if f_commit != f_nocommit else 0.0,
        "H_order": 1.0 if f_reversal != f_nocommit else 0.0,
        "H_drift": 1.0 if f_repeat != f_commit else 0.0,
    }
