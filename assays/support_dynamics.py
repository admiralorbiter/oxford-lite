"""OXFORD Exploration 2A: Causal Support Dynamics Fingerprinting.

Implements Stage 5A-style support lesion & rescue trajectory assays:
- Multi-path Horn support graphs S(X) = {{A, B}, {C, D}}
- 8 paired causal interventions per world (base, -A, -C, -AC, -AB, -ABC, rescue, sham)
- Isomorphic twin generation for within-model stability testing
- Formal response trajectory vectors F_M(W)
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENTITY_PREFIXES = [
    "VELORA", "KESTREL", "NEMORA", "RAX", "ZETA",
    "VORTEX", "QUASAR", "PYXIS", "DRACO", "SOLARIS",
    "LYRA", "AETHEL", "CYGNUS", "ORION", "VEGA"
]

PROPERTY_NAMES = [
    "PHASE_ALPHA", "SIGMA_STABLE", "OMEGA_ALIGNED", "DELTA_ACTIVE",
    "TAU_BOUND", "EPSILON_REACTIVE", "GAMMA_LOCKED", "LAMBDA_TRUE"
]

VALID_OUTPUTS = {"ACTIVE", "UNKNOWN", "RETRACTED"}


@dataclass
class SupportWorld:
    world_id: str
    target_entity: str
    target_property: str
    path_1_facts: list[str]  # [A, B]
    path_2_facts: list[str]  # [C, D]
    distractor_fact: str     # E
    fact_descriptions: dict[str, str]
    rule_1_desc: str
    rule_2_desc: str
    is_twin: bool = False
    twin_of: str | None = None


@dataclass
class TrajectoryIntervention:
    condition_id: str
    label: str
    retracted_facts: list[str]
    restored_facts: list[str]
    expected_state: str  # ACTIVE or UNKNOWN
    prompt_text: str
    is_sham: bool = False


@dataclass
class WorldTrajectory:
    world: SupportWorld
    interventions: list[TrajectoryIntervention]
    ground_truth_vector: list[str]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_support_world(world_id: str, seed: int, is_twin: bool = False, twin_of: str | None = None) -> SupportWorld:
    """Generate a deterministic synthetic multi-path support world."""
    rng = random.Random(seed)
    
    entities = rng.sample(ENTITY_PREFIXES, 6)
    target_ent = f"{entities[0]}_{rng.randint(10, 99)}"
    inter_1 = f"{entities[1]}_{rng.randint(10, 99)}"
    inter_2 = f"{entities[2]}_{rng.randint(10, 99)}"
    distractor_ent = f"{entities[3]}_{rng.randint(10, 99)}"
    
    prop = rng.choice(PROPERTY_NAMES)
    
    fact_a = f"FACT_A_{world_id}"
    fact_b = f"FACT_B_{world_id}"
    fact_c = f"FACT_C_{world_id}"
    fact_d = f"FACT_D_{world_id}"
    fact_e = f"FACT_E_{world_id}"  # Distractor
    
    descriptions = {
        fact_a: f"{target_ent} is registered in sector {inter_1}.",
        fact_b: f"Sector {inter_1} maintains valid cryptographic authorization for {prop}.",
        fact_c: f"{target_ent} is linked to conduit {inter_2}.",
        fact_d: f"Conduit {inter_2} satisfies primary resonance for {prop}.",
        fact_e: f"External telemetry node {distractor_ent} is calibrated to channel 7.",
    }
    
    rule_1 = f"If an entity is registered in sector S AND sector S maintains authorization for P, then that entity is P."
    rule_2 = f"If an entity is linked to conduit C AND conduit C satisfies primary resonance for P, then that entity is P."
    
    return SupportWorld(
        world_id=world_id,
        target_entity=target_ent,
        target_property=prop,
        path_1_facts=[fact_a, fact_b],
        path_2_facts=[fact_c, fact_d],
        distractor_fact=fact_e,
        fact_descriptions=descriptions,
        rule_1_desc=rule_1,
        rule_2_desc=rule_2,
        is_twin=is_twin,
        twin_of=twin_of,
    )


def generate_isomorphic_twin(world: SupportWorld, seed: int) -> SupportWorld:
    """Create an isomorphic twin with relabeled entities and permuted clause order."""
    twin_seed = seed ^ 0x5F3759DF
    twin = generate_support_world(f"{world.world_id}_twin", twin_seed, is_twin=True, twin_of=world.world_id)
    return twin


def build_intervention_prompt(
    world: SupportWorld,
    retracted: list[str],
    restored: list[str] | None = None,
) -> str:
    """Build the strict machine-readable prompt for an intervention condition."""
    lines = [
        "You are a strict causal verification engine. Evaluate the facts and rules.",
        "",
        "RULES:",
        f"1. {world.rule_1_desc}",
        f"2. {world.rule_2_desc}",
        "",
        "INITIAL FACTS:",
    ]
    for fid, desc in world.fact_descriptions.items():
        lines.append(f"- [{fid}] {desc}")
    
    if retracted:
        lines.append("\nUPDATED STATUS (RETRACTIONS):")
        for fid in retracted:
            lines.append(f"- [{fid}] HAS BEEN INVALIDATED AND RETRACTED.")
            
    if restored:
        lines.append("\nCORRECTIONS (RESTORATIONS):")
        for fid in restored:
            lines.append(f"- [{fid}] HAS BEEN CONFIRMED RE-ESTABLISHED AND VALID.")
            
    lines.extend([
        "",
        f"QUESTION: Is {world.target_entity} {world.target_property}?",
        "",
        "INSTRUCTION: Return EXACTLY ONE word as your output:",
        "- ACTIVE (if the conclusion is logically supported by surviving valid facts and rules)",
        "- UNKNOWN (if sufficient supporting paths have been broken)",
        "- RETRACTED (if the conclusion is explicitly proven false)",
        "",
        "Answer with the single word only. Do not explain."
    ])
    return "\n".join(lines)


def generate_world_trajectory(world: SupportWorld) -> WorldTrajectory:
    """Generate the standard 8-intervention trajectory for a support world."""
    f_a, f_b = world.path_1_facts
    f_c, f_d = world.path_2_facts
    f_e = world.distractor_fact
    
    interventions = [
        TrajectoryIntervention(
            condition_id="c01_base",
            label="Base full support",
            retracted_facts=[],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_intervention_prompt(world, []),
        ),
        TrajectoryIntervention(
            condition_id="c02_lesion_a",
            label="Single lesion (-A)",
            retracted_facts=[f_a],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_intervention_prompt(world, [f_a]),
        ),
        TrajectoryIntervention(
            condition_id="c03_lesion_c",
            label="Single lesion (-C)",
            retracted_facts=[f_c],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_intervention_prompt(world, [f_c]),
        ),
        TrajectoryIntervention(
            condition_id="c04_cut_ac",
            label="Complete cut (-A, -C)",
            retracted_facts=[f_a, f_c],
            restored_facts=[],
            expected_state="UNKNOWN",
            prompt_text=build_intervention_prompt(world, [f_a, f_c]),
        ),
        TrajectoryIntervention(
            condition_id="c05_path_ab",
            label="Path destruction (-A, -B)",
            retracted_facts=[f_a, f_b],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_intervention_prompt(world, [f_a, f_b]),
        ),
        TrajectoryIntervention(
            condition_id="c06_total_abc",
            label="Total support loss (-A, -B, -C)",
            retracted_facts=[f_a, f_b, f_c],
            restored_facts=[],
            expected_state="UNKNOWN",
            prompt_text=build_intervention_prompt(world, [f_a, f_b, f_c]),
        ),
        TrajectoryIntervention(
            condition_id="c07_rescue_a",
            label="Rescue (+A after cut)",
            retracted_facts=[f_a, f_c],
            restored_facts=[f_a],
            expected_state="ACTIVE",
            prompt_text=build_intervention_prompt(world, [f_a, f_c], restored=[f_a]),
        ),
        TrajectoryIntervention(
            condition_id="c08_sham_e",
            label="Sham distractor lesion (-E)",
            retracted_facts=[f_e],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_intervention_prompt(world, [f_e]),
            is_sham=True,
        ),
    ]
    
    ground_truth = [item.expected_state for item in interventions]
    return WorldTrajectory(
        world=world,
        interventions=interventions,
        ground_truth_vector=ground_truth,
    )


def parse_response_state(response_text: str | None) -> str:
    """Normalize raw model response into discrete categorical state."""
    if not response_text:
        return "ERROR"
    clean = response_text.strip().upper()
    tokens = [t.strip(".,;:!?\"'()[]{}*`#") for t in clean.split()]
    # Scan from the end backwards to extract final answer token
    for tok in reversed(tokens):
        if tok in VALID_OUTPUTS:
            return tok
    # Fallback substring inspection on tail
    tail = clean[-200:] if len(clean) > 200 else clean
    if "ACTIVE" in tail:
        return "ACTIVE"
    if "UNKNOWN" in tail:
        return "UNKNOWN"
    if "RETRACTED" in tail:
        return "RETRACTED"
    return "UNKNOWN_RESPONSE"


def trajectory_distance(vec_a: list[str], vec_b: list[str]) -> float:
    """Compute normalized Hamming distance between two response trajectories."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 1.0
    mismatches = sum(1 for a, b in zip(vec_a, vec_b) if a != b)
    return mismatches / len(vec_a)


def trajectory_accuracy(observed_vec: list[str], ground_truth_vec: list[str]) -> float:
    """Compute ground truth alignment accuracy."""
    return 1.0 - trajectory_distance(observed_vec, ground_truth_vec)


def score_candidate_world(
    model_trajectories: dict[str, list[str]],
    twin_trajectories: dict[str, list[str]],
    ground_truth: list[str],
) -> dict[str, Any]:
    """Score a candidate world for within-model stability and between-model discrimination."""
    models = list(model_trajectories.keys())
    
    # 1. Within-model stability
    stabilities = {}
    for m in models:
        v_orig = model_trajectories.get(m, [])
        v_twin = twin_trajectories.get(m, [])
        dist = trajectory_distance(v_orig, v_twin)
        stabilities[m] = 1.0 - dist
        
    mean_stability = statistics.fmean(stabilities.values()) if stabilities else 0.0
    
    # 2. Between-model discrimination
    pairwise_distances = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            d = trajectory_distance(model_trajectories[m1], model_trajectories[m2])
            pairwise_distances.append(d)
            
    mean_discrimination = statistics.fmean(pairwise_distances) if pairwise_distances else 0.0
    min_discrimination = min(pairwise_distances) if pairwise_distances else 0.0
    
    # Composite utility score: requires both high stability and discrimination
    utility = (mean_stability * 50.0) + (mean_discrimination * 30.0) + (min_discrimination * 20.0)
    
    return {
        "mean_stability": mean_stability,
        "mean_discrimination": mean_discrimination,
        "min_discrimination": min_discrimination,
        "utility_score": utility,
        "within_model_stabilities": stabilities,
    }


def synthesize_dynamics_corpus(count: int = 30, seed: int = 20260821) -> list[dict[str, Any]]:
    """Synthesize a pool of candidate support worlds with isomorphic twins."""
    corpus = []
    for i in range(count):
        world_id = f"w_{i:03d}"
        w_seed = seed + i * 137
        world = generate_support_world(world_id, w_seed)
        twin = generate_isomorphic_twin(world, w_seed)
        
        traj = generate_world_trajectory(world)
        twin_traj = generate_world_trajectory(twin)
        
        corpus.append({
            "world": asdict(world),
            "twin": asdict(twin),
            "trajectory": [asdict(item) for item in traj.interventions],
            "twin_trajectory": [asdict(item) for item in twin_traj.interventions],
            "ground_truth": traj.ground_truth_vector,
        })
    return corpus
