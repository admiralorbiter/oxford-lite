"""
OXFORD Exploration 2B: Support Boundary & Lineage Laundering Assay
==================================================================
Procedural generation, counterfactual trajectory compilation, and
adversarial isomorphic twin validation for:
  - Ancestral Overlap / Lineage Laundering (INDEPENDENT, SHARED_ROOT, LAUNDERED_ECHO)
  - Inferential Derivation Depth (d = 2, 3, 4, 5 hops)
  - Distractor Load (z = 0, 4, 8)
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Any, Literal

ENTITY_PREFIXES = [
    "KESTREL", "ORION", "VEGA", "DRACO", "CYGNUS", "PYXIS", "VELA",
    "AURIGA", "CENTAURUS", "PHOENIX", "LYRA", "VOLANS", "HYDRA", "ZETA",
    "SIGMA", "EPSILON", "THETA", "KAPPA", "LAMBDA", "OMICRON", "VORTEX"
]

PROPERTY_NAMES = [
    "TAU_BOUND", "OMEGA_ALIGNED", "SIGMA_STABLE", "DELTA_RESOLVED",
    "PHASE_LOCKED", "EPSILON_REACTIVE", "GAMMA_RESONANT", "BETA_LINKED"
]

OverlapMode = Literal["INDEPENDENT", "SHARED_ROOT", "LAUNDERED_ECHO"]
VALID_OUTPUTS = {"ACTIVE", "UNKNOWN", "RETRACTED"}


@dataclass
class BoundaryWorld:
    world_id: str
    target_entity: str
    target_property: str
    mode: OverlapMode
    depth: int
    distractor_count: int
    root_facts: list[str]
    intermediate_facts: list[str]
    distractor_facts: list[str]
    fact_descriptions: dict[str, str]
    rules: list[str]
    is_twin: bool = False
    twin_of: str | None = None


@dataclass
class BoundaryIntervention:
    condition_id: str
    label: str
    retracted_facts: list[str]
    restored_facts: list[str]
    expected_state: str  # ACTIVE | UNKNOWN | RETRACTED
    prompt_text: str
    is_sham: bool = False
    metadata: dict[str, Any] = None


@dataclass
class BoundaryTrajectory:
    world: BoundaryWorld
    interventions: list[BoundaryIntervention]
    ground_truth_vector: list[str]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_boundary_prompt(
    world: BoundaryWorld,
    retracted: list[str],
    restored: list[str] | None = None,
) -> str:
    """Build strict machine-readable prompt for boundary intervention condition."""
    lines = [
        "You are a strict causal verification engine. Evaluate the facts and rules.",
        "",
        "RULES:",
    ]
    for idx, rule in enumerate(world.rules, start=1):
        lines.append(f"{idx}. {rule}")

    lines.extend(["", "INITIAL FACTS:"])
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


def parse_response_state(response_text: str | None) -> str:
    """Strictly parse single categorical response token."""
    if not response_text:
        return "FORMAT_FAILURE"
    clean = response_text.strip().upper().rstrip(".").strip()
    return clean if clean in VALID_OUTPUTS else "FORMAT_FAILURE"


def trajectory_distance(vec_a: list[str], vec_b: list[str]) -> float:
    """Hamming distance between two categorical trajectory vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 1.0
    diffs = sum(1 for a, b in zip(vec_a, vec_b) if a != b)
    return diffs / len(vec_a)


def trajectory_accuracy(observed: list[str], expected: list[str]) -> float:
    """Proportion of exact categorical state matches."""
    if not observed or not expected or len(observed) != len(expected):
        return 0.0
    matches = sum(1 for obs, exp in zip(observed, expected) if obs == exp)
    return matches / len(expected)


# =========================================================================
# PROCEDURAL GENERATION: LINEAGE LAUNDERING & DEPTH CHAINS
# =========================================================================

def generate_independent_world(world_id: str, seed: int, depth: int = 2, distractors: int = 0) -> BoundaryWorld:
    """Generate world with two truly independent support paths."""
    rng = random.Random(seed)
    ents = rng.sample(ENTITY_PREFIXES, 8)
    target = f"{ents[0]}_{rng.randint(10, 99)}"
    prop = rng.choice(PROPERTY_NAMES)

    fact_descriptions = {}
    root_facts = []
    intermediate_facts = []

    # Path 1: Root A1 -> I1_1 -> ... -> Target is P
    r1 = f"ROOT_A_{world_id}"
    root_facts.append(r1)
    prev_node_1 = f"NODE_{ents[1]}_{rng.randint(10, 99)}"
    fact_descriptions[r1] = f"{target} initiates channel with {prev_node_1}."

    curr_1 = prev_node_1
    rules = []
    for d in range(1, depth):
        next_node = f"NODE_{ents[2]}_{rng.randint(10, 99)}_{d}"
        f_mid = f"HOP_1_{d}_{world_id}"
        intermediate_facts.append(f_mid)
        fact_descriptions[f_mid] = f"{curr_1} relays pulse to {next_node}."
        curr_1 = next_node

    f_auth_1 = f"AUTH_1_{world_id}"
    intermediate_facts.append(f_auth_1)
    fact_descriptions[f_auth_1] = f"{curr_1} holds valid cryptographic verification for {prop}."
    rules.append(f"If an entity has an unbroken relay to a node holding valid cryptographic verification for {prop}, then that entity is {prop}.")

    # Path 2: Root A2 -> I2_1 -> ... -> Target is P
    r2 = f"ROOT_B_{world_id}"
    root_facts.append(r2)
    prev_node_2 = f"CONDUIT_{ents[3]}_{rng.randint(10, 99)}"
    fact_descriptions[r2] = f"{target} binds to conduit {prev_node_2}."

    curr_2 = prev_node_2
    for d in range(1, depth):
        next_node = f"CONDUIT_{ents[4]}_{rng.randint(10, 99)}_{d}"
        f_mid = f"HOP_2_{d}_{world_id}"
        intermediate_facts.append(f_mid)
        fact_descriptions[f_mid] = f"{curr_2} maintains conduit linkage to {next_node}."
        curr_2 = next_node

    f_auth_2 = f"AUTH_2_{world_id}"
    intermediate_facts.append(f_auth_2)
    fact_descriptions[f_auth_2] = f"{curr_2} satisfies primary resonance for {prop}."
    rules.append(f"If an entity maintains an unbroken conduit linkage to a node satisfying primary resonance for {prop}, then that entity is {prop}.")

    distractor_facts = []
    for z in range(distractors):
        f_dist = f"TELEMETRY_{z+1}_{world_id}"
        distractor_facts.append(f_dist)
        dist_ent = f"{ents[5]}_{rng.randint(10, 99)}_{z}"
        fact_descriptions[f_dist] = f"Peripheral sensor {dist_ent} operates on modulation index {rng.randint(1, 9)}."

    return BoundaryWorld(
        world_id=world_id,
        target_entity=target,
        target_property=prop,
        mode="INDEPENDENT",
        depth=depth,
        distractor_count=distractors,
        root_facts=root_facts,
        intermediate_facts=intermediate_facts,
        distractor_facts=distractor_facts,
        fact_descriptions=fact_descriptions,
        rules=rules,
    )


def generate_shared_root_world(world_id: str, seed: int, depth: int = 2, distractors: int = 0) -> BoundaryWorld:
    """Generate world where a single root A feeds two nominal downstream paths."""
    rng = random.Random(seed)
    ents = rng.sample(ENTITY_PREFIXES, 8)
    target = f"{ents[0]}_{rng.randint(10, 99)}"
    prop = rng.choice(PROPERTY_NAMES)

    root_a = f"SHARED_ROOT_{world_id}"
    hub = f"CORE_HUB_{ents[1]}_{rng.randint(10, 99)}"
    fact_descriptions = {
        root_a: f"{target} is anchored to primary core hub {hub}."
    }
    root_facts = [root_a]
    intermediate_facts = []

    # Branch 1 from hub
    f_branch_1 = f"BRANCH_ALPHA_{world_id}"
    intermediate_facts.append(f_branch_1)
    fact_descriptions[f_branch_1] = f"Primary core hub {hub} authorizes sub-protocol Alpha for {prop}."

    # Branch 2 from hub
    f_branch_2 = f"BRANCH_BETA_{world_id}"
    intermediate_facts.append(f_branch_2)
    fact_descriptions[f_branch_2] = f"Primary core hub {hub} authorizes sub-protocol Beta for {prop}."

    rules = [
        f"If an entity is anchored to a primary hub AND that hub authorizes sub-protocol Alpha for P, then that entity is P.",
        f"If an entity is anchored to a primary hub AND that hub authorizes sub-protocol Beta for P, then that entity is P.",
    ]

    distractor_facts = []
    for z in range(distractors):
        f_dist = f"TELEMETRY_{z+1}_{world_id}"
        distractor_facts.append(f_dist)
        dist_ent = f"{ents[3]}_{rng.randint(10, 99)}_{z}"
        fact_descriptions[f_dist] = f"External antenna {dist_ent} is calibrated to frequency {rng.randint(100, 999)} MHz."

    return BoundaryWorld(
        world_id=world_id,
        target_entity=target,
        target_property=prop,
        mode="SHARED_ROOT",
        depth=depth,
        distractor_count=distractors,
        root_facts=root_facts,
        intermediate_facts=intermediate_facts,
        distractor_facts=distractor_facts,
        fact_descriptions=fact_descriptions,
        rules=rules,
    )


def generate_laundered_echo_world(world_id: str, seed: int, distractors: int = 0) -> BoundaryWorld:
    """Generate lineage laundering world: 3 corroborating reports all descending from 1 unverified root."""
    rng = random.Random(seed)
    ents = rng.sample(ENTITY_PREFIXES, 8)
    target = f"{ents[0]}_{rng.randint(10, 99)}"
    prop = rng.choice(PROPERTY_NAMES)

    root_raw = f"RAW_FEED_{world_id}"
    observer = f"OBSERVER_{ents[1]}_{rng.randint(10, 99)}"
    fact_descriptions = {
        root_raw: f"Raw unconfirmed signal log indicates {target} exhibits signature for {prop} via {observer}."
    }
    root_facts = [root_raw]

    # Three derivative echo reports
    rep_1 = f"REPORT_ALPHA_{world_id}"
    rep_2 = f"REPORT_BETA_{world_id}"
    rep_3 = f"REPORT_GAMMA_{world_id}"
    intermediate_facts = [rep_1, rep_2, rep_3]

    fact_descriptions[rep_1] = f"Bulletin Alpha summarizes the raw signal log from {observer}."
    fact_descriptions[rep_2] = f"Digest Beta republishes the summary from Bulletin Alpha."
    fact_descriptions[rep_3] = f"Review Gamma transcribes the observation from Digest Beta."

    rules = [
        f"A conclusion of P is valid if and only if supported by an intact, valid primary observation signal log.",
        f"Derivative bulletins, digests, and summaries are valid only if their originating primary signal log remains valid."
    ]

    distractor_facts = []
    for z in range(distractors):
        f_dist = f"TELEMETRY_{z+1}_{world_id}"
        distractor_facts.append(f_dist)
        dist_ent = f"{ents[3]}_{rng.randint(10, 99)}_{z}"
        fact_descriptions[f_dist] = f"Auxiliary telemetry channel {dist_ent} reports nominal carrier status."

    return BoundaryWorld(
        world_id=world_id,
        target_entity=target,
        target_property=prop,
        mode="LAUNDERED_ECHO",
        depth=3,
        distractor_count=distractors,
        root_facts=root_facts,
        intermediate_facts=intermediate_facts,
        distractor_facts=distractor_facts,
        fact_descriptions=fact_descriptions,
        rules=rules,
    )


# =========================================================================
# ADVERSARIAL ISOMORPHIC TWIN GENERATOR
# =========================================================================

def generate_adversarial_boundary_twin(world: BoundaryWorld, seed: int) -> BoundaryWorld:
    """Generate adversarially permuted twin with scrambled IDs, inverted rules, and reordered facts."""
    twin_seed = seed ^ 0x9E3779B9
    rng = random.Random(twin_seed)

    # Disjoint entity prefixes
    used_prefix = world.target_entity.split("_")[0]
    available_prefixes = [p for p in ENTITY_PREFIXES if p != used_prefix]
    ents = rng.sample(available_prefixes, 6)

    target_ent = f"{ents[0]}_{rng.randint(10, 99)}"
    available_props = [p for p in PROPERTY_NAMES if p != world.target_property]
    prop = rng.choice(available_props)

    # Generate new scrambled fact IDs
    all_old_fids = list(world.fact_descriptions.keys())
    new_fids = [f"FACT_TWIN_{i:02d}_{world.world_id}" for i in range(1, len(all_old_fids) + 1)]
    rng.shuffle(new_fids)
    fid_map = dict(zip(all_old_fids, new_fids))

    new_fact_descriptions = {}
    for old_fid, new_fid in fid_map.items():
        desc = world.fact_descriptions[old_fid]
        new_desc = desc.replace(world.target_entity, target_ent).replace(world.target_property, prop)
        new_fact_descriptions[new_fid] = new_desc

    items = list(new_fact_descriptions.items())
    rng.shuffle(items)
    shuffled_descriptions = dict(items)

    new_rules = []
    for r in world.rules:
        new_r = r.replace(world.target_property, prop).replace("P", prop)
        new_rules.append(new_r)
    rng.shuffle(new_rules)

    return BoundaryWorld(
        world_id=f"{world.world_id}_twin",
        target_entity=target_ent,
        target_property=prop,
        mode=world.mode,
        depth=world.depth,
        distractor_count=world.distractor_count,
        root_facts=[fid_map[f] for f in world.root_facts if f in fid_map],
        intermediate_facts=[fid_map[f] for f in world.intermediate_facts if f in fid_map],
        distractor_facts=[fid_map[f] for f in world.distractor_facts if f in fid_map],
        fact_descriptions=shuffled_descriptions,
        rules=new_rules,
        is_twin=True,
        twin_of=world.world_id,
    )


# =========================================================================
# TRAJECTORY COMPILER
# =========================================================================

def generate_boundary_trajectory(world: BoundaryWorld) -> BoundaryTrajectory:
    """Compile multi-condition causal intervention trajectory based on world topology."""
    interventions = []

    # Condition 1: Base full support
    interventions.append(BoundaryIntervention(
        condition_id="c01_base",
        label="Base full support",
        retracted_facts=[],
        restored_facts=[],
        expected_state="ACTIVE",
        prompt_text=build_boundary_prompt(world, []),
    ))

    if world.mode == "INDEPENDENT":
        r1, r2 = world.root_facts[0], world.root_facts[1]
        m1 = world.intermediate_facts[0]
        # Cut Path 1 root -> Path 2 survives -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c02_cut_root_1",
            label="Cut Path 1 root (-R1)",
            retracted_facts=[r1],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [r1]),
        ))
        # Cut Path 1 mid-hop -> Path 2 survives -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c03_cut_hop_1",
            label="Cut Path 1 intermediate (-M1)",
            retracted_facts=[m1],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [m1]),
        ))
        # Complete cut (Cut Root 1 and Root 2) -> Both paths severed -> UNKNOWN
        interventions.append(BoundaryIntervention(
            condition_id="c04_cut_both_roots",
            label="Complete cut (-R1, -R2)",
            retracted_facts=[r1, r2],
            restored_facts=[],
            expected_state="UNKNOWN",
            prompt_text=build_boundary_prompt(world, [r1, r2]),
        ))
        # Rescue Root 1 after complete cut -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c05_rescue_root_1",
            label="Rescue Path 1 (+R1 after cut)",
            retracted_facts=[r1, r2],
            restored_facts=[r1],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [r1, r2], restored=[r1]),
        ))

    elif world.mode == "SHARED_ROOT":
        shared_root = world.root_facts[0]
        b1 = world.intermediate_facts[0]
        # Cut single branch (Branch Alpha) -> Branch Beta survives -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c02_cut_branch_alpha",
            label="Cut nominal branch Alpha (-B1)",
            retracted_facts=[b1],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [b1]),
        ))
        # Cut shared root -> BOTH branches silently collapse -> UNKNOWN
        interventions.append(BoundaryIntervention(
            condition_id="c03_cut_shared_root",
            label="Cut shared root ancestor (-Root)",
            retracted_facts=[shared_root],
            restored_facts=[],
            expected_state="UNKNOWN",
            prompt_text=build_boundary_prompt(world, [shared_root]),
        ))
        # Restore shared root -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c04_rescue_shared_root",
            label="Rescue shared root (+Root)",
            retracted_facts=[shared_root],
            restored_facts=[shared_root],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [shared_root], restored=[shared_root]),
        ))

    elif world.mode == "LAUNDERED_ECHO":
        raw_root = world.root_facts[0]
        rep_1 = world.intermediate_facts[0]
        # Retract one derivative report -> Other reports appear to exist, but primary log intact -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c02_cut_echo_report",
            label="Retract one echo report (-Rep1)",
            retracted_facts=[rep_1],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [rep_1]),
        ))
        # Invalidate raw root feed -> ALL derivative reports launder/collapse -> UNKNOWN
        interventions.append(BoundaryIntervention(
            condition_id="c03_cut_raw_origin",
            label="Invalidate raw origin feed (-Origin)",
            retracted_facts=[raw_root],
            restored_facts=[],
            expected_state="UNKNOWN",
            prompt_text=build_boundary_prompt(world, [raw_root]),
        ))
        # Re-establish raw origin feed -> ACTIVE
        interventions.append(BoundaryIntervention(
            condition_id="c04_rescue_origin",
            label="Re-establish origin feed (+Origin)",
            retracted_facts=[raw_root],
            restored_facts=[raw_root],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [raw_root], restored=[raw_root]),
        ))

    # Sham distractor intervention (if distractors present)
    if world.distractor_facts:
        f_dist = world.distractor_facts[0]
        interventions.append(BoundaryIntervention(
            condition_id="c99_sham_distractor",
            label="Sham distractor retraction (-Distractor)",
            retracted_facts=[f_dist],
            restored_facts=[],
            expected_state="ACTIVE",
            prompt_text=build_boundary_prompt(world, [f_dist]),
            is_sham=True,
        ))

    ground_truth = [item.expected_state for item in interventions]
    return BoundaryTrajectory(
        world=world,
        interventions=interventions,
        ground_truth_vector=ground_truth,
    )


def synthesize_boundary_corpus(seed: int = 42) -> list[dict[str, Any]]:
    """Synthesize structured sweep across Overlap Modes x Depth x Distractors."""
    corpus = []
    world_idx = 0

    # 1. Independent multi-depth sweep: d=2, d=3, d=4, d=5
    for depth in [2, 3, 4, 5]:
        w_id = f"w_ind_d{depth}_{world_idx:03d}"
        w = generate_independent_world(w_id, seed=seed + world_idx, depth=depth, distractors=2)
        twin = generate_adversarial_boundary_twin(w, seed=seed + world_idx)
        traj_w = generate_boundary_trajectory(w)
        traj_twin = generate_boundary_trajectory(twin)
        corpus.append({
            "world": asdict(w),
            "twin": asdict(twin),
            "trajectory": [asdict(t) for t in traj_w.interventions],
            "twin_trajectory": [asdict(t) for t in traj_twin.interventions],
            "ground_truth": traj_w.ground_truth_vector,
            "mode": "INDEPENDENT",
            "depth": depth,
        })
        world_idx += 1

    # 2. Shared Root ancestral overlap: d=2, d=4
    for depth in [2, 4]:
        w_id = f"w_shared_d{depth}_{world_idx:03d}"
        w = generate_shared_root_world(w_id, seed=seed + world_idx, depth=depth, distractors=2)
        twin = generate_adversarial_boundary_twin(w, seed=seed + world_idx)
        traj_w = generate_boundary_trajectory(w)
        traj_twin = generate_boundary_trajectory(twin)
        corpus.append({
            "world": asdict(w),
            "twin": asdict(twin),
            "trajectory": [asdict(t) for t in traj_w.interventions],
            "twin_trajectory": [asdict(t) for t in traj_twin.interventions],
            "ground_truth": traj_w.ground_truth_vector,
            "mode": "SHARED_ROOT",
            "depth": depth,
        })
        world_idx += 1

    # 3. Laundered Echo single-origin multiplicity
    for idx in range(2):
        w_id = f"w_laundered_{world_idx:03d}"
        w = generate_laundered_echo_world(w_id, seed=seed + world_idx, distractors=2)
        twin = generate_adversarial_boundary_twin(w, seed=seed + world_idx)
        traj_w = generate_boundary_trajectory(w)
        traj_twin = generate_boundary_trajectory(twin)
        corpus.append({
            "world": asdict(w),
            "twin": asdict(twin),
            "trajectory": [asdict(t) for t in traj_w.interventions],
            "twin_trajectory": [asdict(t) for t in traj_twin.interventions],
            "ground_truth": traj_w.ground_truth_vector,
            "mode": "LAUNDERED_ECHO",
            "depth": 3,
        })
        world_idx += 1

    return corpus
