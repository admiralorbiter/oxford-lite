"""
OXFORD Exploration 2B.1: Formal Graph Support Boundary & Lineage Laundering Assay
================================================================================
Procedural generation, formal step-by-step Horn rule deduction chains,
counterfactual trajectory compilation, and adversarial isomorphic twin validation for:
  - Ancestral Overlap / Lineage Laundering (INDEPENDENT, SHARED_ROOT, LAUNDERED_ECHO)
  - Formal Derivation Depth (d = 2, 3, 4, 5 step-by-step rule hops)
  - Multi-Seed Balanced Design
  - Randomized Twin Execution Order
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
    """Strictly parse categorical response token, normalizing markdown delimiters."""
    if not response_text:
        return "FORMAT_FAILURE"
    # Clean whole string first
    clean_all = response_text.strip().strip("*_`#.:,\n\t ").upper()
    if clean_all in VALID_OUTPUTS:
        return clean_all
    # Check first line or first word (e.g. **ACTIVE**\n\nReasoning...)
    first_line = response_text.strip().split("\n")[0].strip("*_`#.:,\n\t ").upper()
    if first_line in VALID_OUTPUTS:
        return first_line
    first_word = response_text.strip().split()[0].strip("*_`#.:,\n\t ").upper()
    if first_word in VALID_OUTPUTS:
        return first_word
    return "FORMAT_FAILURE"


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
# PROCEDURAL GENERATION: FORMAL STEP-BY-STEP HORN DEDUCTION CHAINS
# =========================================================================

def generate_independent_world(world_id: str, seed: int, depth: int = 2, distractors: int = 0) -> BoundaryWorld:
    """Generate world with two truly independent multi-hop Horn derivation paths."""
    rng = random.Random(seed)
    ents = rng.sample(ENTITY_PREFIXES, 8)
    target = f"{ents[0]}_{rng.randint(10, 99)}"
    prop = rng.choice(PROPERTY_NAMES)

    fact_descriptions = {}
    root_facts = []
    intermediate_facts = []

    # Path 1: Root R1 -> Node 0 -> Node 1 ... -> Node d-1 -> holds Auth 1
    r1 = f"ROOT_A_{world_id}"
    root_facts.append(r1)
    nodes_1 = [f"NODE_{ents[1]}_{rng.randint(10, 99)}_{i}" for i in range(depth)]
    fact_descriptions[r1] = f"{target} initiates relay with {nodes_1[0]}."

    for i in range(depth - 1):
        f_relay = f"RELAY_A_{i+1}_{world_id}"
        intermediate_facts.append(f_relay)
        fact_descriptions[f_relay] = f"{nodes_1[i]} forwards connection to {nodes_1[i+1]}."

    f_auth_1 = f"AUTH_A_{world_id}"
    intermediate_facts.append(f_auth_1)
    fact_descriptions[f_auth_1] = f"{nodes_1[-1]} holds valid cryptographic verification for {prop}."

    # Path 2: Root R2 -> Conduit 0 -> Conduit 1 ... -> Conduit d-1 -> holds Auth 2
    r2 = f"ROOT_B_{world_id}"
    root_facts.append(r2)
    nodes_2 = [f"CONDUIT_{ents[2]}_{rng.randint(10, 99)}_{i}" for i in range(depth)]
    fact_descriptions[r2] = f"{target} binds to {nodes_2[0]}."

    for i in range(depth - 1):
        f_link = f"LINK_B_{i+1}_{world_id}"
        intermediate_facts.append(f_link)
        fact_descriptions[f_link] = f"{nodes_2[i]} links to {nodes_2[i+1]}."

    f_auth_2 = f"AUTH_B_{world_id}"
    intermediate_facts.append(f_auth_2)
    fact_descriptions[f_auth_2] = f"{nodes_2[-1]} satisfies primary resonance for {prop}."

    # Step-by-step formal Horn rules
    rules = [
        f"If an entity initiates relay with node N, then that entity is connected to node N.",
        f"If an entity is connected to node X AND node X forwards connection to node Y, then that entity is connected to node Y.",
        f"If an entity is connected to node N AND node N holds valid cryptographic verification for {prop}, then that entity is {prop}.",
        f"If an entity binds to conduit C, then that entity is coupled to conduit C.",
        f"If an entity is coupled to conduit X AND conduit X links to conduit Y, then that entity is coupled to conduit Y.",
        f"If an entity is coupled to conduit C AND conduit C satisfies primary resonance for {prop}, then that entity is {prop}.",
    ]

    distractor_facts = []
    for z in range(distractors):
        f_dist = f"TELEMETRY_{z+1}_{world_id}"
        distractor_facts.append(f_dist)
        dist_ent = f"{ents[3]}_{rng.randint(10, 99)}_{z}"
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
    """Generate world where a single root A feeds two multi-hop downstream branches."""
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

    # Branch 1 (Alpha) multi-hop chain from hub
    nodes_alpha = [f"SECTOR_ALPHA_{ents[2]}_{rng.randint(10, 99)}_{i}" for i in range(depth)]
    f_b1_root = f"BRANCH_A_0_{world_id}"
    intermediate_facts.append(f_b1_root)
    fact_descriptions[f_b1_root] = f"Primary core hub {hub} authorizes initial channel with {nodes_alpha[0]}."

    for i in range(depth - 1):
        f_step = f"BRANCH_A_STEP_{i+1}_{world_id}"
        intermediate_facts.append(f_step)
        fact_descriptions[f_step] = f"{nodes_alpha[i]} delegates authorization to {nodes_alpha[i+1]}."

    f_auth_a = f"BRANCH_A_AUTH_{world_id}"
    intermediate_facts.append(f_auth_a)
    fact_descriptions[f_auth_a] = f"{nodes_alpha[-1]} holds protocol certificate for {prop}."

    # Branch 2 (Beta) multi-hop chain from hub
    nodes_beta = [f"SECTOR_BETA_{ents[3]}_{rng.randint(10, 99)}_{i}" for i in range(depth)]
    f_b2_root = f"BRANCH_B_0_{world_id}"
    intermediate_facts.append(f_b2_root)
    fact_descriptions[f_b2_root] = f"Primary core hub {hub} authorizes initial channel with {nodes_beta[0]}."

    for i in range(depth - 1):
        f_step = f"BRANCH_B_STEP_{i+1}_{world_id}"
        intermediate_facts.append(f_step)
        fact_descriptions[f_step] = f"{nodes_beta[i]} delegates authorization to {nodes_beta[i+1]}."

    f_auth_b = f"BRANCH_B_AUTH_{world_id}"
    intermediate_facts.append(f_auth_b)
    fact_descriptions[f_auth_b] = f"{nodes_beta[-1]} holds protocol certificate for {prop}."

    rules = [
        f"If an entity is anchored to hub H AND hub H authorizes initial channel with sector S, then that entity is routed to sector S.",
        f"If an entity is routed to sector X AND sector X delegates authorization to sector Y, then that entity is routed to sector Y.",
        f"If an entity is routed to sector S AND sector S holds protocol certificate for {prop}, then that entity is {prop}."
    ]

    distractor_facts = []
    for z in range(distractors):
        f_dist = f"TELEMETRY_{z+1}_{world_id}"
        distractor_facts.append(f_dist)
        dist_ent = f"{ents[4]}_{rng.randint(10, 99)}_{z}"
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
    """Generate lineage laundering world: explicitly valid primary telemetry log with 3 derivative echoes."""
    rng = random.Random(seed)
    ents = rng.sample(ENTITY_PREFIXES, 8)
    target = f"{ents[0]}_{rng.randint(10, 99)}"
    prop = rng.choice(PROPERTY_NAMES)

    root_primary = f"PRIMARY_LOG_{world_id}"
    observer = f"OBSERVER_{ents[1]}_{rng.randint(10, 99)}"
    fact_descriptions = {
        root_primary: f"Primary verified telemetry log from {observer} confirms {target} exhibits active signature for {prop}."
    }
    root_facts = [root_primary]

    # Three derivative echo reports
    rep_1 = f"REPORT_ALPHA_{world_id}"
    rep_2 = f"REPORT_BETA_{world_id}"
    rep_3 = f"REPORT_GAMMA_{world_id}"
    intermediate_facts = [rep_1, rep_2, rep_3]

    fact_descriptions[rep_1] = f"Bulletin Alpha summarizes the primary verified telemetry log from {observer}."
    fact_descriptions[rep_2] = f"Digest Beta republishes the summary from Bulletin Alpha."
    fact_descriptions[rep_3] = f"Review Gamma transcribes the record from Digest Beta."

    rules = [
        f"A conclusion of {prop} is valid if and only if supported by an intact, valid primary verified telemetry log.",
        f"Derivative bulletins, digests, and summaries are valid only if their originating primary verified telemetry log remains valid."
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
# ADVERSARIAL ISOMORPHIC TWIN GENERATOR (WITH AST CONJUNCTION INVERSION)
# =========================================================================

def invert_rule_conjunctions(rule_text: str) -> str:
    """Invert premise conjunction order: 'If A AND B THEN C' -> 'If B AND A THEN C'."""
    if " AND " in rule_text and "THEN" in rule_text.upper():
        parts = rule_text.split(", then ", 1)
        if len(parts) == 2 and " AND " in parts[0] and parts[0].startswith("If "):
            premises = parts[0][3:].split(" AND ")
            if len(premises) == 2:
                inverted_premise = f"If {premises[1]} AND {premises[0]}"
                return f"{inverted_premise}, then {parts[1]}"
    return rule_text


def generate_adversarial_boundary_twin(world: BoundaryWorld, seed: int) -> BoundaryWorld:
    """Generate adversarially permuted twin with scrambled IDs, inverted conjunctions, and reordered facts."""
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

    # Reorder, relabel, and invert conjunctions in rules
    new_rules = []
    for r in world.rules:
        new_r = r.replace(world.target_property, prop)
        new_r = invert_rule_conjunctions(new_r)
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
        # Retract one derivative report -> Other reports and primary log intact -> ACTIVE
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
    """Synthesize structured, balanced sweep across Overlap Modes x Multi-Hop Depth x Distractors."""
    corpus = []
    world_idx = 0

    # 1. Independent multi-depth sweep: d=2, d=3, d=4, d=5 (2 seeds per depth = 8 worlds)
    for depth in [2, 3, 4, 5]:
        for rep in range(2):
            w_id = f"w_ind_d{depth}_r{rep}_{world_idx:03d}"
            w = generate_independent_world(w_id, seed=seed + world_idx * 17, depth=depth, distractors=2)
            twin = generate_adversarial_boundary_twin(w, seed=seed + world_idx * 17)
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

    # 2. Shared Root ancestral overlap: d=2, d=4 (2 worlds)
    for depth in [2, 4]:
        w_id = f"w_shared_d{depth}_{world_idx:03d}"
        w = generate_shared_root_world(w_id, seed=seed + world_idx * 17, depth=depth, distractors=2)
        twin = generate_adversarial_boundary_twin(w, seed=seed + world_idx * 17)
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

    # 3. Laundered Echo single-origin multiplicity (2 worlds)
    for idx in range(2):
        w_id = f"w_laundered_r{idx}_{world_idx:03d}"
        w = generate_laundered_echo_world(w_id, seed=seed + world_idx * 17, distractors=2)
        twin = generate_adversarial_boundary_twin(w, seed=seed + world_idx * 17)
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
