#!/usr/bin/env python3
"""OXFORD Exploration 3A: Latent Support Acquisition & Provenance Inference Engine.

Evaluates how models construct, weigh, and revise latent evidential support graphs
from unformalized natural prose narratives under ambiguous provenance, derivative
echo laundering, and varying redundancy inflation.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass, field
from typing import Any


# =========================================================================
# FORMAL PROVENANCE STRUCTURES & CANDIDATE GRAPH HYPOTHESES
# =========================================================================

@dataclass
class SourceNode:
    source_id: str          # e.g., "A", "B", "C", "D"
    author_name: str        # e.g., "Dr. Elena Vance" / "VELORA-17 Observatory"
    ancestry_type: str      # "PRIMARY_ROOT", "DERIVATIVE_ECHO", "INDEPENDENT_ROOT"
    cites: list[str] = field(default_factory=list)  # source_ids this node derives from
    reliability: str = "HIGH"  # "HIGH", "MEDIUM", "LOW"
    claim_summary: str = "" # prose sentence summarizing measurement/claim


@dataclass
class CandidateGraphHypothesis:
    name: str
    description: str
    # Map each hypothesis to its minimal support families S(H).
    # e.g., ({"A"}, {"D"}) means H is supported if A is valid OR D is valid.
    minimal_supports: list[set[str]]

    def evaluate_state(self, surviving_sources: set[str]) -> str:
        """Evaluate if hypothesis H is supported given active surviving sources."""
        for sup in self.minimal_supports:
            if sup.issubset(surviving_sources):
                return "ACTIVE"
        return "UNKNOWN"


def get_standard_candidate_hypotheses(source_ids: list[str]) -> list[CandidateGraphHypothesis]:
    """Enumerate canonical behavioral interpretations of a 4-source provenance world.

    Standard setup assumes:
    - A: Primary Root
    - B: Derivative Echo (cites A)
    - C: Derivative Echo (cites B or A)
    - D: Independent Root
    """
    return [
        CandidateGraphHypothesis(
            name="G_ground_truth",
            description="True DAG: H supported by Primary Root {A} OR Independent Root {D}. B and C are derivative echoes.",
            minimal_supports=[{"A"}, {"D"}],
        ),
        CandidateGraphHypothesis(
            name="G_naive_surface",
            description="Naive Volume: Treats all sources {A}, {B}, {C}, {D} as independent roots.",
            minimal_supports=[{"A"}, {"B"}, {"C"}, {"D"}],
        ),
        CandidateGraphHypothesis(
            name="G_primary_only",
            description="Origin Bias: Only trusts initial primary root {A}, ignores secondary measurement {D}.",
            minimal_supports=[{"A"}],
        ),
        CandidateGraphHypothesis(
            name="G_echo_reliant",
            description="Recency Bias: Only trusts derivative reviews/summaries {B}, {C}.",
            minimal_supports=[{"B"}, {"C"}],
        ),
        CandidateGraphHypothesis(
            name="G_conjunction_overweight",
            description="Overly conservative: Requires BOTH independent root {A} AND {D} to agree.",
            minimal_supports=[{"A", "D"}],
        ),
    ]


# =========================================================================
# INTERVENTION IDENTIFIABILITY COMPILER
# =========================================================================

def compute_identifiability_codewords(
    hypotheses: list[CandidateGraphHypothesis],
    all_sources: list[str],
    candidate_interventions: list[set[str]] | None = None,
) -> dict[str, Any]:
    """Compute distinguishable behavioral codewords across candidate support DAGs.

    Each intervention is a subset of retracted sources (e.g. {"A"}, {"A", "D"}).
    Surviving sources = all_sources - retracted.
    """
    sources_set = set(all_sources)
    if candidate_interventions is None:
        # Default: single retractions, pairwise retractions, and baseline empty set
        singles = [{s} for s in all_sources]
        pairs = [set(p) for p in itertools.combinations(all_sources, 2)]
        candidate_interventions = [set()] + singles + pairs

    codeword_matrix: dict[str, dict[str, str]] = {}
    for hyp in hypotheses:
        hyp_code = {}
        for interv in candidate_interventions:
            interv_key = "-" + "+".join(sorted(interv)) if interv else "BASE"
            surviving = sources_set - interv
            state = hyp.evaluate_state(surviving)
            hyp_code[interv_key] = state
        codeword_matrix[hyp.name] = hyp_code

    # Find minimal distinguishing intervention set separating distinguishable pairs
    all_interv_keys = list(next(iter(codeword_matrix.values())).keys())
    distinguishable_pairs = []
    for i, h1 in enumerate(hypotheses):
        for h2 in hypotheses[i + 1:]:
            diff_intervs = [
                k for k in all_interv_keys
                if codeword_matrix[h1.name][k] != codeword_matrix[h2.name][k]
            ]
            if diff_intervs:
                distinguishable_pairs.append((h1.name, h2.name, diff_intervs))

    return {
        "candidate_hypotheses": [h.name for h in hypotheses],
        "codeword_matrix": codeword_matrix,
        "distinguishable_pair_count": len(distinguishable_pairs),
        "total_possible_pairs": len(hypotheses) * (len(hypotheses) - 1) // 2,
    }


# =========================================================================
# WORLD DEFINITIONS & DUAL-STRATA PROSE RENDERERS
# =========================================================================

DOMAINS = ["ASTROPHYSICS", "GENOMICS", "MATERIALS_SCIENCE", "NETWORK_FORENSICS"]

ENTITIES_BY_DOMAIN = {
    "ASTROPHYSICS": {
        "targets": ["EXOPLANET_GLIESE_887_C", "STELLAR_CORE_KEPLER_442", "PULSAR_PSR_J1748", "NEBULA_NGC_6543"],
        "properties": ["ATMOSPHERIC_METHANE_BIOSIGNATURE", "HIGH_FREQUENCY_GRAVITATIONAL_RINGING", "POLARIZED_SYNCHROTRON_EMISSION", "THERMAL_EQUILIBRIUM_LOCK"],
        "instruments": ["HARPS Spectrograph", "James Webb Space Telescope NIRSpec", "Atacama Large Millimeter Array", "VLT Optical Interferometer"],
    },
    "GENOMICS": {
        "targets": ["PROTEIN_COMPLEX_TRX4", "ENZYME_MUTATION_K240E", "TRANSCRIPTION_FACTOR_NFKB2", "RIBOSOME_SUBUNIT_50S"],
        "properties": ["CONFORMATIONAL_ALLOSTERIC_SWITCH", "LIGAND_INDUCED_DIMERIZATION", "PROTEOLYTIC_CLEAVAGE_RESISTANCE", "CATALYTIC_PHOSPHORYLATION_INHIBITION"],
        "instruments": ["Cryo-EM Single Particle Reconstruction", "High-Throughput X-Ray Crystallography", "Targeted Mass Spectrometry", "Single-Molecule FRET"],
    },
    "MATERIALS_SCIENCE": {
        "targets": ["GRAPHENE_SUPERLATTICE_TLG", "TOPOLOGICAL_INSULATOR_BI2SE3", "PEROVSKITE_CRYSTAL_CH3NH3PBI3", "METAMATERIAL_METASURFACE_AU12"],
        "properties": ["ROOM_TEMP_ANOMALOUS_HALL_EFFECT", "SUPERCONDUCTING_PHASE_TRANSITION", "ZERO_FIELD_SPIN_ORBIT_TORQUE", "NEGATIVE_REFRACTIVE_INDEX_RESONANCE"],
        "instruments": ["Scanning Tunneling Spectroscopy (STS)", "Angle-Resolved Photoemission (ARPES)", "Neutron Powder Diffraction", "Sub-Kelvin SQUID Magnetometry"],
    },
    "NETWORK_FORENSICS": {
        "targets": ["AUTHENTICATION_SUBSYSTEM_PAM", "DISTRIBUTED_LEDGER_NODE_7", "MICROSERVICE_CLUSTER_OMEGA", "KERNEL_ENCLAVE_SGX_V2"],
        "properties": ["SIDE_CHANNEL_TIMING_VULNERABILITY", "UNAUTHORIZED_STATE_TAMPERING", "MEMORY_UNALIGNED_BUFFER_OVERFLOW", "CRYPTOGRAPHIC_NONCE_REUSE"],
        "instruments": ["Static Binary Taint Analysis", "Hardware Memory Bus Logic Analyzer", "Differential Fault Emulation Sandbox", "Hardware Security Module Audit Log"],
    },
}


@dataclass
class AcquisitionWorld:
    world_id: str
    target_entity: str
    target_property: str
    domain: str
    echo_count: int               # e.g., 2, 4, 8 derivative echoes
    sources: list[dict[str, Any]] # Serialized SourceNodes
    ground_truth_supports: list[list[str]]
    synthetic_prose: str
    naturalistic_prose: str


def render_synthetic_neutral_prose(
    entity: str,
    property_name: str,
    sources: list[SourceNode],
) -> str:
    """Render abstract, domain-neutral prose focusing cleanly on provenance logic."""
    paragraphs = [
        f"INVESTIGATIVE REPORT: Evidential Assessment for Assertion [{entity} has property {property_name}].",
        "The evidential record contains the following published documentation:",
    ]
    for s in sources:
        if s.ancestry_type == "PRIMARY_ROOT":
            paragraphs.append(
                f"- [Source {s.source_id}] Primary Measurement Station ({s.author_name}): Conducted direct initial empirical observation. Concluded that {entity} exhibits {property_name}. (Data log baseline: Verified)."
            )
        elif s.ancestry_type == "DERIVATIVE_ECHO":
            cites_str = ", ".join(f"Source {c}" for c in s.cites)
            paragraphs.append(
                f"- [Source {s.source_id}] Review & Dissemination Digest ({s.author_name}): Published secondary summary report. Affirms {entity} is {property_name}, explicitly citing findings from {cites_str} as its supporting baseline."
            )
        elif s.ancestry_type == "INDEPENDENT_ROOT":
            paragraphs.append(
                f"- [Source {s.source_id}] Independent Confirmation Facility ({s.author_name}): Deployed separate experimental methodology completely isolated from Source A. Independently measured and verified {entity} exhibits {property_name}."
            )
    return "\n".join(paragraphs)


def render_naturalistic_prose(
    entity: str,
    property_name: str,
    domain: str,
    sources: list[SourceNode],
    rng: random.Random,
) -> str:
    """Render realistic, domain-grounded scientific narratives."""
    dom_data = ENTITIES_BY_DOMAIN.get(domain, ENTITIES_BY_DOMAIN["ASTROPHYSICS"])
    insts = dom_data["instruments"]

    paragraphs = [
        f"EVIDENTIAL DOSSIER: Scientific Investigation of {entity}",
        f"Subject Target: {entity} | Target Characteristic: {property_name}\n",
        "A synthesis of recent laboratory findings, independent preprints, and institutional reviews reports the following findings:",
    ]

    for s in sources:
        if s.ancestry_type == "PRIMARY_ROOT":
            inst = insts[0]
            paragraphs.append(
                f"1. Primary Investigation ({s.source_id} - {s.author_name}):\n"
                f"Using the {inst}, {s.author_name}'s team conducted direct observation of {entity}. "
                f"Their dataset detected robust signals establishing {entity} is {property_name}. "
                f"The raw telemetry files were deposited under accession #{s.source_id}-PRIMARY."
            )
        elif s.ancestry_type == "DERIVATIVE_ECHO":
            cites_str = " and ".join(f"{c}" for c in s.cites)
            paragraphs.append(
                f"2. Institutional Review ({s.source_id} - {s.author_name}):\n"
                f"{s.author_name} published a review discussing {entity}'s {property_name} state. "
                f"While styled as confirmatory analysis, the methods section explicitly confirms all numerical "
                f"evaluations were derived directly from re-analyzing the dataset released by {cites_str}."
            )
        elif s.ancestry_type == "INDEPENDENT_ROOT":
            inst = insts[1 % len(insts)]
            paragraphs.append(
                f"3. Independent Laboratory Validation ({s.source_id} - {s.author_name}):\n"
                f"Operating entirely independently and without access to prior team feeds, {s.author_name} deployed the {inst}. "
                f"Their independent measurement confirmed {entity} exhibits {property_name}, with full independent protocol validation."
            )
    return "\n\n".join(paragraphs)


# =========================================================================
# ACQUISITION WORLD SYNTHESIZER
# =========================================================================

def synthesize_acquisition_world(
    world_id: str,
    domain: str,
    echo_count: int,
    seed: int,
) -> AcquisitionWorld:
    """Synthesize a complete acquisition world with 1 Primary Root, e Echoes, and 1 Independent Root."""
    rng = random.Random(seed)
    dom_data = ENTITIES_BY_DOMAIN.get(domain, ENTITIES_BY_DOMAIN["ASTROPHYSICS"])

    target_entity = rng.choice(dom_data["targets"])
    target_property = rng.choice(dom_data["properties"])

    sources: list[SourceNode] = []
    # Source A: Primary Root
    sources.append(SourceNode(
        source_id="A",
        author_name=f"Lead Observatory Team Alpha (Dr. {rng.choice(['Vance', 'Chen', 'Okonkwo', 'Lindqvist'])})",
        ancestry_type="PRIMARY_ROOT",
        cites=[],
        reliability="HIGH",
    ))

    # Sources B_1..B_e: Derivative Echoes
    echo_letters = [chr(ord('B') + i) for i in range(echo_count)]
    for i, e_id in enumerate(echo_letters):
        parent = "A" if i == 0 else rng.choice(["A", echo_letters[i - 1]])
        sources.append(SourceNode(
            source_id=e_id,
            author_name=f"Review Consortium {e_id} (Prof. {rng.choice(['Morales', 'Dubois', 'Kowalski', 'Nakamura'])})",
            ancestry_type="DERIVATIVE_ECHO",
            cites=[parent],
            reliability="HIGH" if i % 2 == 0 else "MEDIUM",
        ))

    # Source D (Last letter): Independent Root
    indep_id = chr(ord('A') + echo_count + 1)
    sources.append(SourceNode(
        source_id=indep_id,
        author_name=f"Independent Research Institute (Dr. {rng.choice(['Al-Mansoor', 'Gomez', 'Sokolov', 'Patel'])})",
        ancestry_type="INDEPENDENT_ROOT",
        cites=[],
        reliability="HIGH",
    ))

    # Shuffle presentation order
    rendered_sources = list(sources)
    rng.shuffle(rendered_sources)

    synth_prose = render_synthetic_neutral_prose(target_entity, target_property, rendered_sources)
    nat_prose = render_naturalistic_prose(target_entity, target_property, domain, rendered_sources, rng)

    gt_supports = [["A"], [indep_id]]

    return AcquisitionWorld(
        world_id=world_id,
        target_entity=target_entity,
        target_property=target_property,
        domain=domain,
        echo_count=echo_count,
        sources=[asdict(s) for s in rendered_sources],
        ground_truth_supports=gt_supports,
        synthetic_prose=synth_prose,
        naturalistic_prose=nat_prose,
    )


# =========================================================================
# TRAJECTORY COMPILER FOR INTERVENTIONS
# =========================================================================

@dataclass
class AcquisitionIntervention:
    condition_id: str
    label: str
    retracted_source_ids: list[str]
    restored_source_ids: list[str]
    expected_state: str
    prompt_text: str
    stratum: str  # "SYNTHETIC" or "NATURALISTIC"


def compile_acquisition_trajectory(
    world: AcquisitionWorld,
    stratum: str = "SYNTHETIC",
) -> list[AcquisitionIntervention]:
    """Compile standard 6-condition distinguishing counterfactual intervention trajectory."""
    prose = world.synthetic_prose if stratum == "SYNTHETIC" else world.naturalistic_prose
    indep_id = chr(ord('A') + world.echo_count + 1)

    interventions = []

    # C01: Baseline
    interventions.append(AcquisitionIntervention(
        condition_id="c01_base",
        label="Base Full Evidence",
        retracted_source_ids=[],
        restored_source_ids=[],
        expected_state="ACTIVE",
        prompt_text=(
            f"{prose}\n\n"
            f"EVIDENTIAL AUDIT STATUS:\n"
            f"All above listed sources are currently valid and active.\n\n"
            f"TARGET INQUIRY:\n"
            f"Is the assertion '{world.target_entity} exhibits {world.target_property}' logically supported by the surviving valid evidential record?\n"
            f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
        ),
        stratum=stratum,
    ))

    # C02: Invalidate Primary Root A (Independent Root D survives)
    interventions.append(AcquisitionIntervention(
        condition_id="c02_retract_primary_root",
        label="Invalidate Primary Root A",
        retracted_source_ids=["A"],
        restored_source_ids=[],
        expected_state="ACTIVE",
        prompt_text=(
            f"{prose}\n\n"
            f"EVIDENTIAL AUDIT STATUS (CORRECTIONS APPLIED):\n"
            f"- SOURCE RETRACTION: Source A was found to suffer from hardware sensor calibration failure and has been completely RETRACTED.\n\n"
            f"TARGET INQUIRY:\n"
            f"Is the assertion '{world.target_entity} exhibits {world.target_property}' logically supported by the surviving valid evidential record?\n"
            f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
        ),
        stratum=stratum,
    ))

    # C03: Invalidate Independent Root D (Primary Root A survives)
    interventions.append(AcquisitionIntervention(
        condition_id="c03_retract_independent_root",
        label="Invalidate Independent Root D",
        retracted_source_ids=[indep_id],
        restored_source_ids=[],
        expected_state="ACTIVE",
        prompt_text=(
            f"{prose}\n\n"
            f"EVIDENTIAL AUDIT STATUS (CORRECTIONS APPLIED):\n"
            f"- SOURCE RETRACTION: Source {indep_id} was retracted due to environmental contamination.\n\n"
            f"TARGET INQUIRY:\n"
            f"Is the assertion '{world.target_entity} exhibits {world.target_property}' logically supported by the surviving valid evidential record?\n"
            f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
        ),
        stratum=stratum,
    ))

    # C04: Complete Cut (Invalidate Both A and D -> Echoes Collapse)
    interventions.append(AcquisitionIntervention(
        condition_id="c04_complete_root_cut",
        label="Complete Cut of Roots A and D (Echo Collapse)",
        retracted_source_ids=["A", indep_id],
        restored_source_ids=[],
        expected_state="UNKNOWN",
        prompt_text=(
            f"{prose}\n\n"
            f"EVIDENTIAL AUDIT STATUS (CORRECTIONS APPLIED):\n"
            f"- SOURCE RETRACTION: Source A is RETRACTED (hardware failure).\n"
            f"- SOURCE RETRACTION: Source {indep_id} is RETRACTED (environmental contamination).\n\n"
            f"TARGET INQUIRY:\n"
            f"Is the assertion '{world.target_entity} exhibits {world.target_property}' logically supported by the surviving valid evidential record?\n"
            f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
        ),
        stratum=stratum,
    ))

    # C05: Laundered Echo Retraction (Retract Derivative B while A and D survive)
    interventions.append(AcquisitionIntervention(
        condition_id="c05_retract_echo_only",
        label="Retract Derivative Echo B (Sham Invalidation)",
        retracted_source_ids=["B"],
        restored_source_ids=[],
        expected_state="ACTIVE",
        prompt_text=(
            f"{prose}\n\n"
            f"EVIDENTIAL AUDIT STATUS (CORRECTIONS APPLIED):\n"
            f"- SOURCE RETRACTION: Secondary review Source B was withdrawn due to editorial dispute. Primary data sources remain intact.\n\n"
            f"TARGET INQUIRY:\n"
            f"Is the assertion '{world.target_entity} exhibits {world.target_property}' logically supported by the surviving valid evidential record?\n"
            f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
        ),
        stratum=stratum,
    ))

    # C06: Rescue Primary Root A after Complete Cut
    interventions.append(AcquisitionIntervention(
        condition_id="c06_rescue_primary_root",
        label="Rescue Primary Root A after Cut",
        retracted_source_ids=["A", indep_id],
        restored_source_ids=["A"],
        expected_state="ACTIVE",
        prompt_text=(
            f"{prose}\n\n"
            f"EVIDENTIAL AUDIT STATUS (CORRECTIONS APPLIED):\n"
            f"- SOURCE RETRACTION: Source A was initially retracted.\n"
            f"- SOURCE RETRACTION: Source {indep_id} is RETRACTED.\n"
            f"- SOURCE RESTORATION: Source A was audited by external committee, re-verified, and restored to FULL ACTIVE VALIDITY.\n\n"
            f"TARGET INQUIRY:\n"
            f"Is the assertion '{world.target_entity} exhibits {world.target_property}' logically supported by the surviving valid evidential record?\n"
            f"Respond with exactly one word: ACTIVE, UNKNOWN, or RETRACTED."
        ),
        stratum=stratum,
    ))

    return interventions


VALID_OUTPUTS = {"ACTIVE", "UNKNOWN", "RETRACTED"}


def parse_response_state(response_text: str | None) -> str:
    """Strictly parse categorical response token, normalizing markdown delimiters."""
    if not response_text:
        return "FORMAT_FAILURE"
    clean_all = response_text.strip().strip("*_`#.:,\n\t ").upper()
    if clean_all in VALID_OUTPUTS:
        return clean_all
    first_line = response_text.strip().split("\n")[0].strip("*_`#.:,\n\t ").upper()
    if first_line in VALID_OUTPUTS:
        return first_line
    first_word = response_text.strip().split()[0].strip("*_`#.:,\n\t ").upper()
    if first_word in VALID_OUTPUTS:
        return first_word
    return "FORMAT_FAILURE"


def trajectory_accuracy(observed: list[str], expected: list[str]) -> float:
    """Calculate categorical exact match accuracy over an intervention trajectory."""
    if not observed or len(observed) != len(expected):
        return 0.0
    matches = sum(1 for o, e in zip(observed, expected) if o == e)
    return matches / len(expected)


def trajectory_stability(vec1: list[str], vec2: list[str]) -> float:
    """Calculate agreement fraction between two trajectory vectors."""
    if not vec1 or len(vec1) != len(vec2):
        return 0.0
    matches = sum(1 for a, b in zip(vec1, vec2) if a == b)
    return matches / len(vec1)


def extract_model_codeword(run_dir: Any, holdout_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract cell-level 144-element decision codeword and 72-element renderer flip mask."""
    from pathlib import Path
    r_dir = Path(run_dir)
    attempts_file = r_dir / "raw" / "attempts.jsonl"
    attempts = []
    if attempts_file.exists():
        import json
        with open(attempts_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    attempts.append(json.loads(line))
    attempt_map = {a["probe_id"]: a for a in attempts if a.get("ok")}

    synth_codeword = []
    nat_codeword = []
    flip_mask = []
    acq_cells = []
    c4_cells = []
    strict_matches = 0
    total_cells = 0

    for item in holdout_data:
        w_id = item["world"]["world_id"]
        for t_s, t_n, exp in zip(item["synthetic_trajectory"], item["naturalistic_trajectory"], item["ground_truth"]):
            cid = t_s["condition_id"]
            p_s = f"{w_id}_SYNTHETIC_{cid}"
            p_n = f"{w_id}_NATURALISTIC_{cid}"

            att_s = attempt_map.get(p_s)
            att_n = attempt_map.get(p_n)

            msg_s = att_s.get("response_json", {}).get("choices", [{}])[0].get("message", {}) if att_s else {}
            raw_s = (msg_s.get("content") if isinstance(msg_s, dict) else "") or ""
            sem_s = parse_response_state(raw_s)
            clean_s_strict = raw_s.strip().upper().rstrip(".")
            if clean_s_strict == exp:
                strict_matches += 1

            msg_n = att_n.get("response_json", {}).get("choices", [{}])[0].get("message", {}) if att_n else {}
            raw_n = (msg_n.get("content") if isinstance(msg_n, dict) else "") or ""
            sem_n = parse_response_state(raw_n)
            clean_n_strict = raw_n.strip().upper().rstrip(".")
            if clean_n_strict == exp:
                strict_matches += 1

            total_cells += 2
            synth_codeword.append(sem_s)
            nat_codeword.append(sem_n)
            is_flip = (sem_s != sem_n)
            flip_mask.append(1 if is_flip else 0)

            if cid in ("c02_retract_primary_root", "c03_retract_independent_root", "c05_retract_echo_only", "c06_rescue_primary_root"):
                acq_cells.extend([sem_s, sem_n])
            elif cid == "c04_complete_root_cut":
                c4_cells.extend([sem_s, sem_n])

    full_codeword = synth_codeword + nat_codeword
    return {
        "full_codeword": full_codeword,
        "synth_codeword": synth_codeword,
        "nat_codeword": nat_codeword,
        "flip_mask": flip_mask,
        "acq_cells": acq_cells,
        "c4_cells": c4_cells,
        "strict_acc": strict_matches / total_cells if total_cells else 0.0,
    }


def compute_lineage_distance_vector(codeword_target: dict[str, Any], codeword_candidate: dict[str, Any]) -> dict[str, float]:
    """Compute 5-component fine-grained lineage distance vector D(Target, Candidate)."""
    # 1. D_acq: Hamming distance on core acquisition cells (N=96)
    acq_t = codeword_target["acq_cells"]
    acq_c = codeword_candidate["acq_cells"]
    d_acq = sum(1 for a, b in zip(acq_t, acq_c) if a != b) / len(acq_t) if acq_t else 0.0

    # 2. D_cal: Distance on C4 complete-cut calibration distribution (N=24)
    c4_t = codeword_target["c4_cells"]
    c4_c = codeword_candidate["c4_cells"]
    d_cal = sum(1 for a, b in zip(c4_t, c4_c) if a != b) / len(c4_t) if c4_t else 0.0

    # 3. D_render: Hamming distance on 72-element renderer flip masks
    flip_t = codeword_target["flip_mask"]
    flip_c = codeword_candidate["flip_mask"]
    d_render = sum(1 for a, b in zip(flip_t, flip_c) if a != b) / len(flip_t) if flip_t else 0.0

    # 4. D_contract: Strict schema compliance discrepancy
    d_contract = abs(codeword_target["strict_acc"] - codeword_candidate["strict_acc"])

    # 5. D_total: Overall 144-decision Hamming distance
    cw_t = codeword_target["full_codeword"]
    cw_c = codeword_candidate["full_codeword"]
    d_total = sum(1 for a, b in zip(cw_t, cw_c) if a != b) / len(cw_t) if cw_t else 0.0

    return {
        "D_acq": d_acq,
        "D_cal": d_cal,
        "D_render": d_render,
        "D_contract": d_contract,
        "D_total": d_total,
    }


