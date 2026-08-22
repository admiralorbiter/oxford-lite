#!/usr/bin/env python3
"""OXFORD Lineage Calibration and Leave-One-Relative-Out (LORO) Engine.

Version: 1.0.0
Specification: Four-Channel Lineage Architecture (Structural, Cognitive, Calibration, Surface).
All metrics are strictly versioned, printing explicit numerators and denominators,
and computing cell-level Hamming distances strictly over shared evaluable cells.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC_SPEC_VERSION = "1.0.0"
HOLDOUT_SHA256 = "b54933d6f793594b346fc2264b4e92361de7cd12b4dfe94055bb3fc614203038"


def load_run_attempts(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load successful attempts map from a run folder."""
    attempts_file = run_dir / "raw" / "attempts.jsonl"
    if not attempts_file.exists():
        return {}
    attempts = {}
    with open(attempts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("probe_id"):
                    attempts[rec["probe_id"]] = rec
    return attempts


def compute_four_channel_vector(
    model_name: str,
    attempts: dict[str, dict[str, Any]],
    holdout_data: list[dict[str, Any]],
    tokenizer_family: str = "UNKNOWN",
) -> dict[str, Any]:
    """Compile the complete 4-channel forensic profile for a model."""
    from assays import support_acquisition as sa

    valid_outputs = sa.VALID_OUTPUTS
    
    total_evaluated = 0
    strict_correct = 0
    semantic_correct = 0
    
    c2_correct, c2_total = 0, 0
    c3_correct, c3_total = 0, 0
    c4_unknown, c4_retracted, c4_active, c4_total = 0, 0, 0, 0
    c5_correct, c5_total = 0, 0
    c6_correct, c6_total = 0, 0
    
    false_retraction_count = 0
    perturbed_active_total = 0
    
    ric_by_echo = defaultdict(lambda: {"active": 0, "total": 0})
    
    decisions = {}
    contracts = {}
    flip_masks = {}

    for item in holdout_data:
        w_id = item["world"]["world_id"]
        echo = item["world"]["echo_count"]
        gt = item["ground_truth"]

        for t_s, t_n, exp in zip(item["synthetic_trajectory"], item["naturalistic_trajectory"], gt):
            pid_s = f"{w_id}_SYNTHETIC_{t_s['condition_id']}"
            pid_n = f"{w_id}_NATURALISTIC_{t_n['condition_id']}"
            pair_key = (w_id, t_s["condition_id"])

            for pid, t in [(pid_s, t_s), (pid_n, t_n)]:
                att = attempts.get(pid)
                if not att:
                    continue

                msg = att.get("response_json", {}).get("choices", [{}])[0].get("message", {})
                raw = (msg.get("content") if isinstance(msg, dict) else "") or ""
                clean_strict = raw.strip().upper().rstrip(".")
                is_contract = 1 if clean_strict in valid_outputs else 0
                sem_state = sa.parse_response_state(raw)
                
                decisions[pid] = sem_state
                contracts[pid] = is_contract
                total_evaluated += 1

                if clean_strict == exp:
                    strict_correct += 1
                if sem_state == exp:
                    semantic_correct += 1

                cid = t["condition_id"]
                if cid in ("c02_retract_primary_root", "c03_retract_independent_root", "c05_retract_echo_only", "c06_rescue_primary_root"):
                    perturbed_active_total += 1
                    if sem_state in ("UNKNOWN", "RETRACTED"):
                        false_retraction_count += 1

                if cid == "c02_retract_primary_root":
                    c2_total += 1
                    if sem_state == exp:
                        c2_correct += 1
                elif cid == "c03_retract_independent_root":
                    c3_total += 1
                    if sem_state == exp:
                        c3_correct += 1
                elif cid == "c04_complete_root_cut":
                    c4_total += 1
                    if sem_state == "UNKNOWN":
                        c4_unknown += 1
                    elif sem_state == "RETRACTED":
                        c4_retracted += 1
                    elif sem_state == "ACTIVE":
                        c4_active += 1
                    ric_by_echo[echo]["total"] += 1
                    if sem_state == "ACTIVE":
                        ric_by_echo[echo]["active"] += 1
                elif cid == "c05_retract_echo_only":
                    c5_total += 1
                    if sem_state == exp:
                        c5_correct += 1
                elif cid == "c06_rescue_primary_root":
                    c6_total += 1
                    if sem_state == exp:
                        c6_correct += 1

            # Stratum comparison for flip mask
            st_s = decisions.get(pid_s)
            st_n = decisions.get(pid_n)
            if st_s and st_n:
                flip_masks[pair_key] = (1 if st_s != st_n else 0)

    # 1. Structural Channel
    ch_structural = {
        "tokenizer_family": tokenizer_family,
    }

    # 2. Cognitive Channel
    ch_cognitive = {
        "semantic_acc": (semantic_correct, total_evaluated, semantic_correct / total_evaluated if total_evaluated else 0.0),
        "c2_root_retention": (c2_correct, c2_total, c2_correct / c2_total if c2_total else 0.0),
        "c3_root_retention": (c3_correct, c3_total, c3_correct / c3_total if c3_total else 0.0),
        "false_retraction": (false_retraction_count, perturbed_active_total, false_retraction_count / perturbed_active_total if perturbed_active_total else 0.0),
        "false_survival": (c4_active, c4_total, c4_active / c4_total if c4_total else 0.0),
        "ric_2": (ric_by_echo[2]["active"], ric_by_echo[2]["total"], ric_by_echo[2]["active"] / ric_by_echo[2]["total"] if ric_by_echo[2]["total"] else 0.0),
        "ric_4": (ric_by_echo[4]["active"], ric_by_echo[4]["total"], ric_by_echo[4]["active"] / ric_by_echo[4]["total"] if ric_by_echo[4]["total"] else 0.0),
        "ric_8": (ric_by_echo[8]["active"], ric_by_echo[8]["total"], ric_by_echo[8]["active"] / ric_by_echo[8]["total"] if ric_by_echo[8]["total"] else 0.0),
    }

    # 3. Calibration Channel (Standard vs Decoupled Codebook)
    ch_calibration = {
        "false_falsification_standard": (c4_retracted, c4_total, c4_retracted / c4_total if c4_total else 0.0),
        "unknown_calibration_standard": (c4_unknown, c4_total, c4_unknown / c4_total if c4_total else 0.0),
    }

    # 4. Surface Channel
    n_flips = sum(flip_masks.values())
    tot_pairs = len(flip_masks)
    ch_surface = {
        "contract_adherence": (strict_correct, total_evaluated, strict_correct / total_evaluated if total_evaluated else 0.0),
        "renderer_stability": (tot_pairs - n_flips, tot_pairs, (tot_pairs - n_flips) / tot_pairs if tot_pairs else 0.0),
        "contracts": contracts,
        "flip_masks": flip_masks,
    }

    return {
        "spec_version": METRIC_SPEC_VERSION,
        "fixture_sha256": HOLDOUT_SHA256,
        "model_name": model_name,
        "total_evaluated": total_evaluated,
        "decisions": decisions,
        "contracts": contracts,
        "flip_masks": flip_masks,
        "structural": ch_structural,
        "cognitive": ch_cognitive,
        "calibration": ch_calibration,
        "surface": ch_surface,
    }


def compute_pairwise_distances(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    holdout_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute pairwise cell-level Hamming distances strictly across shared evaluated cells."""
    d_a = profile_a["decisions"]
    d_b = profile_b["decisions"]
    k_a = profile_a["contracts"]
    k_b = profile_b["contracts"]
    r_a = profile_a["flip_masks"]
    r_b = profile_b["flip_masks"]
    
    # 1. Total Decision Hamming Distance
    shared_pids = [pid for pid in d_a if pid in d_b]
    n_shared = len(shared_pids)
    if n_shared == 0:
        return {
            "n_shared": 0,
            "D_total": (0, 0, 0.0),
            "D_acq": (0, 0, 0.0),
            "D_cal": (0, 0, 0.0),
            "D_contract": (0, 0, 0.0),
            "D_render": (0, 0, 0.0),
        }

    total_diff = sum(1 for pid in shared_pids if d_a[pid] != d_b[pid])
    
    # 2. Cognitive Acquisition Distance (C2, C3, C5, C6)
    acq_pids = [pid for pid in shared_pids if any(c in pid for c in ("c02_", "c03_", "c05_", "c06_"))]
    acq_diff = sum(1 for pid in acq_pids if d_a[pid] != d_b[pid])
    
    # 3. Calibration Distance (C4)
    c4_pids = [pid for pid in shared_pids if "c04_" in pid]
    c4_diff = sum(1 for pid in c4_pids if d_a[pid] != d_b[pid])

    # 4. Surface Contract Mask Distance (strictly over shared evaluated cells)
    k_shared_pids = [pid for pid in k_a if pid in k_b]
    k_diff = sum(1 for pid in k_shared_pids if k_a[pid] != k_b[pid])
    
    # 5. Surface Renderer Flip Mask Distance (strictly over shared strata pairs)
    shared_pairs = [pair for pair in r_a if pair in r_b]
    r_diff = sum(1 for pair in shared_pairs if r_a[pair] != r_b[pair])

    return {
        "n_shared": n_shared,
        "D_total": (total_diff, n_shared, total_diff / n_shared if n_shared else 0.0),
        "D_acq": (acq_diff, len(acq_pids), acq_diff / len(acq_pids) if acq_pids else 0.0),
        "D_cal": (c4_diff, len(c4_pids), c4_diff / len(c4_pids) if c4_pids else 0.0),
        "D_contract": (k_diff, len(k_shared_pids), k_diff / len(k_shared_pids) if k_shared_pids else 0.0),
        "D_render": (r_diff, len(shared_pairs), r_diff / len(shared_pairs) if shared_pairs else 0.0),
    }


def compute_common_cells_matrix(
    profiles: dict[str, dict[str, Any]],
    holdout_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute distance matrix on the exact intersection of cells evaluable by ALL models."""
    models = list(profiles.keys())
    if not models:
        return {}

    # Intersection of all evaluated probe_ids
    common_pids = [pid for pid in profiles[models[0]]["decisions"] if all(pid in profiles[m]["decisions"] for m in models)]
    acq_common = [pid for pid in common_pids if any(c in pid for c in ("c02_", "c03_", "c05_", "c06_"))]
    c4_common = [pid for pid in common_pids if "c04_" in pid]
    
    # Intersection of all evaluated renderer pairs
    common_pairs = [pair for pair in profiles[models[0]]["flip_masks"] if all(pair in profiles[m]["flip_masks"] for m in models)]

    matrix = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            d1, d2 = profiles[m1]["decisions"], profiles[m2]["decisions"]
            k1, k2 = profiles[m1]["contracts"], profiles[m2]["contracts"]
            r1, r2 = profiles[m1]["flip_masks"], profiles[m2]["flip_masks"]

            dec_diff = sum(1 for pid in common_pids if d1[pid] != d2[pid])
            acq_diff = sum(1 for pid in acq_common if d1[pid] != d2[pid])
            c4_diff = sum(1 for pid in c4_common if d1[pid] != d2[pid])
            k_diff = sum(1 for pid in common_pids if k1[pid] != k2[pid])
            r_diff = sum(1 for pair in common_pairs if r1[pair] != r2[pair])

            matrix[(m1, m2)] = {
                "n_common": len(common_pids),
                "D_total": (dec_diff, len(common_pids), dec_diff / len(common_pids) if common_pids else 0.0),
                "D_acq": (acq_diff, len(acq_common), acq_diff / len(acq_common) if acq_common else 0.0),
                "D_cal": (c4_diff, len(c4_common), c4_diff / len(c4_common) if c4_common else 0.0),
                "D_contract": (k_diff, len(common_pids), k_diff / len(common_pids) if common_pids else 0.0),
                "D_render": (r_diff, len(common_pairs), r_diff / len(common_pairs) if common_pairs else 0.0),
            }

    return {
        "n_common_cells": len(common_pids),
        "n_common_pairs": len(common_pairs),
        "matrix": matrix,
    }


def evaluate_loro_clustering(
    profiles: dict[str, dict[str, Any]],
    known_families: dict[str, list[str]],
    holdout_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute Leave-One-Relative-Out (LORO) validation across known families."""
    loro_results = {}
    
    for fam_name, members in known_families.items():
        if len(members) < 2:
            continue
        for held_out in members:
            if held_out not in profiles:
                continue
            other_family_members = [m for m in members if m != held_out and m in profiles]
            non_family_models = [m for m in profiles if m not in members]

            if not other_family_members or not non_family_models:
                continue

            intra_dists = [
                compute_pairwise_distances(profiles[held_out], profiles[rel], holdout_data)["D_total"][2]
                for rel in other_family_members
            ]
            cross_dists = [
                compute_pairwise_distances(profiles[held_out], profiles[ctrl], holdout_data)["D_total"][2]
                for ctrl in non_family_models
            ]

            min_intra = min(intra_dists)
            min_cross = min(cross_dists)
            recovered = (min_intra < min_cross)

            loro_results[f"{fam_name}:{held_out}"] = {
                "family": fam_name,
                "held_out": held_out,
                "min_intra_dist": min_intra,
                "min_cross_dist": min_cross,
                "recovered_correctly": recovered,
            }

    total_tests = len(loro_results)
    successful_recoveries = sum(1 for r in loro_results.values() if r["recovered_correctly"])
    
    return {
        "total_loro_tests": total_tests,
        "successful_recoveries": successful_recoveries,
        "loro_accuracy": successful_recoveries / total_tests if total_tests else 0.0,
        "details": loro_results,
    }
