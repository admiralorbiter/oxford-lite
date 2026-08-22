#!/usr/bin/env python3
"""OXFORD Lineage Calibration and Leave-One-Relative-Out (LORO) Engine.

Decomposes model similarity across four orthogonal evidence channels:
1. Structural Channel: Tokenizer geometry & byte merges (E1)
2. Cognitive Channel: Support acquisition, root tracking, & RIC(e) (E3A)
3. Calibration Channel: Epistemic state mapping under decoupled codebooks (E3C)
4. Surface Channel: Schema adherence & localized renderer-flip distributions (E3A)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_run_attempts(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load attempts map from a run folder."""
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
    all_pids = []
    synth_codeword = []
    nat_codeword = []
    flip_mask = []
    contract_mask = []
    
    total_evaluated = 0
    strict_correct = 0
    semantic_correct = 0
    
    c2_correct, c2_total = 0, 0
    c3_correct, c3_total = 0, 0
    c4_unknown, c4_retracted, c4_active, c4_total = 0, 0, 0, 0
    c5_correct, c5_total = 0, 0
    c6_correct, c6_total = 0, 0
    
    ric_by_echo = defaultdict(lambda: {"active": 0, "total": 0})
    decisions = {}

    for item in holdout_data:
        w_id = item["world"]["world_id"]
        echo = item["world"]["echo_count"]
        gt = item["ground_truth"]

        for t_s, t_n, exp in zip(item["synthetic_trajectory"], item["naturalistic_trajectory"], gt):
            pid_s = f"{w_id}_SYNTHETIC_{t_s['condition_id']}"
            pid_n = f"{w_id}_NATURALISTIC_{t_n['condition_id']}"
            all_pids.extend([pid_s, pid_n])

            for pid, t in [(pid_s, t_s), (pid_n, t_n)]:
                att = attempts.get(pid)
                if not att:
                    decisions[pid] = "MISSING"
                    contract_mask.append(0)
                    continue

                msg = att.get("response_json", {}).get("choices", [{}])[0].get("message", {})
                raw = (msg.get("content") if isinstance(msg, dict) else "") or ""
                clean_strict = raw.strip().upper().rstrip(".")
                is_contract = 1 if clean_strict in valid_outputs else 0
                sem_state = sa.parse_response_state(raw)
                
                decisions[pid] = sem_state
                contract_mask.append(is_contract)
                total_evaluated += 1

                if clean_strict == exp:
                    strict_correct += 1
                if sem_state == exp:
                    semantic_correct += 1

                cid = t["condition_id"]
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
            st_s = decisions.get(pid_s, "MISSING")
            st_n = decisions.get(pid_n, "MISSING")
            if st_s != "MISSING" and st_n != "MISSING":
                flip_mask.append(1 if st_s != st_n else 0)
                synth_codeword.append(st_s)
                nat_codeword.append(st_n)

    # Compile 4 channels
    # 1. Structural
    ch_structural = {"tokenizer_family": tokenizer_family}

    # 2. Cognitive
    perturbed_active_tot = c2_total + c3_total + c5_total + c6_total
    perturbed_active_cor = c2_correct + c3_correct + c5_correct + c6_correct
    ch_cognitive = {
        "semantic_acc": semantic_correct / total_evaluated if total_evaluated else 0.0,
        "c2_root_retention": c2_correct / c2_total if c2_total else 0.0,
        "c3_root_retention": c3_correct / c3_total if c3_total else 0.0,
        "false_retraction": 1.0 - (perturbed_active_cor / perturbed_active_tot) if perturbed_active_tot else 0.0,
        "false_survival": c4_active / c4_total if c4_total else 0.0,
        "ric_2": ric_by_echo[2]["active"] / ric_by_echo[2]["total"] if ric_by_echo[2]["total"] else 0.0,
        "ric_4": ric_by_echo[4]["active"] / ric_by_echo[4]["total"] if ric_by_echo[4]["total"] else 0.0,
        "ric_8": ric_by_echo[8]["active"] / ric_by_echo[8]["total"] if ric_by_echo[8]["total"] else 0.0,
    }

    # 3. Calibration
    ch_calibration = {
        "false_falsification_standard": c4_retracted / c4_total if c4_total else 0.0,
        "unknown_calibration_standard": c4_unknown / c4_total if c4_total else 0.0,
    }

    # 4. Surface
    ch_surface = {
        "contract_adherence": strict_correct / total_evaluated if total_evaluated else 0.0,
        "renderer_stability": 1.0 - (sum(flip_mask) / len(flip_mask)) if flip_mask else 0.0,
        "contract_mask": contract_mask,
        "flip_mask": flip_mask,
    }

    return {
        "model_name": model_name,
        "total_evaluated": total_evaluated,
        "decisions": decisions,
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
    """Compute pairwise shared distances between two model profiles."""
    d_a = profile_a["decisions"]
    d_b = profile_b["decisions"]
    
    shared_pids = [pid for pid in d_a if pid in d_b and d_a[pid] != "MISSING" and d_b[pid] != "MISSING"]
    n_shared = len(shared_pids)
    if n_shared == 0:
        return {"n_shared": 0, "D_total": 0.0, "D_acq": 0.0, "D_contract": 0.0, "D_render": 0.0}

    total_diff = sum(1 for pid in shared_pids if d_a[pid] != d_b[pid])
    
    # Cognitive acquisition cells (C2, C3, C5, C6)
    acq_pids = [pid for pid in shared_pids if any(c in pid for c in ("c02_", "c03_", "c05_", "c06_"))]
    acq_diff = sum(1 for pid in acq_pids if d_a[pid] != d_b[pid])
    
    # Surface contract mask distance
    k_a = profile_a["surface"]["contract_mask"]
    k_b = profile_b["surface"]["contract_mask"]
    min_k = min(len(k_a), len(k_b))
    k_diff = sum(1 for i in range(min_k) if k_a[i] != k_b[i])
    
    # Surface renderer flip mask distance
    r_a = profile_a["surface"]["flip_mask"]
    r_b = profile_b["surface"]["flip_mask"]
    min_r = min(len(r_a), len(r_b))
    r_diff = sum(1 for i in range(min_r) if r_a[i] != r_b[i])

    return {
        "n_shared": n_shared,
        "D_total": total_diff / n_shared,
        "D_acq": acq_diff / len(acq_pids) if acq_pids else 0.0,
        "D_contract": k_diff / min_k if min_k else 0.0,
        "D_render": r_diff / min_r if min_r else 0.0,
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
            non_family_models = [m for m, p in profiles.items() if m not in members]

            if not other_family_members or not non_family_models:
                continue

            intra_dists = [
                compute_pairwise_distances(profiles[held_out], profiles[rel], holdout_data)["D_total"]
                for rel in other_family_members
            ]
            cross_dists = [
                compute_pairwise_distances(profiles[held_out], profiles[ctrl], holdout_data)["D_total"]
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
