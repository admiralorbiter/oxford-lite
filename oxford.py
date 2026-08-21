#!/usr/bin/env python3
"""OXFORD Lite: a tiny black-box model-lineage pilot.

The pilot is intentionally narrow. It compares differential prompt-token counts
across a small synthetic corpus and saves raw observations for later audit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import random
import statistics
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
RUNS_DIR = ROOT / "runs"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PILOT_VERSION = "0.1.0"
DEFAULT_SEED = 20260821
DEFAULT_DELAY = 0.5

MODELS: list[dict[str, str]] = [
    {
        "id": "ox-alpha",
        "slug": "stealth/ox-alpha",
        "label": "Ox Alpha",
        "role": "target",
    },
    {
        "id": "glm-5.2",
        "slug": "z-ai/glm-5.2:free",
        "label": "GLM-5.2 (free)",
        "role": "candidate",
    },
    {
        "id": "gemma-4",
        "slug": "google/gemma-4-26b-a4b-it:free",
        "label": "Gemma 4 26B A4B (free)",
        "role": "negative_control",
    },
]

# Fresh pilot probes. They intentionally do not copy public community Ox/GLM
# fingerprint strings. The full study should generate/freeze a much larger
# candidate-only probe pool before touching the target.
PROBES: list[dict[str, str]] = [
    {
        "id": "p01-mixed-boundaries",
        "label": "Mixed boundaries",
        "text": "fjord_7F9Q::Δ::xYz__0042",
    },
    {
        "id": "p02-multiscript",
        "label": "Multiscript",
        "text": "Kestrel中文テストالعربية—naïve—Привет",
    },
    {
        "id": "p03-emoji-joiners",
        "label": "Emoji + joiners",
        "text": "orbit🛰️|family👨‍👩‍👧‍👦|keycap7️⃣|flag🇰🇪",
    },
    {
        "id": "p04-code",
        "label": "Code syntax",
        "text": "def μ(x:int)->str:\n    return f\"v::{x:08x}::{x**2}\"",
    },
    {
        "id": "p05-structured",
        "label": "Structured identifiers",
        "text": "urn:oxford:9f2c1d73-4a6b-48e1-a77d-00ff19ab73c2?x=17&y=A_B-C.D",
    },
    {
        "id": "p06-repetition",
        "label": "Repetition + whitespace",
        "text": "abababababababab  zzzzzzzzzzzz\tA__A__A__A\nEND",
    },
]

COMMON_PREFIX = "Return the single word OK. Do not explain.\n\nPayload:\n"


@dataclass
class Observation:
    run_id: str
    ordinal: int
    collected_at_utc: str
    model_id: str
    model_slug: str
    model_role: str
    probe_id: str
    probe_label: str
    probe_sha256: str
    status_code: int
    elapsed_ms: float
    ok: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_model: str | None
    response_id: str | None
    selected_headers: dict[str, str]
    request_payload: dict[str, Any]
    response_json: Any
    error: str | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def corpus_hash() -> str:
    return sha256_text(canonical_json(PROBES))


def make_run_id(prefix: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{prefix}"


def ensure_run_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_payload(model_slug: str, probe_text: str) -> dict[str, Any]:
    return {
        "model": model_slug,
        "messages": [
            {
                "role": "user",
                "content": COMMON_PREFIX + probe_text,
            }
        ],
        # Keep completion tiny: this pilot measures prompt token accounting,
        # not response quality. Avoid model-specific reasoning parameters.
        "max_tokens": 8,
    }


def selected_response_headers(headers: requests.structures.CaseInsensitiveDict) -> dict[str, str]:
    names = [
        "x-request-id",
        "cf-ray",
        "content-type",
        "server",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    ]
    return {name: headers[name] for name in names if name in headers}


def parse_usage(body: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(body, dict):
        return None, None, None
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        return None, None, None

    def int_or_none(v: Any) -> int | None:
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    return (
        int_or_none(usage.get("prompt_tokens")),
        int_or_none(usage.get("completion_tokens")),
        int_or_none(usage.get("total_tokens")),
    )


def perform_request(
    session: requests.Session,
    api_key: str,
    run_id: str,
    ordinal: int,
    model: dict[str, str],
    probe: dict[str, str],
) -> Observation:
    payload = build_payload(model["slug"], probe["text"])
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "OXFORD Lite",
    }
    started = time.perf_counter()
    collected = utc_now()
    try:
        response = session.post(API_URL, headers=headers, json=payload, timeout=180)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            body: Any = response.json()
        except ValueError:
            body = {"_non_json_body": response.text}
        prompt_tokens, completion_tokens, total_tokens = parse_usage(body)
        response_model = body.get("model") if isinstance(body, dict) else None
        response_id = body.get("id") if isinstance(body, dict) else None
        err = None
        if not response.ok:
            err = summarize_error(body) or f"HTTP {response.status_code}"
        return Observation(
            run_id=run_id,
            ordinal=ordinal,
            collected_at_utc=collected,
            model_id=model["id"],
            model_slug=model["slug"],
            model_role=model["role"],
            probe_id=probe["id"],
            probe_label=probe["label"],
            probe_sha256=sha256_text(probe["text"]),
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            ok=response.ok and prompt_tokens is not None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_model=response_model,
            response_id=response_id,
            selected_headers=selected_response_headers(response.headers),
            request_payload=payload,
            response_json=body,
            error=err,
        )
    except requests.RequestException as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return Observation(
            run_id=run_id,
            ordinal=ordinal,
            collected_at_utc=collected,
            model_id=model["id"],
            model_slug=model["slug"],
            model_role=model["role"],
            probe_id=probe["id"],
            probe_label=probe["label"],
            probe_sha256=sha256_text(probe["text"]),
            status_code=0,
            elapsed_ms=elapsed_ms,
            ok=False,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            response_model=None,
            response_id=None,
            selected_headers={},
            request_payload=payload,
            response_json=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def summarize_error(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, str):
        return err[:400]
    if isinstance(err, dict):
        message = err.get("message") or err.get("code")
        if message is not None:
            return str(message)[:400]
    message = body.get("message")
    return str(message)[:400] if message is not None else None


def build_request_plan(seed: int) -> list[tuple[dict[str, str], dict[str, str]]]:
    plan = [(model, probe) for model in MODELS for probe in PROBES]
    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def model_map() -> dict[str, dict[str, str]]:
    return {m["id"]: m for m in MODELS}


def probe_order() -> list[str]:
    return [p["id"] for p in PROBES]


def count_matrix(observations: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {m["id"]: {} for m in MODELS}
    for record in observations:
        value = record.get("prompt_tokens")
        if record.get("ok") and isinstance(value, int):
            matrix.setdefault(record["model_id"], {})[record["probe_id"]] = value
    return matrix


def pairwise_comparison(
    matrix: dict[str, dict[str, int]],
    target_id: str,
    other_id: str,
) -> dict[str, Any]:
    order = probe_order()
    common = [p for p in order if p in matrix.get(target_id, {}) and p in matrix.get(other_id, {})]
    result: dict[str, Any] = {
        "target_id": target_id,
        "other_id": other_id,
        "n_common": len(common),
        "probe_ids": common,
        "n_deltas": max(0, len(common) - 1),
        "offsets": {},
        "constant_offset": False,
        "offset_value": None,
        "offset_span": None,
        "shape_exact_matches": 0,
        "shape_match_ratio": None,
        "shape_mae": None,
        "shape_max_abs_error": None,
        "target_normalized": {},
        "other_normalized": {},
    }
    if not common:
        return result

    offsets = {
        pid: matrix[target_id][pid] - matrix[other_id][pid]
        for pid in common
    }
    offset_values = list(offsets.values())
    result["offsets"] = offsets
    result["offset_span"] = max(offset_values) - min(offset_values)
    result["constant_offset"] = len(set(offset_values)) == 1
    if result["constant_offset"]:
        result["offset_value"] = offset_values[0]

    # Differential geometry: subtract the first available probe within each
    # model. Any constant model-specific wrapper overhead cancels.
    baseline = common[0]
    target_norm = {pid: matrix[target_id][pid] - matrix[target_id][baseline] for pid in common}
    other_norm = {pid: matrix[other_id][pid] - matrix[other_id][baseline] for pid in common}
    # The baseline delta is zero by construction, so it carries no
    # discriminatory information and is excluded from the match statistics.
    informative = common[1:]
    abs_errors = [abs(target_norm[pid] - other_norm[pid]) for pid in informative]
    exact = sum(1 for e in abs_errors if e == 0)
    result["target_normalized"] = target_norm
    result["other_normalized"] = other_norm
    result["shape_exact_matches"] = exact
    if informative:
        result["shape_match_ratio"] = exact / len(informative)
        result["shape_mae"] = statistics.fmean(abs_errors)
        result["shape_max_abs_error"] = max(abs_errors)
    return result


def analyze(observations: list[dict[str, Any]], demo: bool) -> dict[str, Any]:
    matrix = count_matrix(observations)
    target = next(m for m in MODELS if m["role"] == "target")
    comparisons = []
    for model in MODELS:
        if model["id"] != target["id"]:
            comparisons.append(pairwise_comparison(matrix, target["id"], model["id"]))

    def rank_key(item: dict[str, Any]) -> tuple[float, float, int]:
        ratio = item["shape_match_ratio"]
        mae = item["shape_mae"]
        return (
            -1.0 if ratio is None else ratio,
            -math.inf if mae is None else -mae,
            item["n_common"],
        )

    ranked = sorted(comparisons, key=rank_key, reverse=True)
    strongest = ranked[0] if ranked and ranked[0]["n_common"] else None
    successful = sum(1 for r in observations if r.get("ok"))
    failed = len(observations) - successful
    return {
        "pilot_version": PILOT_VERSION,
        "demo": demo,
        "generated_at_utc": utc_now(),
        "requests_total": len(observations),
        "requests_successful": successful,
        "requests_failed": failed,
        "counts": matrix,
        "comparisons": comparisons,
        "strongest_structural_match": strongest,
        "interpretation": interpretation_text(strongest, failed, demo),
    }


def interpretation_text(strongest: dict[str, Any] | None, failed: int, demo: bool) -> str:
    if demo:
        return "Synthetic demo only. No inference about Ox Alpha is permitted from these values."
    if failed:
        return (
            f"{failed} request(s) failed or lacked prompt-token usage. Treat this run as a plumbing check "
            "and rerun before interpreting structural similarity."
        )
    if strongest is None:
        return "No complete pairwise comparison was available."
    other = model_map()[strongest["other_id"]]["label"]
    ratio = strongest["shape_match_ratio"] or 0.0
    if strongest["constant_offset"] and ratio == 1.0:
        return (
            f"Across this six-probe pilot, {other} has the same differential prompt-token shape as Ox Alpha "
            f"with a constant absolute-count offset of {strongest['offset_value']} tokens. This is an interesting "
            "structural signal, not a model/checkpoint/provider attribution."
        )
    if ratio >= 0.8:
        return (
            f"Across this small pilot, {other} is the closest tested structural match to Ox Alpha "
            f"({strongest['shape_exact_matches']}/{strongest['n_deltas']} informative normalized deltas match exactly). "
            "The corpus is too small for confirmatory attribution."
        )
    return (
        f"None of the tested controls reproduces Ox Alpha's differential token-count shape closely in this small pilot. "
        f"The nearest tested model is {other}; this does not imply an unmodeled lineage without a larger preregistered study."
    )


def demo_observations(run_id: str) -> list[dict[str, Any]]:
    # Synthetic values chosen only to exercise the report. Here Ox = GLM + 75
    # across every probe while Gemma differs. These are deliberately NOT real
    # observations and the report prominently marks them as demo data.
    glm_counts = [34, 41, 47, 50, 45, 42]
    ox_counts = [x + 75 for x in glm_counts]
    gemma_counts = [29, 44, 52, 46, 49, 38]
    by_model = {
        "ox-alpha": ox_counts,
        "glm-5.2": glm_counts,
        "gemma-4": gemma_counts,
    }
    rows: list[dict[str, Any]] = []
    ordinal = 1
    for model in MODELS:
        for i, probe in enumerate(PROBES):
            rows.append(
                {
                    "run_id": run_id,
                    "ordinal": ordinal,
                    "collected_at_utc": utc_now(),
                    "model_id": model["id"],
                    "model_slug": model["slug"],
                    "model_role": model["role"],
                    "probe_id": probe["id"],
                    "probe_label": probe["label"],
                    "probe_sha256": sha256_text(probe["text"]),
                    "status_code": 200,
                    "elapsed_ms": 800 + ordinal * 11,
                    "ok": True,
                    "prompt_tokens": by_model[model["id"]][i],
                    "completion_tokens": 1,
                    "total_tokens": by_model[model["id"]][i] + 1,
                    "response_model": model["slug"],
                    "response_id": f"demo-{ordinal:02d}",
                    "selected_headers": {},
                    "request_payload": build_payload(model["slug"], probe["text"]),
                    "response_json": {"demo": True},
                    "error": None,
                }
            )
            ordinal += 1
    return rows


def manifest(run_id: str, kind: str, seed: int, request_order: list[tuple[dict[str, str], dict[str, str]]] | None) -> dict[str, Any]:
    order = []
    if request_order is not None:
        order = [
            {"ordinal": i + 1, "model_id": model["id"], "probe_id": probe["id"]}
            for i, (model, probe) in enumerate(request_order)
        ]
    return {
        "run_id": run_id,
        "kind": kind,
        "pilot_version": PILOT_VERSION,
        "created_at_utc": utc_now(),
        "seed": seed,
        "api_url": API_URL,
        "models": MODELS,
        "probes": [
            {**p, "sha256": sha256_text(p["text"])}
            for p in PROBES
        ],
        "probe_corpus_sha256": corpus_hash(),
        "common_prefix_sha256": sha256_text(COMMON_PREFIX),
        "requests_expected": len(MODELS) * len(PROBES),
        "request_order": order,
        "analysis_note": "Pilot only; no calibrated probabilities and no provider/checkpoint attribution.",
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any], run_id: str) -> str:
    demo = summary["demo"]
    label = "DEMO / SYNTHETIC DATA" if demo else "REAL PILOT"
    models = model_map()
    lines = [
        f"# OXFORD Lite — {label}",
        "",
        "> **Pilot only.** This report measures a tiny structural fingerprint. It does not identify an exact model, checkpoint, or provider.",
        "",
        f"Run: `{run_id}`  ",
        f"Generated: `{summary['generated_at_utc']}`  ",
        f"Successful requests: **{summary['requests_successful']}/{summary['requests_total']}**",
        "",
        "## Pairwise structural comparison",
        "",
        "| Ox Alpha vs. | Common probes | Exact normalized deltas | Shape MAE | Constant offset | Offset span |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for comp in summary["comparisons"]:
        label_m = models[comp["other_id"]]["label"]
        const = f"Yes ({comp['offset_value']:+d})" if comp["constant_offset"] and comp["offset_value"] is not None else "No"
        lines.append(
            f"| {label_m} | {comp['n_common']} | {comp['shape_exact_matches']}/{comp['n_deltas']} ({pct(comp['shape_match_ratio'])}) | "
            f"{num(comp['shape_mae'])} | {const} | {num(comp['offset_span'])} |"
        )
    lines.extend([
        "",
        "## Prompt-token counts",
        "",
        "| Probe | Ox Alpha | GLM-5.2 | Gemma 4 |",
        "|---|---:|---:|---:|",
    ])
    counts = summary["counts"]
    for p in PROBES:
        lines.append(
            f"| {p['label']} | {counts.get('ox-alpha', {}).get(p['id'], '—')} | "
            f"{counts.get('glm-5.2', {}).get(p['id'], '—')} | {counts.get('gemma-4', {}).get(p['id'], '—')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Scientific boundary",
        "",
        "- Existing public Ox/GLM fingerprint strings are not in this six-probe corpus.",
        "- Absolute token counts can include wrapper/chat-template overhead; the main comparison uses differential shape.",
        "- This pilot does not pin providers and contains no behavior/tool/vision assays.",
        "- A successful-looking pilot is a build/plumbing green light, not confirmatory evidence.",
        "",
    ])
    return "\n".join(lines)


def render_html(summary: dict[str, Any], run_id: str) -> str:
    models = model_map()
    demo = summary["demo"]
    badge = "SYNTHETIC DEMO" if demo else "REAL PILOT"
    strongest = summary.get("strongest_structural_match")
    strongest_name = "—"
    strongest_ratio = "—"
    strongest_mae = "—"
    if strongest:
        strongest_name = models[strongest["other_id"]]["label"]
        strongest_ratio = f"{strongest['shape_exact_matches']}/{strongest['n_deltas']}"
        strongest_mae = num(strongest["shape_mae"])

    comp_rows = []
    for comp in summary["comparisons"]:
        name = html.escape(models[comp["other_id"]]["label"])
        const = (
            f"Yes · {comp['offset_value']:+d} tokens"
            if comp["constant_offset"] and comp["offset_value"] is not None
            else "No"
        )
        comp_rows.append(
            "<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{comp['n_common']}</td>"
            f"<td>{comp['shape_exact_matches']}/{comp['n_deltas']} <span class='muted'>({pct(comp['shape_match_ratio'])})</span></td>"
            f"<td>{num(comp['shape_mae'])}</td>"
            f"<td>{html.escape(const)}</td>"
            f"<td>{num(comp['offset_span'])}</td>"
            "</tr>"
        )

    counts = summary["counts"]
    count_rows = []
    for p in PROBES:
        count_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(p['label'])}</strong><div class='mono small'>{html.escape(p['id'])}</div></td>"
            f"<td>{counts.get('ox-alpha', {}).get(p['id'], '—')}</td>"
            f"<td>{counts.get('glm-5.2', {}).get(p['id'], '—')}</td>"
            f"<td>{counts.get('gemma-4', {}).get(p['id'], '—')}</td>"
            "</tr>"
        )

    demo_warning = (
        "<div class='callout demo'><strong>Demo data.</strong> The values on this page are synthetic and exist only to test the report UI.</div>"
        if demo
        else ""
    )
    fail_warning = ""
    if summary["requests_failed"]:
        fail_warning = (
            f"<div class='callout warn'><strong>Incomplete run.</strong> {summary['requests_failed']} request(s) failed or lacked prompt-token usage. "
            "Do not interpret similarity until you have a complete run.</div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OXFORD Lite · {html.escape(run_id)}</title>
<style>
:root {{ --ink:#111827; --muted:#667085; --line:#e5e7eb; --paper:#ffffff; --wash:#f6f7f9; --accent:#1f4b99; --accent2:#7a3e9d; --good:#17633a; --warn:#8a4b08; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink); background:var(--wash); }}
.shell {{ max-width:1120px; margin:0 auto; padding:34px 22px 64px; }}
.hero {{ background:linear-gradient(135deg,#0f172a,#1f3a69 62%,#492f67); color:white; border-radius:22px; padding:30px 32px 28px; box-shadow:0 12px 35px rgba(15,23,42,.14); }}
.eyebrow {{ font-size:12px; letter-spacing:.14em; font-weight:800; opacity:.78; }}
h1 {{ margin:8px 0 4px; font-size:34px; letter-spacing:-.04em; }}
.sub {{ max-width:760px; color:#dbe4f2; line-height:1.55; }}
.badges {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }}
.badge {{ display:inline-flex; border:1px solid rgba(255,255,255,.25); padding:6px 9px; border-radius:999px; font-size:12px; background:rgba(255,255,255,.08); }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:18px 0; }}
.card {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 4px 16px rgba(15,23,42,.035); }}
.card .k {{ color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }}
.card .v {{ margin-top:7px; font-size:24px; font-weight:800; letter-spacing:-.03em; }}
section {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; margin-top:14px; padding:22px; }}
h2 {{ margin:0 0 6px; font-size:20px; letter-spacing:-.02em; }}
p {{ line-height:1.6; }}
.muted {{ color:var(--muted); }}
.small {{ font-size:11px; color:var(--muted); margin-top:3px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:14px; }}
th {{ text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:10px 9px; border-bottom:1px solid var(--line); }}
td {{ padding:12px 9px; border-bottom:1px solid #eef0f3; vertical-align:top; }}
tr:last-child td {{ border-bottom:0; }}
.callout {{ border-left:4px solid var(--accent); padding:12px 14px; border-radius:8px; background:#f1f6ff; margin-top:14px; line-height:1.55; }}
.callout.demo {{ border-left-color:var(--accent2); background:#faf4ff; }}
.callout.warn {{ border-left-color:#c36b0a; background:#fff7ed; }}
.interpret {{ font-size:17px; line-height:1.6; }}
.boundary {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.boundary div {{ background:#fafafa; border:1px solid var(--line); border-radius:12px; padding:14px; }}
.footer {{ color:var(--muted); font-size:12px; margin:18px 3px 0; }}
@media(max-width:800px) {{ .grid {{ grid-template-columns:1fr 1fr; }} .boundary {{ grid-template-columns:1fr; }} }}
@media(max-width:520px) {{ .grid {{ grid-template-columns:1fr; }} .hero {{ padding:24px 20px; }} h1 {{ font-size:29px; }} }}
</style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <div class="eyebrow">OXFORD · BLACK-BOX MODEL LINEAGE</div>
    <h1>Structural fingerprint pilot</h1>
    <div class="sub">A deliberately tiny plumbing test: six fresh prompts, three free model routes, differential prompt-token geometry, raw observations preserved.</div>
    <div class="badges"><span class="badge">{html.escape(badge)}</span><span class="badge">PILOT · NOT CONFIRMATORY</span><span class="badge mono">{html.escape(run_id)}</span></div>
  </div>
  {demo_warning}
  {fail_warning}
  <div class="grid">
    <div class="card"><div class="k">Successful requests</div><div class="v">{summary['requests_successful']}/{summary['requests_total']}</div></div>
    <div class="card"><div class="k">Closest tested model</div><div class="v" style="font-size:18px">{html.escape(strongest_name)}</div></div>
    <div class="card"><div class="k">Exact normalized deltas</div><div class="v">{strongest_ratio}</div></div>
    <div class="card"><div class="k">Shape MAE</div><div class="v">{strongest_mae}</div></div>
  </div>

  <section>
    <h2>What the pilot found</h2>
    <p class="interpret">{html.escape(summary['interpretation'])}</p>
  </section>

  <section>
    <h2>Pairwise structural comparison</h2>
    <p class="muted">For each control, counts are normalized by subtracting the first common probe. A shared constant wrapper offset therefore disappears.</p>
    <table>
      <thead><tr><th>Ox Alpha vs.</th><th>Common</th><th>Exact deltas</th><th>Shape MAE</th><th>Constant offset</th><th>Offset span</th></tr></thead>
      <tbody>{''.join(comp_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Observed prompt-token counts</h2>
    <p class="muted">These are the API's reported prompt-token counts. Absolute counts are shown for auditability but are not treated as attribution by themselves.</p>
    <table>
      <thead><tr><th>Probe</th><th>Ox Alpha</th><th>GLM-5.2</th><th>Gemma 4</th></tr></thead>
      <tbody>{''.join(count_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Scientific boundary</h2>
    <div class="boundary">
      <div><strong>This pilot can tell us</strong><p class="muted">Whether the runner works, raw records are usable, the report communicates the result, and a tiny differential-token signal is worth scaling.</p></div>
      <div><strong>This pilot cannot tell us</strong><p class="muted">Exact checkpoint, model operator, inference provider, calibrated probability, or publication-grade lineage attribution.</p></div>
    </div>
  </section>
  <div class="footer">OXFORD Lite v{PILOT_VERSION} · corpus SHA-256 {html.escape(corpus_hash()[:16])}… · generated {html.escape(summary['generated_at_utc'])}</div>
</div>
</body>
</html>"""


def save_run(run_dir: Path, run_manifest: dict[str, Any], observations: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    write_json(run_dir / "manifest.json", run_manifest)
    write_jsonl(run_dir / "raw.jsonl", observations)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_markdown(summary, run_manifest["run_id"]), encoding="utf-8")
    report_path = run_dir / "report.html"
    report_path.write_text(render_html(summary, run_manifest["run_id"]), encoding="utf-8")
    return report_path


def print_comparison_console(summary: dict[str, Any]) -> None:
    print("\nPairwise structural comparison (Ox Alpha target):")
    for comp in summary["comparisons"]:
        label = model_map()[comp["other_id"]]["label"]
        if comp["n_common"]:
            const = f"constant offset {comp['offset_value']:+d}" if comp["constant_offset"] else f"offset span {comp['offset_span']}"
            print(
                f"  - {label}: {comp['shape_exact_matches']}/{comp['n_deltas']} informative normalized deltas exact; "
                f"MAE={num(comp['shape_mae'])}; {const}"
            )
        else:
            print(f"  - {label}: no complete observations")
    print(f"\n{summary['interpretation']}")


def command_doctor() -> int:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    print(f"OXFORD Lite v{PILOT_VERSION}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"requests: {requests.__version__}")
    print(f"API key present: {'yes' if key else 'NO'}")
    print(f"Real pilot requests: {len(MODELS) * len(PROBES)}")
    print("Models:")
    for model in MODELS:
        print(f"  - {model['role']}: {model['slug']}")
    if not key:
        print("\nAdd OPENROUTER_API_KEY to .env before running the real pilot.")
        return 2
    print("\nConfiguration looks ready. Doctor makes no model/API request.")
    return 0


def command_demo(open_report: bool) -> int:
    run_id = make_run_id("demo")
    run_dir = ensure_run_dir(run_id)
    observations = demo_observations(run_id)
    summary = analyze(observations, demo=True)
    run_manifest = manifest(run_id, "synthetic_demo", DEFAULT_SEED, None)
    report = save_run(run_dir, run_manifest, observations, summary)
    print(f"Demo report created: {report}")
    print_comparison_console(summary)
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0


def command_pilot(seed: int, delay: float, open_report: bool) -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 2
    if delay < 0:
        print("--delay must be >= 0", file=sys.stderr)
        return 2

    run_id = make_run_id("pilot")
    run_dir = ensure_run_dir(run_id)
    plan = build_request_plan(seed)
    run_manifest = manifest(run_id, "real_pilot", seed, plan)
    write_json(run_dir / "manifest.json", run_manifest)

    print(f"OXFORD Lite real pilot · {len(plan)} requests")
    print(f"Run folder: {run_dir}")
    print("No retries and no model fallback. Failures are preserved as observations.\n")

    observations: list[dict[str, Any]] = []
    session = requests.Session()
    for ordinal, (model, probe) in enumerate(plan, start=1):
        print(f"[{ordinal:02d}/{len(plan)}] {model['label']} · {probe['label']} ... ", end="", flush=True)
        obs = perform_request(session, api_key, run_id, ordinal, model, probe)
        row = asdict(obs)
        observations.append(row)
        # Persist after every request so an interrupted pilot still has data.
        write_jsonl(run_dir / "raw.jsonl", observations)
        if obs.ok:
            print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
        else:
            print(f"FAILED · {obs.status_code} · {obs.error}")
        if ordinal != len(plan) and delay:
            time.sleep(delay)

    summary = analyze(observations, demo=False)
    report = save_run(run_dir, run_manifest, observations, summary)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    print(f"Raw observations: {run_dir / 'raw.jsonl'}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oxford.py",
        description="OXFORD Lite black-box model-lineage pilot",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check local configuration; makes zero model calls")

    demo = sub.add_parser("demo", help="Create a synthetic sample report; makes zero model calls")
    demo.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in your browser")

    pilot = sub.add_parser("pilot", help="Run the real 18-request pilot")
    pilot.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in your browser")
    pilot.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"request-order shuffle seed (default {DEFAULT_SEED})")
    pilot.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("OXFORD_DELAY_SECONDS", str(DEFAULT_DELAY))),
        help=f"seconds between requests (default {DEFAULT_DELAY})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return command_doctor()
    if args.command == "demo":
        return command_demo(args.open_report)
    if args.command == "pilot":
        return command_pilot(args.seed, args.delay, args.open_report)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
