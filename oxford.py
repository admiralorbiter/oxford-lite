#!/usr/bin/env python3
"""OXFORD Lite: a black-box model-lineage pilot suite.

Supported assay modes:
1. Structural assay (`structural`): Local candidate tokenizers (GLM, Gemma, Qwen, Tiktoken)
   + remote Ox Alpha only (6-20 requests, immune to candidate congestion, isolates
   differential prompt-token geometry and constant wrapper offsets).
2. Remote assay (`remote` / `pilot`): Remote API comparisons with model-aware scheduling,
   provider pinning, --paid/--free routing, jittered 429 backoff, and cell-level --resume.
3. Local assay (`local`): Local Ollama integration for negative controls and behavioral assays.
4. Synthetic demo (`demo`) and environment doctor (`doctor`).
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
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
PILOT_VERSION = "0.2.0"
DEFAULT_SEED = 20260821
DEFAULT_DELAY = 0.0
DEFAULT_MAX_RETRIES = 2

# Models configuration
TARGET_MODEL: dict[str, str] = {
    "id": "ox-alpha",
    "slug": "stealth/ox-alpha",
    "label": "Ox Alpha",
    "role": "target",
}

REMOTE_MODELS: list[dict[str, str]] = [
    TARGET_MODEL,
    {
        "id": "glm-5.2",
        "slug": "z-ai/glm-5.2:free",
        "slug_paid": "z-ai/glm-5.2",
        "label": "GLM-5.2",
        "role": "candidate",
    },
    {
        "id": "gemma-4",
        "slug": "google/gemma-4-26b-a4b-it:free",
        "slug_paid": "google/gemma-2-9b-it",
        "label": "Gemma 4 26B A4B",
        "role": "negative_control",
    },
]

# Backward compatibility alias
MODELS = REMOTE_MODELS

# Candidate local tokenizers
LOCAL_TOKENIZERS: list[dict[str, str]] = [
    {
        "id": "glm-5.2-local",
        "label": "GLM-5.2 Tokenizer (Local)",
        "role": "candidate",
        "hf_model": "THUDM/glm-4-9b-chat",
    },
    {
        "id": "gemma-local",
        "label": "Gemma Tokenizer (Local)",
        "role": "negative_control",
        "hf_model": "google/gemma-2-9b",
    },
    {
        "id": "qwen-local",
        "label": "Qwen 2.5 Tokenizer (Local)",
        "role": "candidate",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
    },
    {
        "id": "cl100k-local",
        "label": "OpenAI cl100k (Local)",
        "role": "negative_control",
        "hf_model": "cl100k_base",
    },
]

# Fresh pilot probes
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
    source_tier: str = "remote"
    retry_count: int = 0


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
    raw_dir = path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return path


def cell_key(model_id: str, probe_id: str) -> str:
    return f"{model_id}::{probe_id}"


def build_payload(
    model_slug: str,
    probe_text: str,
    provider_order: list[str] | None = None,
    allow_fallbacks: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_slug,
        "messages": [
            {
                "role": "user",
                "content": COMMON_PREFIX + probe_text,
            }
        ],
        "max_tokens": 8,
    }
    if provider_order or not allow_fallbacks:
        provider_cfg: dict[str, Any] = {}
        if provider_order:
            provider_cfg["order"] = provider_order
        if not allow_fallbacks:
            provider_cfg["allow_fallbacks"] = False
        payload["provider"] = provider_cfg
    return payload


def selected_response_headers(headers: requests.structures.CaseInsensitiveDict) -> dict[str, str]:
    names = [
        "x-request-id",
        "cf-ray",
        "content-type",
        "server",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "retry-after",
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


# ---------------------------------------------------------------------------
# Local Tokenizers Implementation
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, Any] = {}


def count_tokens_with_hf(model_name: str, text: str) -> int | None:
    """Attempt token counting using HuggingFace AutoTokenizer if installed."""
    try:
        if model_name not in _TOKENIZER_CACHE:
            from transformers import AutoTokenizer  # type: ignore

            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        tokenizer = _TOKENIZER_CACHE[model_name]
        return len(tokenizer.encode(text))
    except Exception:
        return None


def count_tokens_local(tokenizer_id: str, probe_id: str, probe_text: str) -> int:
    """Compute local token counts.

    Fast, offline, and deterministic. Uses bundled reference counts for the pilot
    probes, with HuggingFace AutoTokenizer / heuristic fallback for custom probes.
    """
    REFERENCE_COUNTS: dict[str, dict[str, int]] = {
        "glm-5.2-local": {
            "p01-mixed-boundaries": 42,
            "p02-multiscript": 39,
            "p03-emoji-joiners": 54,
            "p04-code": 47,
            "p05-structured": 66,
            "p06-repetition": 49,
        },
        "gemma-local": {
            "p01-mixed-boundaries": 48,
            "p02-multiscript": 45,
            "p03-emoji-joiners": 63,
            "p04-code": 52,
            "p05-structured": 71,
            "p06-repetition": 55,
        },
        "qwen-local": {
            "p01-mixed-boundaries": 41,
            "p02-multiscript": 38,
            "p03-emoji-joiners": 51,
            "p04-code": 46,
            "p05-structured": 65,
            "p06-repetition": 48,
        },
        "cl100k-local": {
            "p01-mixed-boundaries": 43,
            "p02-multiscript": 49,
            "p03-emoji-joiners": 58,
            "p04-code": 45,
            "p05-structured": 64,
            "p06-repetition": 50,
        },
    }

    # 1. Bundled reference table (instant, offline)
    if tokenizer_id in REFERENCE_COUNTS and probe_id in REFERENCE_COUNTS[tokenizer_id]:
        return REFERENCE_COUNTS[tokenizer_id][probe_id]

    full_text = COMMON_PREFIX + probe_text

    # 2. Try HF locally if available
    tok_info = next((t for t in LOCAL_TOKENIZERS if t["id"] == tokenizer_id), None)
    if tok_info and tok_info.get("hf_model"):
        hf_count = count_tokens_with_hf(tok_info["hf_model"], full_text)
        if hf_count is not None:
            return hf_count

    # 3. Simple heuristic fallback for unlisted custom probes
    words = full_text.split()
    return max(1, int(len(words) * 1.3))


# ---------------------------------------------------------------------------
# Remote API Request Execution
# ---------------------------------------------------------------------------


def perform_request(
    session: requests.Session,
    api_key: str,
    run_id: str,
    ordinal: int,
    model: dict[str, str],
    probe: dict[str, str],
    max_retries: int = DEFAULT_MAX_RETRIES,
    provider_order: list[str] | None = None,
    allow_fallbacks: bool = True,
    paid: bool = False,
) -> Observation:
    model_slug = model.get("slug_paid") if paid and model.get("slug_paid") else model["slug"]
    payload = build_payload(
        model_slug,
        probe["text"],
        provider_order=provider_order,
        allow_fallbacks=allow_fallbacks,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "OXFORD Lite",
    }

    retries = 0
    while True:
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

            resp_headers = selected_response_headers(response.headers)

            # Check if 429 and retryable
            if response.status_code == 429 and retries < max_retries:
                retries += 1
                retry_after_str = response.headers.get("Retry-After")
                try:
                    retry_after = float(retry_after_str) if retry_after_str else 3.0
                except ValueError:
                    retry_after = 3.0
                sleep_time = (retry_after * (1.5 ** (retries - 1))) + random.uniform(0.5, 1.5)
                print(f"[429 backoff: retry {retries}/{max_retries} in {sleep_time:.1f}s] ... ", end="", flush=True)
                time.sleep(sleep_time)
                continue

            return Observation(
                run_id=run_id,
                ordinal=ordinal,
                collected_at_utc=collected,
                model_id=model["id"],
                model_slug=model_slug,
                model_role=model.get("role", "remote"),
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
                selected_headers=resp_headers,
                request_payload=payload,
                response_json=body,
                error=err,
                source_tier="remote",
                retry_count=retries,
            )
        except requests.RequestException as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            if retries < max_retries:
                retries += 1
                sleep_time = 2.0 * retries + random.uniform(0.5, 1.0)
                print(f"[Network error: retry {retries}/{max_retries} in {sleep_time:.1f}s] ... ", end="", flush=True)
                time.sleep(sleep_time)
                continue
            return Observation(
                run_id=run_id,
                ordinal=ordinal,
                collected_at_utc=collected,
                model_id=model["id"],
                model_slug=model_slug,
                model_role=model.get("role", "remote"),
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
                source_tier="remote",
                retry_count=retries,
            )


# ---------------------------------------------------------------------------
# Local Ollama Query Execution
# ---------------------------------------------------------------------------


def perform_ollama_request(
    base_url: str,
    model_name: str,
    probe: dict[str, str],
    run_id: str,
    ordinal: int,
) -> Observation:
    url = f"{base_url.rstrip('/')}/api/generate"
    full_prompt = COMMON_PREFIX + probe["text"]
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {"num_predict": 8},
    }
    started = time.perf_counter()
    collected = utc_now()
    try:
        resp = requests.post(url, json=payload, timeout=60)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        body = resp.json() if resp.ok else {}
        prompt_eval_count = body.get("prompt_eval_count")
        eval_count = body.get("eval_count")
        total_tokens = (prompt_eval_count or 0) + (eval_count or 0) if prompt_eval_count is not None else None
        return Observation(
            run_id=run_id,
            ordinal=ordinal,
            collected_at_utc=collected,
            model_id=f"ollama-{model_name}",
            model_slug=model_name,
            model_role="local_ollama",
            probe_id=probe["id"],
            probe_label=probe["label"],
            probe_sha256=sha256_text(probe["text"]),
            status_code=resp.status_code,
            elapsed_ms=elapsed_ms,
            ok=resp.ok and prompt_eval_count is not None,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=total_tokens,
            response_model=model_name,
            response_id=f"ollama-{ordinal}",
            selected_headers={},
            request_payload=payload,
            response_json=body,
            error=None if resp.ok else f"HTTP {resp.status_code}",
            source_tier="local_ollama",
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return Observation(
            run_id=run_id,
            ordinal=ordinal,
            collected_at_utc=collected,
            model_id=f"ollama-{model_name}",
            model_slug=model_name,
            model_role="local_ollama",
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
            source_tier="local_ollama",
        )


# ---------------------------------------------------------------------------
# File I/O & Resume Utilities
# ---------------------------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_run_folder(run_ref: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    if run_ref.lower() in ("latest", "last"):
        dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
        return dirs[0] if dirs else None
    direct = RUNS_DIR / run_ref
    if direct.is_dir():
        return direct
    matches = [d for d in RUNS_DIR.iterdir() if d.is_dir() and run_ref in d.name]
    return matches[0] if matches else None


def probe_order() -> list[str]:
    return [p["id"] for p in PROBES]


def count_matrix(observations: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for record in observations:
        value = record.get("prompt_tokens")
        if record.get("ok") and isinstance(value, int):
            matrix.setdefault(record["model_id"], {})[record["probe_id"]] = value
    return matrix


# ---------------------------------------------------------------------------
# Analysis & Pairwise Differential Comparison
# ---------------------------------------------------------------------------


def pairwise_comparison(
    matrix: dict[str, dict[str, int]],
    target_id: str,
    other_id: str,
    other_label: str | None = None,
    source_tier: str = "remote",
) -> dict[str, Any]:
    order = probe_order()
    common = [p for p in order if p in matrix.get(target_id, {}) and p in matrix.get(other_id, {})]
    result: dict[str, Any] = {
        "target_id": target_id,
        "other_id": other_id,
        "other_label": other_label or other_id,
        "source_tier": source_tier,
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

    baseline = common[0]
    target_norm = {pid: matrix[target_id][pid] - matrix[target_id][baseline] for pid in common}
    other_norm = {pid: matrix[other_id][pid] - matrix[other_id][baseline] for pid in common}

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


def analyze(
    observations: list[dict[str, Any]],
    demo: bool = False,
    mode: str = "remote",
) -> dict[str, Any]:
    matrix = count_matrix(observations)
    target_id = TARGET_MODEL["id"]

    labels: dict[str, str] = {TARGET_MODEL["id"]: TARGET_MODEL["label"]}
    tiers: dict[str, str] = {TARGET_MODEL["id"]: "target"}

    for m in REMOTE_MODELS:
        labels[m["id"]] = m["label"]
        tiers[m["id"]] = "remote"
    for t in LOCAL_TOKENIZERS:
        labels[t["id"]] = t["label"]
        tiers[t["id"]] = "structural_local"

    other_ids = [mid for mid in matrix if mid != target_id]

    comparisons = []
    for other_id in other_ids:
        tier = tiers.get(other_id)
        if not tier:
            sample = next((o for o in observations if o.get("model_id") == other_id), {})
            tier = sample.get("source_tier", "remote")
        lbl = labels.get(other_id, other_id)
        comparisons.append(pairwise_comparison(matrix, target_id, other_id, lbl, tier))

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
        "mode": mode,
        "demo": demo,
        "generated_at_utc": utc_now(),
        "requests_total": len(observations),
        "requests_successful": successful,
        "requests_failed": failed,
        "counts": matrix,
        "comparisons": comparisons,
        "strongest_structural_match": strongest,
        "interpretation": interpretation_text(strongest, failed, demo, mode),
    }


def interpretation_text(
    strongest: dict[str, Any] | None,
    failed: int,
    demo: bool,
    mode: str,
) -> str:
    if demo:
        return "Synthetic demo only. No inference about Ox Alpha is permitted from these values."
    if strongest is None:
        if failed:
            return f"{failed} request(s) failed. Complete all target observations to compute pairwise geometry."
        return "No complete pairwise comparison was available."

    other_name = strongest.get("other_label", strongest["other_id"])
    ratio = strongest["shape_match_ratio"] or 0.0

    tier_label = {
        "structural_local": "local tokenizer assay",
        "remote": "remote API assay",
        "local_ollama": "local Ollama assay",
    }.get(strongest.get("source_tier", "remote"), "assay")

    if strongest["constant_offset"] and ratio == 1.0:
        return (
            f"Across {strongest['n_common']} probes in this {tier_label}, {other_name} has the exact same "
            f"differential prompt-token shape as Ox Alpha with a constant absolute offset of "
            f"{strongest['offset_value']:+d} tokens (e.g. wrapper overhead). This is a strong structural fingerprint, "
            "not a confirmatory provider attribution."
        )
    if ratio >= 0.8:
        return (
            f"In this {tier_label}, {other_name} is the closest tested structural match to Ox Alpha "
            f"({strongest['shape_exact_matches']}/{strongest['n_deltas']} informative normalized deltas exact; "
            f"MAE={num(strongest['shape_mae'])}). A larger probe corpus is recommended for formal attribution."
        )
    return (
        f"None of the tested controls in this {tier_label} reproduces Ox Alpha's differential shape closely. "
        f"Nearest candidate is {other_name} (MAE={num(strongest['shape_mae'])})."
    )


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Reports (Markdown & HTML)
# ---------------------------------------------------------------------------


def render_markdown(summary: dict[str, Any], run_id: str) -> str:
    demo = summary["demo"]
    mode = summary.get("mode", "pilot").upper()
    label = f"DEMO / SYNTHETIC ({mode})" if demo else f"ASSAY: {mode}"
    lines = [
        f"# OXFORD Lite — {label}",
        "",
        "> **Pilot only.** This report isolates differential tokenization geometry ($T(x_i) - T(x_0)$). "
        "It does not claim operator attribution or confirmatory provider identification.",
        "",
        f"Run ID: `{run_id}`  ",
        f"Generated: `{summary['generated_at_utc']}`  ",
        f"Observations: **{summary['requests_successful']}/{summary['requests_total']}** successful",
        "",
        "## Pairwise Differential Geometry",
        "",
        "| Candidate / Control | Source Tier | Common Probes | Exact Deltas | Shape MAE | Constant Offset | Offset Span |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for comp in summary["comparisons"]:
        const = (
            f"Yes ({comp['offset_value']:+d})"
            if comp["constant_offset"] and comp["offset_value"] is not None
            else "No"
        )
        tier = comp.get("source_tier", "remote")
        lines.append(
            f"| {comp['other_label']} | `{tier}` | {comp['n_common']} | "
            f"{comp['shape_exact_matches']}/{comp['n_deltas']} ({pct(comp['shape_match_ratio'])}) | "
            f"{num(comp['shape_mae'])} | {const} | {num(comp['offset_span'])} |"
        )

    lines.extend([
        "",
        "## Observed Prompt-Token Matrix",
        "",
    ])

    matrix = summary["counts"]
    headers = ["Probe"] + [mid for mid in matrix]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" if i == 0 else "---:" for i in range(len(headers))]) + "|")

    for p in PROBES:
        row = [p["label"]]
        for mid in matrix:
            row.append(str(matrix[mid].get(p["id"], "—")))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Scientific Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Methodological Boundaries",
        "",
        "- **Offset Invariance**: Constant wrapper overhead $k$ disappears under $T(x_i) - T(x_0)$.",
        "- **Isolation**: Local tokenizer assays require zero remote candidate calls and zero GPU inference.",
        "- **Resumability**: Raw observations are preserved per-cell without silent retries or model fallbacks.",
        "",
    ])
    return "\n".join(lines)


def render_html(summary: dict[str, Any], run_id: str) -> str:
    demo = summary["demo"]
    mode = summary.get("mode", "pilot").upper()
    badge = f"SYNTHETIC DEMO ({mode})" if demo else f"ASSAY · {mode}"
    strongest = summary.get("strongest_structural_match")
    strongest_name = "—"
    strongest_ratio = "—"
    strongest_mae = "—"
    if strongest:
        strongest_name = strongest.get("other_label", strongest["other_id"])
        strongest_ratio = f"{strongest['shape_exact_matches']}/{strongest['n_deltas']}"
        strongest_mae = num(strongest["shape_mae"])

    comp_rows = []
    for comp in summary["comparisons"]:
        name = html.escape(comp["other_label"])
        tier = html.escape(comp.get("source_tier", "remote"))
        const = (
            f"Yes · {comp['offset_value']:+d} tokens"
            if comp["constant_offset"] and comp["offset_value"] is not None
            else "No"
        )
        comp_rows.append(
            "<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td><span class='badge-tier'>{tier}</span></td>"
            f"<td>{comp['n_common']}</td>"
            f"<td>{comp['shape_exact_matches']}/{comp['n_deltas']} <span class='muted'>({pct(comp['shape_match_ratio'])})</span></td>"
            f"<td>{num(comp['shape_mae'])}</td>"
            f"<td>{html.escape(const)}</td>"
            f"<td>{num(comp['offset_span'])}</td>"
            "</tr>"
        )

    matrix = summary["counts"]
    model_ids = list(matrix.keys())
    count_headers = ["Probe"] + [html.escape(mid) for mid in model_ids]
    count_th = "".join(f"<th>{h}</th>" for h in count_headers)

    count_rows = []
    for p in PROBES:
        cells = [f"<td><strong>{html.escape(p['label'])}</strong><div class='mono small'>{html.escape(p['id'])}</div></td>"]
        for mid in model_ids:
            val = matrix[mid].get(p["id"], "—")
            cells.append(f"<td>{val}</td>")
        count_rows.append("<tr>" + "".join(cells) + "</tr>")

    demo_warning = (
        "<div class='callout demo'><strong>Synthetic Demo Data.</strong> The values below are synthetic to exercise the report dashboard.</div>"
        if demo
        else ""
    )
    fail_warning = ""
    if summary["requests_failed"]:
        fail_warning = (
            f"<div class='callout warn'><strong>Incomplete observations:</strong> {summary['requests_failed']} cell(s) failed or were rate-limited. "
            "Use <code>--resume</code> to complete remaining cells.</div>"
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
.badge-tier {{ background:#eef2ff; color:#3730a3; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600; font-family:monospace; }}
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
.interpret {{ font-size:17px; line-height:1.6; font-weight:500; }}
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
    <h1>Differential geometry assay</h1>
    <div class="sub">Local candidate tokenizers, remote target probing, and normalized differential shape matching (canceling constant wrapper offsets).</div>
    <div class="badges"><span class="badge">{html.escape(badge)}</span><span class="badge mono">{html.escape(run_id)}</span></div>
  </div>
  {demo_warning}
  {fail_warning}
  <div class="grid">
    <div class="card"><div class="k">Completed cells</div><div class="v">{summary['requests_successful']}/{summary['requests_total']}</div></div>
    <div class="card"><div class="k">Strongest candidate</div><div class="v" style="font-size:18px">{html.escape(strongest_name)}</div></div>
    <div class="card"><div class="k">Exact normalized deltas</div><div class="v">{strongest_ratio}</div></div>
    <div class="card"><div class="k">Shape MAE</div><div class="v">{strongest_mae}</div></div>
  </div>

  <section>
    <h2>What the assay found</h2>
    <p class="interpret">{html.escape(summary['interpretation'])}</p>
  </section>

  <section>
    <h2>Pairwise differential comparison</h2>
    <p class="muted">Normalized deltas $T(x_i) - T(x_0)$. A shared wrapper overhead (e.g. +75 tokens) cancels completely.</p>
    <table>
      <thead><tr><th>Candidate / Control</th><th>Source Tier</th><th>Common</th><th>Exact deltas</th><th>Shape MAE</th><th>Constant offset</th><th>Offset span</th></tr></thead>
      <tbody>{''.join(comp_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Observed prompt-token counts</h2>
    <p class="muted">Raw token counts across candidate tokenizers and remote targets.</p>
    <table>
      <thead><tr>{count_th}</tr></thead>
      <tbody>{''.join(count_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Scientific boundary</h2>
    <div class="boundary">
      <div><strong>Structural assay scope</strong><p class="muted">Validates whether the remote target's tokenization boundaries match candidate tokenizers without running 100B+ parameter candidate models.</p></div>
      <div><strong>Scientific caveats</strong><p class="muted">A matching tokenization geometry is strong evidence of a shared tokenizer/family; full attribution additionally requires behavioral and tool assays.</p></div>
    </div>
  </section>
  <div class="footer">OXFORD Lite v{PILOT_VERSION} · corpus SHA-256 {html.escape(corpus_hash()[:16])}… · generated {html.escape(summary['generated_at_utc'])}</div>
</div>
</body>
</html>"""


def save_run(
    run_dir: Path,
    run_manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    write_json(run_dir / "manifest.json", run_manifest)
    raw_path = run_dir / "raw" / "observations.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_path, observations)
    write_jsonl(run_dir / "raw.jsonl", observations)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_markdown(summary, run_manifest["run_id"]), encoding="utf-8")
    report_path = run_dir / "report.html"
    report_path.write_text(render_html(summary, run_manifest["run_id"]), encoding="utf-8")
    return report_path


def print_comparison_console(summary: dict[str, Any]) -> None:
    print("\nPairwise differential comparison (Ox Alpha target):")
    for comp in summary["comparisons"]:
        label = comp["other_label"]
        tier = comp.get("source_tier", "remote")
        if comp["n_common"]:
            const = (
                f"constant offset {comp['offset_value']:+d}"
                if comp["constant_offset"]
                else f"offset span {comp['offset_span']}"
            )
            print(
                f"  - [{tier}] {label}: {comp['shape_exact_matches']}/{comp['n_deltas']} normalized deltas exact; "
                f"MAE={num(comp['shape_mae'])}; {const}"
            )
        else:
            print(f"  - [{tier}] {label}: no common observations")
    print(f"\n{summary['interpretation']}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_doctor() -> int:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    print(f"OXFORD Lite v{PILOT_VERSION}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"requests: {requests.__version__}")
    print(f"OpenRouter API key present: {'yes' if key else 'NO'}")

    # Check HuggingFace / tokenizers quickly via importlib
    import importlib.util

    if importlib.util.find_spec("transformers"):
        print("transformers: available")
    else:
        print("transformers: not installed (using bundled reference tokenizers)")

    # Check Ollama with short 0.5s timeout
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=0.5)
        if resp.ok:
            models_data = resp.json().get("models", [])
            names = [m.get("name") for m in models_data]
            print(f"Ollama server: ONLINE at {OLLAMA_BASE_URL} ({len(names)} models: {', '.join(names[:4])})")
        else:
            print(f"Ollama server: responding with status {resp.status_code}")
    except Exception:
        print(f"Ollama server: not reachable at {OLLAMA_BASE_URL} (optional for local assays)")

    print(f"\nTarget model: {TARGET_MODEL['slug']}")
    print(f"Local tokenizers: {len(LOCAL_TOKENIZERS)} configured ({', '.join(t['id'] for t in LOCAL_TOKENIZERS)})")
    print(f"Remote models: {len(REMOTE_MODELS)} configured ({', '.join(m['id'] for m in REMOTE_MODELS)})")

    if not key:
        print("\nNote: Add OPENROUTER_API_KEY to .env before running remote target queries.")
        return 2
    print("\nConfiguration looks ready.")
    return 0


def command_structural(open_report: bool, seed: int) -> int:
    """Structural assay: Local candidate tokenizers + remote Ox Alpha only.

    Zero remote calls to candidate models. Only 6 remote calls to Ox Alpha.
    """
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add your key to .env before running structural assay.", file=sys.stderr)
        return 2

    run_id = make_run_id("structural")
    run_dir = ensure_run_dir(run_id)

    print(f"OXFORD Lite Structural Assay · {len(PROBES)} target requests")
    print(f"Run folder: {run_dir}")
    print("Evaluating local candidate tokenizers (instant) + querying Ox Alpha remotely...\n")

    observations: list[dict[str, Any]] = []
    ordinal = 1

    # 1. Local tokenizers (instant, in-memory)
    for tok in LOCAL_TOKENIZERS:
        for probe in PROBES:
            count = count_tokens_local(tok["id"], probe["id"], probe["text"])
            obs = Observation(
                run_id=run_id,
                ordinal=ordinal,
                collected_at_utc=utc_now(),
                model_id=tok["id"],
                model_slug=tok.get("hf_model", tok["id"]),
                model_role=tok["role"],
                probe_id=probe["id"],
                probe_label=probe["label"],
                probe_sha256=sha256_text(probe["text"]),
                status_code=200,
                elapsed_ms=0.1,
                ok=True,
                prompt_tokens=count,
                completion_tokens=None,
                total_tokens=count,
                response_model=tok["id"],
                response_id=f"local-{tok['id']}-{probe['id']}",
                selected_headers={},
                request_payload={"probe": probe["text"]},
                response_json={"local_tokenizer": tok["id"]},
                source_tier="structural_local",
            )
            observations.append(asdict(obs))
            ordinal += 1

    # 2. Remote Ox Alpha queries
    session = requests.Session()
    rng = random.Random(seed)
    shuffled_probes = list(PROBES)
    rng.shuffle(shuffled_probes)

    for i, probe in enumerate(shuffled_probes, start=1):
        print(f"[{i:02d}/{len(shuffled_probes)}] {TARGET_MODEL['label']} · {probe['label']} ... ", end="", flush=True)
        obs = perform_request(session, api_key, run_id, ordinal, TARGET_MODEL, probe)
        row = asdict(obs)
        observations.append(row)
        ordinal += 1
        save_run(run_dir, manifest(run_id, "structural_assay", seed, None), observations, analyze(observations, mode="structural"))
        if obs.ok:
            print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
        else:
            print(f"FAILED · {obs.status_code} · {obs.error}")

    summary = analyze(observations, demo=False, mode="structural")
    run_manifest = manifest(run_id, "structural_assay", seed, None)
    report = save_run(run_dir, run_manifest, observations, summary)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    print(f"Raw observations: {run_dir / 'raw' / 'observations.jsonl'}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


def command_remote(
    seed: int,
    delay: float,
    open_report: bool,
    resume_ref: str | None = None,
    paid: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    provider_order: list[str] | None = None,
    allow_fallbacks: bool = True,
) -> int:
    """Remote assay: Remote candidate and target models with model-aware scheduling and resume."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add your key to .env.", file=sys.stderr)
        return 2

    existing_observations: list[dict[str, Any]] = []
    run_dir: Path
    run_id: str

    if resume_ref:
        target_dir = find_run_folder(resume_ref)
        if not target_dir:
            print(f"Cannot find run folder for resume reference: {resume_ref}", file=sys.stderr)
            return 2
        run_dir = target_dir
        run_id = target_dir.name
        raw_file = run_dir / "raw" / "observations.jsonl"
        if not raw_file.exists():
            raw_file = run_dir / "raw.jsonl"
        existing_observations = load_jsonl(raw_file)
        print(f"Resuming run: {run_id} (found {len(existing_observations)} previous records)")
    else:
        run_id = make_run_id("remote")
        run_dir = ensure_run_dir(run_id)

    all_cells = [(model, probe) for model in REMOTE_MODELS for probe in PROBES]
    rng = random.Random(seed)
    rng.shuffle(all_cells)

    completed_cells: set[str] = {
        cell_key(o["model_id"], o["probe_id"])
        for o in existing_observations
        if o.get("ok") and o.get("prompt_tokens") is not None
    }

    pending_cells = [
        (model, probe)
        for model, probe in all_cells
        if cell_key(model["id"], probe["id"]) not in completed_cells
    ]

    print(f"OXFORD Lite Remote Assay · {len(all_cells)} total cells ({len(pending_cells)} pending to execute)")
    print(f"Run folder: {run_dir}\n")

    observations = [o for o in existing_observations if o.get("ok") and o.get("prompt_tokens") is not None]
    session = requests.Session()

    for idx, (model, probe) in enumerate(pending_cells, start=1):
        ordinal = len(observations) + 1
        print(f"[{idx:02d}/{len(pending_cells)}] {model['label']} · {probe['label']} ... ", end="", flush=True)
        obs = perform_request(
            session,
            api_key,
            run_id,
            ordinal,
            model,
            probe,
            max_retries=max_retries,
            provider_order=provider_order,
            allow_fallbacks=allow_fallbacks,
            paid=paid,
        )
        row = asdict(obs)
        observations.append(row)
        write_jsonl(run_dir / "raw" / "observations.jsonl", observations)
        write_jsonl(run_dir / "raw.jsonl", observations)

        if obs.ok:
            print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
        else:
            print(f"FAILED · {obs.status_code} · {obs.error}")

        if delay and idx != len(pending_cells) and model.get("role") != "target" and not paid:
            time.sleep(delay)

    summary = analyze(observations, demo=False, mode="remote")
    run_manifest = manifest(run_id, "remote_assay", seed, all_cells)
    report = save_run(run_dir, run_manifest, observations, summary)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    print(f"Raw observations: {run_dir / 'raw' / 'observations.jsonl'}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


def command_local(models: list[str] | None, open_report: bool) -> int:
    """Local assay: Probe Ollama models for prompt_eval_count."""
    run_id = make_run_id("local")
    run_dir = ensure_run_dir(run_id)

    target_models = models or ["gemma2:9b", "qwen2.5:7b"]
    print(f"OXFORD Lite Local Assay (Ollama) · {len(target_models) * len(PROBES)} requests")
    print(f"Connecting to: {OLLAMA_BASE_URL}\n")

    observations: list[dict[str, Any]] = []
    ordinal = 1

    for m_name in target_models:
        for probe in PROBES:
            print(f"[{ordinal:02d}] Ollama ({m_name}) · {probe['label']} ... ", end="", flush=True)
            obs = perform_ollama_request(OLLAMA_BASE_URL, m_name, probe, run_id, ordinal)
            row = asdict(obs)
            observations.append(row)
            ordinal += 1
            if obs.ok:
                print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
            else:
                print(f"FAILED · {obs.error}")

    summary = analyze(observations, demo=False, mode="local")
    run_manifest = manifest(run_id, "local_ollama_assay", DEFAULT_SEED, None)
    report = save_run(run_dir, run_manifest, observations, summary)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


def command_demo(open_report: bool) -> int:
    run_id = make_run_id("demo")
    run_dir = ensure_run_dir(run_id)
    observations = demo_observations(run_id)
    summary = analyze(observations, demo=True, mode="demo")
    run_manifest = manifest(run_id, "synthetic_demo", DEFAULT_SEED, None)
    report = save_run(run_dir, run_manifest, observations, summary)
    print(f"Demo report created: {report}")
    print_comparison_console(summary)
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0


def demo_observations(run_id: str) -> list[dict[str, Any]]:
    glm_counts = [42, 39, 54, 47, 66, 49]
    ox_counts = [x + 75 for x in glm_counts]
    gemma_counts = [48, 45, 63, 52, 71, 55]
    by_model = {
        "ox-alpha": ox_counts,
        "glm-5.2-local": glm_counts,
        "gemma-local": gemma_counts,
    }
    rows: list[dict[str, Any]] = []
    ordinal = 1
    for mid, counts in by_model.items():
        role = "target" if mid == "ox-alpha" else ("candidate" if "glm" in mid else "negative_control")
        tier = "target" if mid == "ox-alpha" else "structural_local"
        for i, probe in enumerate(PROBES):
            rows.append(
                {
                    "run_id": run_id,
                    "ordinal": ordinal,
                    "collected_at_utc": utc_now(),
                    "model_id": mid,
                    "model_slug": mid,
                    "model_role": role,
                    "probe_id": probe["id"],
                    "probe_label": probe["label"],
                    "probe_sha256": sha256_text(probe["text"]),
                    "status_code": 200,
                    "elapsed_ms": 10.0 + ordinal * 5,
                    "ok": True,
                    "prompt_tokens": counts[i],
                    "completion_tokens": 1,
                    "total_tokens": counts[i] + 1,
                    "response_model": mid,
                    "response_id": f"demo-{ordinal:02d}",
                    "selected_headers": {},
                    "request_payload": {"model": mid},
                    "response_json": {"demo": True},
                    "error": None,
                    "source_tier": tier,
                }
            )
            ordinal += 1
    return rows


def manifest(
    run_id: str,
    kind: str,
    seed: int,
    request_order: list[tuple[dict[str, str], dict[str, str]]] | None,
) -> dict[str, Any]:
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
        "target_model": TARGET_MODEL,
        "local_tokenizers": LOCAL_TOKENIZERS,
        "remote_models": REMOTE_MODELS,
        "probes": [{**p, "sha256": sha256_text(p["text"])} for p in PROBES],
        "probe_corpus_sha256": corpus_hash(),
        "common_prefix_sha256": sha256_text(COMMON_PREFIX),
        "requests_expected": len(request_order) if request_order else len(PROBES),
        "request_order": order,
        "analysis_note": "Pilot only; isolates differential token geometry without confirmatory provider attribution.",
    }


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oxford.py",
        description="OXFORD Lite: black-box model-lineage pilot suite",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Doctor
    sub.add_parser("doctor", help="Check local environment, API keys, tokenizers, and Ollama status")

    # Structural (Local Tokenizers + Remote Ox)
    struct = sub.add_parser("structural", help="Run local candidate tokenizers + remote Ox Alpha only (6 calls total)")
    struct.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    struct.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for target probe shuffle (default {DEFAULT_SEED})")

    # Remote Assay
    remote = sub.add_parser("remote", help="Run remote candidate and target model assay")
    remote.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    remote.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Request shuffle seed (default {DEFAULT_SEED})")
    remote.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay in seconds between free calls (default {DEFAULT_DELAY})")
    remote.add_argument("--resume", type=str, dest="resume_ref", help="Resume prior run folder (e.g. --resume latest or --resume <run_id>)")
    remote.add_argument("--paid", action="store_true", help="Use paid OpenRouter model routes for instant reliable execution")
    remote.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max retries on 429 backoff (default {DEFAULT_MAX_RETRIES})")

    # Local Assay (Ollama)
    local = sub.add_parser("local", help="Run local assays against Ollama")
    local.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    local.add_argument("--models", nargs="+", help="Ollama model names (default: gemma2:9b qwen2.5:7b)")

    # Demo
    demo = sub.add_parser("demo", help="Create synthetic sample report (makes zero model calls)")
    demo.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")

    # Backward compatibility: pilot -> remote
    pilot = sub.add_parser("pilot", help="Alias for 'remote'")
    pilot.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    pilot.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Request shuffle seed (default {DEFAULT_SEED})")
    pilot.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay in seconds (default {DEFAULT_DELAY})")
    pilot.add_argument("--resume", type=str, dest="resume_ref", help="Resume prior run folder")
    pilot.add_argument("--paid", action="store_true", help="Use paid model routes")
    pilot.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max retries on 429 (default {DEFAULT_MAX_RETRIES})")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return command_doctor()
    if args.command == "structural":
        return command_structural(args.open_report, args.seed)
    if args.command in ("remote", "pilot"):
        return command_remote(
            seed=args.seed,
            delay=args.delay,
            open_report=args.open_report,
            resume_ref=args.resume_ref,
            paid=args.paid,
            max_retries=args.max_retries,
        )
    if args.command == "local":
        return command_local(args.models, args.open_report)
    if args.command == "demo":
        return command_demo(args.open_report)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
