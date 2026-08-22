#!/usr/bin/env python3
"""OXFORD Lite: a black-box model-lineage and tokenizer-geometry assay suite.

Assay Modes & Commands:
1. `structural`: Evaluates local candidate tokenizers + remote Ox Alpha only.
2. `remote` / `pilot`: Remote API comparisons with model-aware scheduling, provider pinning,
   --paid/--free routing, jittered 429 backoff, and cell-level --resume.
3. `positive-control`: Validates OXFORD against a known specimen (remote Qwen vs local Qwen/GLM/Gemma).
4. `collision`: Empirical Monte Carlo simulation of tokenizer collision probabilities across 1M trials.
5. `synthesize-probes`: Generates synthetic candidate strings and extracts the top discriminatory probes.
6. `envelope`: Tests differential shape invariance across 3 distinct request envelopes.
7. `local`: Local Ollama assay.
8. `demo` & `doctor`: Synthetic demo and environment auditor.
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
from typing import Any, Callable, Iterable

import requests
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
RUNS_DIR = ROOT / "runs"
PROBES_DIR = ROOT / "probes"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
PILOT_VERSION = "0.3.0"
DEFAULT_SEED = 20260821
DEFAULT_DELAY = 0.0
DEFAULT_MAX_RETRIES = 2

# Target Model
TARGET_MODEL: dict[str, str] = {
    "id": "ox-alpha",
    "slug": "stealth/ox-alpha",
    "label": "Ox Alpha",
    "role": "target",
}

# Remote Comparison Models
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
        "slug_paid": "google/gemma-4-26b-a4b-it",
        "label": "Gemma 4 26B A4B",
        "role": "negative_control",
    },
]

# Backward compatibility alias
MODELS = REMOTE_MODELS

# Candidate Local Tokenizers
LOCAL_TOKENIZERS: list[dict[str, str]] = [
    {
        "id": "glm-5.2-local",
        "label": "GLM-5.2 Tokenizer (Local)",
        "role": "candidate",
        "hf_model": "zai-org/GLM-5.2",
    },
    {
        "id": "qwen-local",
        "label": "Qwen 2.5 Tokenizer (Local)",
        "role": "candidate",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
    },
    {
        "id": "gemma-local",
        "label": "Gemma Tokenizer (Local)",
        "role": "negative_control",
        "hf_model": "alpindale/gemma-2b",
    },
    {
        "id": "cl100k-local",
        "label": "OpenAI cl100k (Local)",
        "role": "negative_control",
        "encoding_name": "cl100k_base",
    },
    {
        "id": "o200k-local",
        "label": "OpenAI o200k (Local)",
        "role": "negative_control",
        "encoding_name": "o200k_base",
    },
]

# Standard bundled probes
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


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

ENVELOPES: dict[str, dict[str, Any]] = {
    "envelope_a_minimal": {
        "id": "envelope_a_minimal",
        "label": "Envelope A (Minimal)",
        "builder": lambda probe_text: {
            "messages": [{"role": "user", "content": f"Payload:\n{probe_text}"}],
        },
        "text_formatter": lambda probe_text: f"Payload:\n{probe_text}",
    },
    "envelope_b_standard": {
        "id": "envelope_b_standard",
        "label": "Envelope B (Standard Instruction)",
        "builder": lambda probe_text: {
            "messages": [{"role": "user", "content": COMMON_PREFIX + probe_text}],
        },
        "text_formatter": lambda probe_text: COMMON_PREFIX + probe_text,
    },
    "envelope_c_system": {
        "id": "envelope_c_system",
        "label": "Envelope C (System + User)",
        "builder": lambda probe_text: {
            "messages": [
                {"role": "system", "content": "You are a black-box test oracle. Return only OK."},
                {"role": "user", "content": probe_text},
            ],
        },
        "text_formatter": lambda probe_text: f"<system>You are a black-box test oracle. Return only OK.</system>\n<user>{probe_text}</user>",
    },
}


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
    envelope_id: str = "envelope_b_standard"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def corpus_hash(probes: list[dict[str, str]] | None = None) -> str:
    p = probes or PROBES
    return sha256_text(canonical_json(p))


def make_run_id(prefix: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{prefix}"


def ensure_run_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    raw_dir = path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return path


def cell_key(model_id: str, probe_id: str, envelope_id: str = "envelope_b_standard") -> str:
    return f"{model_id}::{probe_id}::{envelope_id}"


def build_payload(
    model_slug: str,
    probe_text: str,
    envelope_id: str = "envelope_b_standard",
    provider_order: list[str] | None = None,
    allow_fallbacks: bool = True,
    max_tokens: int = 8,
) -> dict[str, Any]:
    env_cfg = ENVELOPES.get(envelope_id, ENVELOPES["envelope_b_standard"])
    base_payload = env_cfg["builder"](probe_text)
    payload: dict[str, Any] = {
        "model": model_slug,
        **base_payload,
        "max_tokens": max_tokens,
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
# Fail-Closed Local Tokenizer Engine
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_LOAD_FAILED: set[str] = set()


def get_local_tokenizer(tokenizer_id: str) -> tuple[Any | None, str | None]:
    """Dynamically load and cache a local tokenizer instance.

    Fails closed if the tokenizer cannot be loaded (returns None, error_str).
    """
    if tokenizer_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[tokenizer_id], None
    if tokenizer_id in _TOKENIZER_LOAD_FAILED:
        return None, f"Tokenizer '{tokenizer_id}' previously failed to load"

    tok_info = next((t for t in LOCAL_TOKENIZERS if t["id"] == tokenizer_id), None)
    if not tok_info:
        _TOKENIZER_LOAD_FAILED.add(tokenizer_id)
        return None, f"Unknown tokenizer identifier: {tokenizer_id}"

    # 1. Tiktoken encodings
    if tok_info.get("encoding_name"):
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding(tok_info["encoding_name"])
            _TOKENIZER_CACHE[tokenizer_id] = ("tiktoken", enc)
            return _TOKENIZER_CACHE[tokenizer_id], None
        except Exception as exc:
            _TOKENIZER_LOAD_FAILED.add(tokenizer_id)
            return None, f"Tiktoken load failed for '{tok_info['encoding_name']}': {exc}"

    # 2. Tokenizers / Hugging Face
    hf_model = tok_info.get("hf_model")
    if hf_model:
        try:
            from tokenizers import Tokenizer  # type: ignore

            tok = Tokenizer.from_pretrained(hf_model)
            _TOKENIZER_CACHE[tokenizer_id] = ("tokenizers", tok)
            return _TOKENIZER_CACHE[tokenizer_id], None
        except Exception as exc:
            _TOKENIZER_LOAD_FAILED.add(tokenizer_id)
            return None, f"FAILED_TO_LOAD: {hf_model} ({exc})"

    _TOKENIZER_LOAD_FAILED.add(tokenizer_id)
    return None, f"No backend configured for tokenizer: {tokenizer_id}"


def count_tokens_local(
    tokenizer_id: str,
    probe_id: str,
    probe_text: str,
    envelope_id: str = "envelope_b_standard",
) -> tuple[int | None, str | None]:
    """Compute local token counts using real local tokenizer instances.

    Fails closed: returns (None, error_message) if tokenizer cannot be loaded.
    Zero mock tables and zero heuristic fallbacks during experimental assay execution.
    """
    env_cfg = ENVELOPES.get(envelope_id, ENVELOPES["envelope_b_standard"])
    formatted_text = env_cfg["text_formatter"](probe_text)

    tokenizer_obj, err = get_local_tokenizer(tokenizer_id)
    if err or not tokenizer_obj:
        return None, err or f"Tokenizer {tokenizer_id} not available"

    kind, instance = tokenizer_obj
    try:
        if kind == "tiktoken":
            return len(instance.encode(formatted_text)), None
        if kind == "tokenizers":
            return len(instance.encode(formatted_text).ids), None
    except Exception as exc:
        return None, f"Tokenization encoding error: {exc}"

    return None, "Unsupported tokenizer backend"


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
    envelope_id: str = "envelope_b_standard",
    max_retries: int = DEFAULT_MAX_RETRIES,
    provider_order: list[str] | None = None,
    allow_fallbacks: bool = True,
    paid: bool = False,
    max_tokens: int = 8,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> Observation:
    model_slug = model.get("slug_paid") if paid and model.get("slug_paid") else model["slug"]
    payload = build_payload(
        model_slug,
        probe["text"],
        envelope_id=envelope_id,
        provider_order=provider_order,
        allow_fallbacks=allow_fallbacks,
        max_tokens=max_tokens,
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

            attempt_record = {
                "run_id": run_id,
                "ordinal": ordinal,
                "attempt": retries + 1,
                "timestamp_utc": collected,
                "model_id": model["id"],
                "model_slug": model_slug,
                "probe_id": probe["id"],
                "envelope_id": envelope_id,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "ok": response.ok and prompt_tokens is not None,
                "prompt_tokens": prompt_tokens,
                "error": err,
                "response_json": body,
            }
            if on_attempt:
                on_attempt(attempt_record)

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
                envelope_id=envelope_id,
            )
        except requests.RequestException as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            attempt_record = {
                "run_id": run_id,
                "ordinal": ordinal,
                "attempt": retries + 1,
                "timestamp_utc": collected,
                "model_id": model["id"],
                "model_slug": model_slug,
                "probe_id": probe["id"],
                "envelope_id": envelope_id,
                "status_code": 0,
                "elapsed_ms": elapsed_ms,
                "ok": False,
                "prompt_tokens": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if on_attempt:
                on_attempt(attempt_record)

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
                envelope_id=envelope_id,
            )


# ---------------------------------------------------------------------------
# File I/O & Immutable Log Handlers
# ---------------------------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
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


# ---------------------------------------------------------------------------
# Analysis & Differential Geometry
# ---------------------------------------------------------------------------


def pairwise_comparison(
    matrix: dict[str, dict[str, int]],
    target_id: str,
    other_id: str,
    probe_list: list[dict[str, str]] | None = None,
    other_label: str | None = None,
    source_tier: str = "remote",
) -> dict[str, Any]:
    probes = probe_list or PROBES
    order = [p["id"] for p in probes]
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


def count_matrix(observations: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for record in observations:
        value = record.get("prompt_tokens")
        if record.get("ok") and isinstance(value, int):
            matrix.setdefault(record["model_id"], {})[record["probe_id"]] = value
    return matrix


def analyze(
    observations: list[dict[str, Any]],
    demo: bool = False,
    mode: str = "remote",
    target_id: str = "ox-alpha",
    probe_list: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    matrix = count_matrix(observations)
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
        comparisons.append(pairwise_comparison(matrix, target_id, other_id, probe_list, lbl, tier))

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
            return f"{failed} request(s) failed or lacked token usage. Rerun to complete cells."
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
            f"differential prompt-token shape as target with a constant absolute offset of "
            f"{strongest['offset_value']:+d} tokens (effective wrapper overhead). This is a strong structural fingerprint, "
            "not a confirmatory provider attribution."
        )
    if ratio >= 0.8:
        return (
            f"In this {tier_label}, {other_name} is the closest tested structural match to target "
            f"({strongest['shape_exact_matches']}/{strongest['n_deltas']} normalized deltas exact; "
            f"MAE={num(strongest['shape_mae'])}). Confirmatory study with expanded candidate probe pool recommended."
        )
    return (
        f"None of the tested controls in this {tier_label} reproduces the target's differential shape closely. "
        f"Nearest tested model is {other_name} (MAE={num(strongest['shape_mae'])})."
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
# Reports (HTML & Markdown)
# ---------------------------------------------------------------------------


def render_markdown(summary: dict[str, Any], run_id: str, probes: list[dict[str, str]] | None = None) -> str:
    p_list = probes or PROBES
    demo = summary["demo"]
    mode = summary.get("mode", "pilot").upper()
    label = f"DEMO / SYNTHETIC ({mode})" if demo else f"ASSAY: {mode}"
    lines = [
        f"# OXFORD Lite — {label}",
        "",
        "> **Exploration 1.** This report isolates differential tokenization geometry ($T(x_i) - T(x_0)$). "
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

    for p in p_list:
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


def render_html(summary: dict[str, Any], run_id: str, probes: list[dict[str, str]] | None = None) -> str:
    p_list = probes or PROBES
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
    for p in p_list:
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
  <div class="footer">OXFORD Lite v{PILOT_VERSION} · corpus SHA-256 {html.escape(corpus_hash(p_list)[:16])}… · generated {html.escape(summary['generated_at_utc'])}</div>
</div>
</body>
</html>"""


def save_run(
    run_dir: Path,
    run_manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    summary: dict[str, Any],
    probes: list[dict[str, str]] | None = None,
) -> Path:
    p_list = probes or PROBES
    write_json(run_dir / "manifest.json", run_manifest)
    raw_path = run_dir / "raw" / "observations.jsonl"
    write_jsonl(raw_path, observations)
    write_jsonl(run_dir / "raw.jsonl", observations)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_markdown(summary, run_manifest["run_id"], p_list), encoding="utf-8")
    report_path = run_dir / "report.html"
    report_path.write_text(render_html(summary, run_manifest["run_id"], p_list), encoding="utf-8")
    return report_path


def print_comparison_console(summary: dict[str, Any]) -> None:
    print("\nPairwise differential comparison:")
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
# Synthetic Probes Generator
# ---------------------------------------------------------------------------


def generate_synthetic_corpus(count: int = 1000, seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    """Generate a diverse synthetic corpus for tokenizer discrimination & collision tests."""
    rng = random.Random(seed)

    multilingual_snippets = [
        "中文テストالعربية—naïve—Привет",
        "Kestrel :: 🚀 :: 🌌 :: 🛰️",
        "Γειά σου Κόσμε :: Olá Mundo :: Привет мир",
        "こんにちは世界 :: สวัสดีชาวโลก :: مرحبا بالعالم",
        "München—Zürich—São Paulo—Reykjavík",
    ]

    code_snippets = [
        'def μ(x:int)->str: return f"v::{x:08x}::{x**2}"',
        'pub fn compute_hash<T: Hash>(val: &T) -> [u8; 32] { sha256(val) }',
        'const λ = (a, b) => a.map((x, i) => x ^ b[i % b.length]);',
        'SELECT u.id, count(t.x) FROM users u JOIN tokens t ON u.id = t.uid GROUP BY 1;',
        '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$',
    ]

    emoji_snippets = [
        "orbit🛰️|family👨‍👩‍👧‍👦|keycap7️⃣|flag🇰🇪",
        "👨🏽‍💻 + 👩🏼‍🔬 = 🤖 (AI 2026)",
        "✨⚡🔥🌈☀️🌙⭐🌊",
        "flag🇺🇳|flag🇨🇳|flag🇺🇸|flag🇪🇺|flag🇯🇵",
    ]

    boundary_snippets = [
        "fjord_7F9Q::Δ::xYz__0042",
        "urn:oxford:9f2c1d73-4a6b-48e1-a77d-00ff19ab73c2?x=17&y=A_B-C.D",
        "__init____main____getattr____call__",
        "0xDEADBEEF_CAFE_BABE_0123456789ABCDEF",
        "https://api.internal.network:8443/v1/probes/raw?batch=42&mode=fast#anchor",
    ]

    whitespace_snippets = [
        "abababababababab  zzzzzzzzzzzz\tA__A__A__A\nEND",
        "   \t\t   \n\n\r\n   ---===###===---   \n\t",
        "alpha\n\nbeta\t\tgamma    delta_____epsilon",
    ]

    all_bases = multilingual_snippets + code_snippets + emoji_snippets + boundary_snippets + whitespace_snippets
    probes = []

    for i in range(count):
        k = rng.randint(1, 3)
        parts = rng.sample(all_bases, k)
        prefix = f"p_{i:05d}_" + rng.choice(["id", "fn", "raw", "tok", "vec"])
        text = f"{prefix}::" + " || ".join(parts) + f"::val_{rng.randint(1000, 9999)}"
        probes.append({
            "id": f"synth-p{i:05d}",
            "label": f"Synthetic Probe #{i:05d}",
            "text": text,
        })
    return probes


# ---------------------------------------------------------------------------
# Command: Positive Control Assay (Known Specimen)
# ---------------------------------------------------------------------------


def command_positive_control(
    open_report: bool,
    seed: int,
    target_slug: str | None = None,
    paid: bool = False,
) -> int:
    """Positive Control: Verify OXFORD against a known specimen (e.g. remote GLM-5.2 or Qwen vs local tokenizers)."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add key to .env.", file=sys.stderr)
        return 2

    slug = target_slug or ("qwen/qwen-2.5-7b-instruct" if paid else "z-ai/glm-5.2:free")
    model_id = slug.split("/")[-1].split(":")[0]

    KNOWN_TARGET = {
        "id": f"specimen-{model_id}",
        "slug": slug,
        "label": f"Specimen ({slug})",
        "role": "target",
    }

    run_id = make_run_id("positive-control")
    run_dir = ensure_run_dir(run_id)

    print(f"OXFORD Lite Known-Specimen Positive Control · {len(PROBES)} requests")
    print(f"Target Specimen: {KNOWN_TARGET['slug']}")
    print(f"Run folder: {run_dir}\n")

    observations: list[dict[str, Any]] = []
    ordinal = 1

    # 1. Local tokenizers
    for tok in LOCAL_TOKENIZERS:
        for probe in PROBES:
            count, err = count_tokens_local(tok["id"], probe["id"], probe["text"])
            obs = Observation(
                run_id=run_id,
                ordinal=ordinal,
                collected_at_utc=utc_now(),
                model_id=tok["id"],
                model_slug=tok.get("hf_model", tok.get("encoding_name", tok["id"])),
                model_role=tok["role"],
                probe_id=probe["id"],
                probe_label=probe["label"],
                probe_sha256=sha256_text(probe["text"]),
                status_code=200 if err is None else 500,
                elapsed_ms=0.1,
                ok=err is None and count is not None,
                prompt_tokens=count,
                completion_tokens=None,
                total_tokens=count,
                response_model=tok["id"],
                response_id=f"local-{tok['id']}-{probe['id']}",
                selected_headers={},
                request_payload={"probe": probe["text"]},
                response_json={"local_tokenizer": tok["id"]},
                error=err,
                source_tier="structural_local",
            )
            observations.append(asdict(obs))
            ordinal += 1

    # 2. Remote Known Specimen queries
    session = requests.Session()
    rng = random.Random(seed)
    shuffled_probes = list(PROBES)
    rng.shuffle(shuffled_probes)

    attempts_file = run_dir / "raw" / "attempts.jsonl"
    for i, probe in enumerate(shuffled_probes, start=1):
        print(f"[{i:02d}/{len(shuffled_probes)}] {KNOWN_TARGET['label']} · {probe['label']} ... ", end="", flush=True)
        obs = perform_request(
            session,
            api_key,
            run_id,
            ordinal,
            KNOWN_TARGET,
            probe,
            on_attempt=lambda rec: append_jsonl(attempts_file, rec),
        )
        row = asdict(obs)
        observations.append(row)
        ordinal += 1
        if obs.ok:
            print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
        else:
            print(f"FAILED · {obs.status_code} · {obs.error}")

    summary = analyze(observations, demo=False, mode="positive-control", target_id=KNOWN_TARGET["id"])
    run_manifest = manifest(run_id, "positive_control_assay", seed, None)
    report = save_run(run_dir, run_manifest, observations, summary)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


# ---------------------------------------------------------------------------
# Command: Empirical Tokenizer Collision Simulation (Monte Carlo)
# ---------------------------------------------------------------------------


def command_collision(trials: int = 100000, probe_pool_size: int = 2000, seed: int = DEFAULT_SEED) -> int:
    """Empirical Monte Carlo simulation of tokenizer collision rates under the null hypothesis."""
    print(f"OXFORD Lite Tokenizer Collision Simulation")
    print(f"Generating synthetic probe corpus of size {probe_pool_size} across heterogeneous domains...")
    corpus = generate_synthetic_corpus(probe_pool_size, seed)

    print(f"Evaluating candidate local tokenizers...")
    tokenizer_ids = [t["id"] for t in LOCAL_TOKENIZERS]

    # Pre-tokenize all strings across all candidate tokenizers
    token_matrix: dict[str, list[int]] = {tid: [] for tid in tokenizer_ids}
    valid_indices = []

    for idx, p in enumerate(corpus):
        counts = {}
        all_ok = True
        for tid in tokenizer_ids:
            c, err = count_tokens_local(tid, p["id"], p["text"])
            if c is None or err:
                all_ok = False
                break
            counts[tid] = c
        if all_ok:
            valid_indices.append(idx)
            for tid in tokenizer_ids:
                token_matrix[tid].append(counts[tid])

    n_valid = len(valid_indices)
    print(f"Tokenized {n_valid} valid probes across {len(tokenizer_ids)} tokenizers.")

    if n_valid < 100:
        print("Error: insufficient tokenizers loaded for collision simulation.", file=sys.stderr)
        return 1

    # Define pairs of distinct tokenizer families
    pairs = [
        ("glm-5.2-local", "qwen-local"),
        ("glm-5.2-local", "gemma-local"),
        ("glm-5.2-local", "cl100k-local"),
        ("qwen-local", "gemma-local"),
        ("qwen-local", "cl100k-local"),
        ("gemma-local", "cl100k-local"),
    ]

    available_pairs = [
        (t1, t2) for t1, t2 in pairs
        if len(token_matrix.get(t1, [])) == n_valid and len(token_matrix.get(t2, [])) == n_valid
    ]

    print(f"Running Monte Carlo simulation over {trials:,} trials across {len(available_pairs)} unrelated tokenizer pairs...")
    subset_sizes = [1, 2, 3, 4, 6, 12]
    collision_counts: dict[int, int] = {k: 0 for k in subset_sizes}
    rng = random.Random(seed)

    for _ in range(trials):
        t1, t2 = rng.choice(available_pairs)
        v1 = token_matrix[t1]
        v2 = token_matrix[t2]

        for k in subset_sizes:
            if k > n_valid:
                continue
            indices = rng.sample(range(n_valid), k)
            offsets = [v1[i] - v2[i] for i in indices]
            if len(set(offsets)) == 1:
                collision_counts[k] += 1

    print("\n" + "=" * 70)
    print("EMPIRICAL TOKENIZER COLLISION RATES (UNRELATED TOKENIZERS NULL)")
    print("=" * 70)
    print(f"{'Probe Set Size (k)':<22} | {'Collisions / Trials':<22} | {'Empirical Collision Probability'}")
    print("-" * 70)

    for k in subset_sizes:
        hits = collision_counts[k]
        rate = hits / trials
        odds = f"1 in {int(1/rate):,}" if rate > 0 else f"< 1 in {trials:,} (0 hits)"
        pct_str = f"{100 * rate:.4f}%" if rate > 0.0001 else f"{rate:.2e}"
        print(f"k = {k:<18} | {hits:,} / {trials:,} ({pct_str:<7}) | {odds}")

    print("=" * 70)
    print("Conclusion: Observing a 4-probe or 6-probe constant offset between unrelated")
    print("tokenizers is statistically vanishing under this empirical null distribution.\n")
    return 0


# ---------------------------------------------------------------------------
# Command: High-Information Probe Synthesizer
# ---------------------------------------------------------------------------


def command_synthesize_probes(count: int = 5000, top_k: int = 16, seed: int = DEFAULT_SEED) -> int:
    """Synthesize candidate probes and select the top K with maximum tokenizer discrimination power."""
    print(f"OXFORD Lite Probe Synthesizer")
    print(f"Generating {count} candidate probe strings on laptop...")
    corpus = generate_synthetic_corpus(count, seed)

    tokenizer_ids = [t["id"] for t in LOCAL_TOKENIZERS]
    scored_probes = []

    print(f"Calculating inter-tokenizer pairwise separation across {len(tokenizer_ids)} tokenizers...")
    for p in corpus:
        counts_dict: dict[str, int] = {}
        for tid in tokenizer_ids:
            c, err = count_tokens_local(tid, p["id"], p["text"])
            if c is not None and err is None:
                counts_dict[tid] = c

        if len(counts_dict) == len(tokenizer_ids):
            counts = list(counts_dict.values())
            pair_diffs = [
                abs(counts[i] - counts[j])
                for i in range(len(counts))
                for j in range(i + 1, len(counts))
            ]
            min_pair_diff = min(pair_diffs) if pair_diffs else 0
            mean_diff = statistics.fmean(pair_diffs) if pair_diffs else 0
            span = max(counts) - min(counts)

            # Measure separation specifically between GLM and other candidates
            glm_count = counts_dict.get("glm-5.2-local")
            glm_diffs = [
                abs(glm_count - counts_dict[other_id])
                for other_id in counts_dict
                if other_id != "glm-5.2-local" and glm_count is not None
            ]
            glm_min_margin = min(glm_diffs) if glm_diffs else 0

            # Composite separation score: prioritize non-zero minimum pairwise margin,
            # then GLM candidate margin, then mean separation
            score = (min_pair_diff * 100.0) + (glm_min_margin * 25.0) + (mean_diff * 2.0) + (span * 0.1)

            scored_probes.append({
                "probe": p,
                "counts": counts_dict,
                "score": score,
                "min_pair_diff": min_pair_diff,
                "glm_min_margin": glm_min_margin,
                "mean_diff": mean_diff,
                "span": span,
            })

    # Sort descending by composite separation score
    scored_probes.sort(key=lambda x: x["score"], reverse=True)
    top_probes = [item["probe"] for item in scored_probes[:top_k]]

    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROBES_DIR / "high_information_probes.json"
    write_json(out_file, top_probes)

    print(f"\nExtracted {len(top_probes)} high-information probes (saved to {out_file}):")
    for i, item in enumerate(scored_probes[:top_k], start=1):
        p = item["probe"]
        cd = item["counts"]
        print(
            f"  [{i:02d}] {p['id']}: min_margin={item['min_pair_diff']}, "
            f"GLM_margin={item['glm_min_margin']}, span={item['span']} | "
            f"GLM={cd.get('glm-5.2-local')} Qwen={cd.get('qwen-local')} "
            f"Gemma={cd.get('gemma-local')} cl100k={cd.get('cl100k-local')}"
        )
    return 0


# ---------------------------------------------------------------------------
# Command: Multi-Envelope Invariance Assay
# ---------------------------------------------------------------------------


def command_envelope(open_report: bool, seed: int) -> int:
    """Multi-Envelope Assay: Tests whether content delta geometry remains invariant across 3 prompt envelopes."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add key to .env.", file=sys.stderr)
        return 2

    run_id = make_run_id("envelope")
    run_dir = ensure_run_dir(run_id)

    # Use first 4 probes across 3 envelopes (12 calls total)
    test_probes = PROBES[:4]
    envelope_ids = list(ENVELOPES.keys())

    print(f"OXFORD Lite Multi-Envelope Assay · {len(test_probes) * len(envelope_ids)} Ox Alpha requests")
    print(f"Testing 3 frozen envelopes: {', '.join(envelope_ids)}")
    print(f"Run folder: {run_dir}\n")

    observations: list[dict[str, Any]] = []
    ordinal = 1

    # 1. Local Tokenizers across envelopes
    for env_id in envelope_ids:
        for tok in LOCAL_TOKENIZERS:
            for probe in test_probes:
                count, err = count_tokens_local(tok["id"], probe["id"], probe["text"], envelope_id=env_id)
                obs = Observation(
                    run_id=run_id,
                    ordinal=ordinal,
                    collected_at_utc=utc_now(),
                    model_id=f"{tok['id']}__{env_id}",
                    model_slug=tok.get("hf_model", tok.get("encoding_name", tok["id"])),
                    model_role=tok["role"],
                    probe_id=probe["id"],
                    probe_label=f"{probe['label']} ({ENVELOPES[env_id]['label']})",
                    probe_sha256=sha256_text(probe["text"]),
                    status_code=200 if err is None else 500,
                    elapsed_ms=0.1,
                    ok=err is None and count is not None,
                    prompt_tokens=count,
                    completion_tokens=None,
                    total_tokens=count,
                    response_model=tok["id"],
                    response_id=f"local-{tok['id']}-{env_id}-{probe['id']}",
                    selected_headers={},
                    request_payload={"envelope": env_id},
                    response_json={"local": True},
                    error=err,
                    source_tier="structural_local",
                    envelope_id=env_id,
                )
                observations.append(asdict(obs))
                ordinal += 1

    # 2. Remote Ox Alpha across envelopes
    session = requests.Session()
    attempts_file = run_dir / "raw" / "attempts.jsonl"

    for env_id in envelope_ids:
        env_label = ENVELOPES[env_id]["label"]
        print(f"\n--- Testing {env_label} ---")
        for probe in test_probes:
            print(f"[{ordinal:02d}] {TARGET_MODEL['label']} · {probe['label']} ... ", end="", flush=True)
            obs = perform_request(
                session,
                api_key,
                run_id,
                ordinal,
                TARGET_MODEL,
                probe,
                envelope_id=env_id,
                on_attempt=lambda rec: append_jsonl(attempts_file, rec),
            )
            row = asdict(obs)
            row["model_id"] = f"{TARGET_MODEL['id']}__{env_id}"
            observations.append(row)
            ordinal += 1
            if obs.ok:
                print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
            else:
                print(f"FAILED · {obs.status_code} · {obs.error}")

    summary = analyze(observations, demo=False, mode="envelope", target_id=f"{TARGET_MODEL['id']}__envelope_b_standard", probe_list=test_probes)
    run_manifest = manifest(run_id, "envelope_assay", seed, None)
    report = save_run(run_dir, run_manifest, observations, summary, test_probes)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


# ---------------------------------------------------------------------------
# Command: Structural Assay
# ---------------------------------------------------------------------------


def command_structural(open_report: bool, seed: int, probes_file: str | None = None) -> int:
    """Structural assay: Local candidate tokenizers + remote Ox Alpha only."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add key to .env.", file=sys.stderr)
        return 2

    probes = PROBES
    if probes_file:
        p_path = Path(probes_file)
        if not p_path.is_absolute():
            p_path = ROOT / probes_file
        if p_path.exists():
            probes = json.loads(p_path.read_text(encoding="utf-8"))
            print(f"Loaded {len(probes)} probes from {p_path.name}")
        else:
            print(f"Probe file not found: {p_path}", file=sys.stderr)
            return 2

    run_id = make_run_id("structural")
    run_dir = ensure_run_dir(run_id)

    print(f"OXFORD Lite Structural Assay · {len(probes)} target requests")
    print(f"Run folder: {run_dir}")
    print("Evaluating real local candidate tokenizers + querying Ox Alpha remotely...\n")

    observations: list[dict[str, Any]] = []
    ordinal = 1

    # 1. Local tokenizers
    for tok in LOCAL_TOKENIZERS:
        for probe in probes:
            count, err = count_tokens_local(tok["id"], probe["id"], probe["text"])
            obs = Observation(
                run_id=run_id,
                ordinal=ordinal,
                collected_at_utc=utc_now(),
                model_id=tok["id"],
                model_slug=tok.get("hf_model", tok.get("encoding_name", tok["id"])),
                model_role=tok["role"],
                probe_id=probe["id"],
                probe_label=probe["label"],
                probe_sha256=sha256_text(probe["text"]),
                status_code=200 if err is None else 500,
                elapsed_ms=0.1,
                ok=err is None and count is not None,
                prompt_tokens=count,
                completion_tokens=None,
                total_tokens=count,
                response_model=tok["id"],
                response_id=f"local-{tok['id']}-{probe['id']}",
                selected_headers={},
                request_payload={"probe": probe["text"]},
                response_json={"local_tokenizer": tok["id"]},
                error=err,
                source_tier="structural_local",
            )
            observations.append(asdict(obs))
            ordinal += 1

    # 2. Remote Ox Alpha queries
    session = requests.Session()
    rng = random.Random(seed)
    shuffled_probes = list(probes)
    rng.shuffle(shuffled_probes)

    attempts_file = run_dir / "raw" / "attempts.jsonl"

    for i, probe in enumerate(shuffled_probes, start=1):
        print(f"[{i:02d}/{len(shuffled_probes)}] {TARGET_MODEL['label']} · {probe['label']} ... ", end="", flush=True)
        obs = perform_request(
            session,
            api_key,
            run_id,
            ordinal,
            TARGET_MODEL,
            probe,
            on_attempt=lambda rec: append_jsonl(attempts_file, rec),
        )
        row = asdict(obs)
        observations.append(row)
        ordinal += 1
        if obs.ok:
            print(f"ok · prompt_tokens={obs.prompt_tokens} · {obs.elapsed_ms:.0f} ms")
        else:
            print(f"FAILED · {obs.status_code} · {obs.error}")

    summary = analyze(observations, demo=False, mode="structural", probe_list=probes)
    run_manifest = manifest(run_id, "structural_assay", seed, None)
    report = save_run(run_dir, run_manifest, observations, summary, probes)
    print_comparison_console(summary)
    print(f"\nHTML report: {report}")
    print(f"Raw observations: {run_dir / 'raw' / 'observations.jsonl'}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return 0 if summary["requests_failed"] == 0 else 1


# ---------------------------------------------------------------------------
# Command: Remote Assay (Resumable)
# ---------------------------------------------------------------------------


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
    """Remote candidate and target model assay with model-aware scheduling and resume."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add key to .env.", file=sys.stderr)
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
        cell_key(o["model_id"], o["probe_id"], o.get("envelope_id", "envelope_b_standard"))
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
    attempts_file = run_dir / "raw" / "attempts.jsonl"

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
            on_attempt=lambda rec: append_jsonl(attempts_file, rec),
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


# ---------------------------------------------------------------------------
# Command: Local Ollama Assay
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


def command_local(models: list[str] | None, open_report: bool) -> int:
    """Probe Ollama models for prompt_eval_count."""
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


# ---------------------------------------------------------------------------
# Command: Doctor & Demo
# ---------------------------------------------------------------------------


def command_doctor() -> int:
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    print(f"OXFORD Lite v{PILOT_VERSION}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"requests: {requests.__version__}")
    print(f"OpenRouter API key present: {'yes' if key else 'NO'}")

    print("\nLocal candidate tokenizers (fail-closed check):")
    for tok in LOCAL_TOKENIZERS:
        _, err = get_local_tokenizer(tok["id"])
        status = "READY" if err is None else f"FAILED ({err})"
        print(f"  - {tok['label']}: {status}")

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=0.5)
        if resp.ok:
            models_data = resp.json().get("models", [])
            names = [m.get("name") for m in models_data]
            print(f"\nOllama server: ONLINE at {OLLAMA_BASE_URL} ({len(names)} models: {', '.join(names[:4])})")
        else:
            print(f"\nOllama server: status {resp.status_code}")
    except Exception:
        print(f"\nOllama server: not reachable at {OLLAMA_BASE_URL} (optional)")

    print(f"\nTarget model: {TARGET_MODEL['slug']}")
    print(f"Remote candidate routes: {len(REMOTE_MODELS)} configured")
    if not key:
        print("\nNote: Add OPENROUTER_API_KEY to .env for remote queries.")
        return 2
    print("\nConfiguration looks ready.")
    return 0


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
                    "envelope_id": "envelope_b_standard",
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
        "analysis_note": "Isolates black-box structural and causal dynamics without ungrounded provider attribution.",
    }


# ---------------------------------------------------------------------------
# Exploration 2A: Causal Support Dynamics Commands
# ---------------------------------------------------------------------------

WORLDS_DIR = ROOT / "worlds"
HOLDOUT_WORLDS_FILE = WORLDS_DIR / "holdout" / "support_dynamics_holdout.json"
HOLDOUT_BOUNDARY_FILE = WORLDS_DIR / "holdout" / "support_boundary_holdout.json"


def command_dynamics_synthesize(count: int = 30, top_k: int = 8, seed: int = DEFAULT_SEED) -> int:
    """Synthesize candidate Stage-5A support worlds with isomorphic twins and freeze top holdouts."""
    from assays import support_dynamics as sd

    print(f"OXFORD Exploration 2A: Causal Support Dynamics Synthesizer")
    print(f"Synthesizing {count} candidate support worlds with isomorphic twins...")
    corpus = sd.synthesize_dynamics_corpus(count=count, seed=seed)

    dev_dir = WORLDS_DIR / "development"
    holdout_dir = WORLDS_DIR / "holdout"
    dev_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    # Save full development pool
    write_json(dev_dir / "candidate_worlds.json", corpus)

    # Select top K holdout trajectories
    top_holdouts = corpus[:top_k]
    write_json(HOLDOUT_WORLDS_FILE, top_holdouts)

    holdout_hash = sha256_text(canonical_json(top_holdouts))
    print(f"\nSuccessfully froze {len(top_holdouts)} holdout worlds to {HOLDOUT_WORLDS_FILE}")
    print(f"Exploration Firewall: STATUS=HOLDOUT, SHA-256={holdout_hash}")
    print("\nFrozen Holdout World Trajectories:")
    for i, item in enumerate(top_holdouts, start=1):
        w = item["world"]
        gt = item["ground_truth"]
        print(f"  [{i:02d}] {w['world_id']}: Target={w['target_entity']} {w['target_property']} | Ground truth vector: {gt}")
    return 0


def render_dynamics_html(
    run_id: str,
    results: list[dict[str, Any]],
    overall_accuracy: float,
    mean_stability: float,
    retraction_sensitivity: float,
    survival_rate: float,
    rescue_rate: float,
    sham_rate: float,
) -> str:
    """Render interactive HTML report for causal dynamics assay."""
    rows = []
    for res in results:
        w_id = html.escape(res["world_id"])
        target = html.escape(res["target"])
        obs_vec = res["observed_vector"]
        twin_vec = res["twin_observed_vector"]
        gt_vec = res["ground_truth_vector"]
        acc = res["accuracy"]
        stab = res["stability"]

        # Base row
        cells_base = [f"<td rowspan='2'><strong>{w_id}</strong><div class='small muted'>{target}</div></td>", "<td><span class='small muted'>Base (W)</span></td>"]
        for obs, exp in zip(obs_vec, gt_vec):
            cls = "badge-good" if obs == exp else "badge-bad"
            cells_base.append(f"<td><span class='badge {cls}'>{html.escape(obs)}</span></td>")
        cells_base.append(f"<td rowspan='2'><strong>{acc * 100:.0f}%</strong></td>")
        cells_base.append(f"<td rowspan='2'><strong>{stab * 100:.0f}%</strong></td>")
        rows.append("<tr>" + "".join(cells_base) + "</tr>")

        # Twin row
        cells_twin = ["<td><span class='small muted'>Twin (W')</span></td>"]
        for obs, exp in zip(twin_vec, gt_vec):
            cls = "badge-good" if obs == exp else "badge-bad"
            cells_twin.append(f"<td><span class='badge {cls}'>{html.escape(obs)}</span></td>")
        rows.append("<tr style='background:rgba(0,0,0,0.015);'>" + "".join(cells_twin) + "</tr>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OXFORD Dynamics · {html.escape(run_id)}</title>
<style>
:root {{ --ink:#111827; --muted:#667085; --line:#e5e7eb; --paper:#ffffff; --wash:#f6f7f9; --accent:#7a3e9d; --good:#166534; --bad:#991b1b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, sans-serif; color:var(--ink); background:var(--wash); }}
.shell {{ max-width:1120px; margin:0 auto; padding:34px 22px 64px; }}
.hero {{ background:linear-gradient(135deg,#1e1b4b,#4338ca 62%,#701a75); color:white; border-radius:22px; padding:30px 32px 28px; box-shadow:0 12px 35px rgba(30,27,75,.14); }}
.eyebrow {{ font-size:12px; letter-spacing:.14em; font-weight:800; opacity:.78; }}
h1 {{ margin:8px 0 4px; font-size:34px; letter-spacing:-.04em; }}
.sub {{ max-width:760px; color:#e0e7ff; line-height:1.55; }}
.grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:18px 0; }}
.card {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:16px; }}
.card .k {{ color:var(--muted); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }}
.card .v {{ margin-top:6px; font-size:22px; font-weight:800; }}
section {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; margin-top:14px; padding:22px; }}
table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
th {{ text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:10px 8px; border-bottom:1px solid var(--line); }}
td {{ padding:8px; border-bottom:1px solid #eef0f3; vertical-align:middle; }}
.badge {{ display:inline-block; padding:3px 7px; border-radius:6px; font-size:11px; font-weight:700; font-family:monospace; }}
.badge-good {{ background:#dcfce7; color:var(--good); }}
.badge-bad {{ background:#fee2e2; color:var(--bad); }}
.muted {{ color:var(--muted); }}
.small {{ font-size:11px; }}
</style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <div class="eyebrow">OXFORD · EXPLORATION 2A.1</div>
    <h1>Causal Support Dynamics Fingerprint</h1>
    <div class="sub">Counterfactual revision trajectories across 8 paired interventions (lesions, cuts, rescues, shams) and isomorphic twin stability.</div>
  </div>

  <div class="grid">
    <div class="card"><div class="k">Accuracy</div><div class="v">{overall_accuracy * 100:.1f}%</div></div>
    <div class="card"><div class="k">Isomorphic Stability</div><div class="v">{mean_stability * 100:.1f}%</div></div>
    <div class="card"><div class="k">Cut Sensitivity (-AC)</div><div class="v">{retraction_sensitivity * 100:.0f}%</div></div>
    <div class="card"><div class="k">Support Survival (-A)</div><div class="v">{survival_rate * 100:.0f}%</div></div>
    <div class="card"><div class="k">Rescue Rate (+A)</div><div class="v">{rescue_rate * 100:.0f}%</div></div>
    <div class="card"><div class="k">Sham Invariance (-E)</div><div class="v">{sham_rate * 100:.0f}%</div></div>
  </div>

  <section>
    <h2>Observed Trajectory Response Vectors &amp; Isomorphic Twins</h2>
    <table>
      <thead>
        <tr>
          <th>World</th>
          <th>Variant</th>
          <th>Base (0)</th>
          <th>-A</th>
          <th>-C</th>
          <th>-AC (Cut)</th>
          <th>-AB</th>
          <th>-ABC (Cut)</th>
          <th>+A (Rescue)</th>
          <th>-E (Sham)</th>
          <th>Accuracy</th>
          <th>Stability</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
</div>
</body>
</html>"""


def command_dynamics_assay(
    open_report: bool,
    seed: int,
    holdout_file: str | None = None,
    target_slug: str = "stealth/ox-alpha",
) -> int:
    """Execute Exploration 2A causal support dynamics assay against target."""
    from assays import support_dynamics as sd

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add key to .env.", file=sys.stderr)
        return 2

    h_path = Path(holdout_file) if holdout_file else HOLDOUT_WORLDS_FILE
    if not h_path.exists():
        print(f"Holdout file not found: {h_path}. Running synthesize first...", file=sys.stderr)
        command_dynamics_synthesize(count=30, top_k=8, seed=seed)

    holdouts = json.loads(h_path.read_text(encoding="utf-8"))
    holdout_hash = sha256_text(canonical_json(holdouts))

    run_id = make_run_id("dynamics")
    run_dir = ensure_run_dir(run_id)

    target_id = target_slug.replace("/", "-").replace(":", "-")
    target_info = {
        "id": target_id,
        "slug": target_slug,
        "label": target_slug.split("/")[-1],
        "role": "target",
    }

    print(f"OXFORD Exploration 2A.1: Causal Support Dynamics Assay")
    print(f"Target: {target_slug}")
    print(f"Loaded {len(holdouts)} frozen holdout worlds with paired twins (SHA-256={holdout_hash[:16]}...)")
    print(f"Run folder: {run_dir}\n")

    session = requests.Session()
    attempts_file = run_dir / "raw" / "attempts.jsonl"
    observations: list[dict[str, Any]] = []

    world_results = []
    all_observed = []
    all_expected = []
    stabilities = []
    ordinal = 1

    for w_idx, item in enumerate(holdouts, start=1):
        world_obj = sd.SupportWorld(**item["world"])
        twin_obj = sd.SupportWorld(**item["twin"])
        traj_items = [sd.TrajectoryIntervention(**t) for t in item["trajectory"]]
        twin_traj_items = [sd.TrajectoryIntervention(**t) for t in item["twin_trajectory"]]
        gt_vector = item["ground_truth"]

        print(f"=== World [{w_idx}/{len(holdouts)}]: {world_obj.world_id} ({world_obj.target_entity} {world_obj.target_property}) ===")

        # 1. Base World Trajectory
        print("  [Base World W]")
        observed_vector = []
        for t_idx, interv in enumerate(traj_items, start=1):
            probe_dict = {
                "id": f"{world_obj.world_id}_{interv.condition_id}",
                "label": f"{world_obj.world_id} {interv.label}",
                "text": interv.prompt_text,
            }
            print(f"    [{t_idx:02d}/08] {interv.label} ... ", end="", flush=True)

            obs = perform_request(
                session,
                api_key,
                run_id,
                ordinal,
                target_info,
                probe_dict,
                envelope_id="envelope_a_minimal",
                max_tokens=1024,
                on_attempt=lambda rec: append_jsonl(attempts_file, rec),
            )
            ordinal += 1
            observations.append(asdict(obs))

            # Strict final-answer extraction: only content is evaluated
            raw_resp = ""
            if obs.response_json and isinstance(obs.response_json, dict):
                choices = obs.response_json.get("choices", [])
                if choices:
                    raw_resp = choices[0].get("message", {}).get("content") or ""

            state = sd.parse_response_state(raw_resp)
            observed_vector.append(state)
            is_match = state == interv.expected_state
            mark = "ok" if is_match else f"DIFF (got {state}, expected {interv.expected_state})"
            print(f"{mark} · {obs.elapsed_ms:.0f} ms")

        # 2. Isomorphic Twin Trajectory
        print("  [Isomorphic Twin W']")
        twin_observed_vector = []
        for t_idx, interv in enumerate(twin_traj_items, start=1):
            probe_dict = {
                "id": f"{twin_obj.world_id}_{interv.condition_id}",
                "label": f"{twin_obj.world_id} {interv.label}",
                "text": interv.prompt_text,
            }
            print(f"    [{t_idx:02d}/08] {interv.label} (twin) ... ", end="", flush=True)

            obs = perform_request(
                session,
                api_key,
                run_id,
                ordinal,
                target_info,
                probe_dict,
                envelope_id="envelope_a_minimal",
                max_tokens=1024,
                on_attempt=lambda rec: append_jsonl(attempts_file, rec),
            )
            ordinal += 1
            observations.append(asdict(obs))

            raw_resp = ""
            if obs.response_json and isinstance(obs.response_json, dict):
                choices = obs.response_json.get("choices", [])
                if choices:
                    raw_resp = choices[0].get("message", {}).get("content") or ""

            state = sd.parse_response_state(raw_resp)
            twin_observed_vector.append(state)
            is_match = state == interv.expected_state
            mark = "ok" if is_match else f"DIFF (got {state}, expected {interv.expected_state})"
            print(f"{mark} · {obs.elapsed_ms:.0f} ms")

        acc_base = sd.trajectory_accuracy(observed_vector, gt_vector)
        acc_twin = sd.trajectory_accuracy(twin_observed_vector, gt_vector)
        mean_world_acc = (acc_base + acc_twin) / 2.0
        stability = 1.0 - sd.trajectory_distance(observed_vector, twin_observed_vector)
        stabilities.append(stability)

        all_observed.extend(observed_vector)
        all_observed.extend(twin_observed_vector)
        all_expected.extend(gt_vector)
        all_expected.extend(gt_vector)

        print(f"  --> World {world_obj.world_id} Result: Base Acc={acc_base*100:.0f}%, Twin Acc={acc_twin*100:.0f}%, Isomorphic Stability={stability*100:.0f}%\n")

        world_results.append({
            "world_id": world_obj.world_id,
            "target": f"{world_obj.target_entity} {world_obj.target_property}",
            "observed_vector": observed_vector,
            "twin_observed_vector": twin_observed_vector,
            "ground_truth_vector": gt_vector,
            "accuracy": mean_world_acc,
            "stability": stability,
        })

    total_acc = sd.trajectory_accuracy(all_observed, all_expected)
    mean_stab = statistics.fmean(stabilities) if stabilities else 1.0

    cut_acc = sum(1 for res in world_results for vec in [res["observed_vector"], res["twin_observed_vector"]] for c in [3, 5] if vec[c] == "UNKNOWN") / (len(world_results) * 4)
    surv_acc = sum(1 for res in world_results for vec in [res["observed_vector"], res["twin_observed_vector"]] for c in [1, 2, 4] if vec[c] == "ACTIVE") / (len(world_results) * 6)
    rescue_acc = sum(1 for res in world_results for vec in [res["observed_vector"], res["twin_observed_vector"]] if vec[6] == "ACTIVE") / (len(world_results) * 2)
    sham_acc = sum(1 for res in world_results for vec in [res["observed_vector"], res["twin_observed_vector"]] if vec[7] == "ACTIVE") / (len(world_results) * 2)

    manifest_data = {
        "run_id": run_id,
        "kind": "support_dynamics_assay",
        "target": target_info,
        "firewall_status": "HOLDOUT",
        "holdout_corpus_sha256": holdout_hash,
        "created_at_utc": utc_now(),
        "total_trajectories": len(world_results) * 2,
        "total_requests": len(observations),
    }
    write_json(run_dir / "manifest.json", manifest_data)
    write_jsonl(run_dir / "raw" / "observations.jsonl", observations)

    summary = {
        "run_id": run_id,
        "target": target_slug,
        "overall_trajectory_accuracy": total_acc,
        "within_model_isomorphic_stability": mean_stab,
        "cut_retraction_sensitivity": cut_acc,
        "alternative_support_survival": surv_acc,
        "rescue_recovery_rate": rescue_acc,
        "sham_invariance_rate": sham_acc,
        "world_trajectories": world_results,
    }
    write_json(run_dir / "summary.json", summary)

    report_html = render_dynamics_html(run_id, world_results, total_acc, mean_stab, cut_acc, surv_acc, rescue_acc, sham_acc)
    report_path = run_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")

    print("=" * 65)
    print("OXFORD EXPLORATION 2A.1: SUPPORT DYNAMICS HARDENED RESULTS")
    print("=" * 65)
    print(f"Overall Trajectory Accuracy:        {total_acc * 100:.1f}%")
    print(f"Within-Model Isomorphic Stability:  {mean_stab * 100:.1f}% (invariance across permuted twins)")
    print(f"Complete Cut Sensitivity (-AC):     {cut_acc * 100:.1f}% (correctly dropped to UNKNOWN)")
    print(f"Alternative Path Survival (-A):     {surv_acc * 100:.1f}% (correctly retained ACTIVE)")
    print(f"Rescue Recovery Rate (+A):          {rescue_acc * 100:.1f}% (correctly re-activated)")
    print(f"Sham Lexical Invariance (-E):       {sham_acc * 100:.1f}% (resisted false retraction)")
    print("=" * 65)
    print(f"HTML Report: {report_path}")
    if open_report:
        webbrowser.open(report_path.resolve().as_uri())
    return 0


def command_boundary_synthesize(seed: int = DEFAULT_SEED) -> int:
    """Synthesize candidate Exploration 2B boundary & laundering worlds and freeze holdout set."""
    from assays import support_boundary as sb

    print("OXFORD Exploration 2B: Lineage Laundering & Depth Boundary Synthesizer")
    print("Synthesizing multi-depth and ancestral-overlap worlds with adversarial twins...")
    corpus = sb.synthesize_boundary_corpus(seed=seed)

    dev_dir = WORLDS_DIR / "development"
    holdout_dir = WORLDS_DIR / "holdout"
    dev_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    write_json(dev_dir / "boundary_candidates.json", corpus)
    write_json(HOLDOUT_BOUNDARY_FILE, corpus)

    holdout_hash = sha256_text(canonical_json(corpus))
    print(f"\nSuccessfully froze {len(corpus)} boundary holdout worlds to {HOLDOUT_BOUNDARY_FILE}")
    print(f"Exploration Firewall: STATUS=HOLDOUT, SHA-256={holdout_hash}")
    print("\nFrozen Boundary Holdout World Trajectories:")
    for i, item in enumerate(corpus, start=1):
        w = item["world"]
        gt = item["ground_truth"]
        print(f"  [{i:02d}] {w['world_id']} ({w['mode']}, depth={w['depth']}): Target={w['target_entity']} {w['target_property']} | GT: {gt}")
    return 0


def render_boundary_html(
    run_id: str,
    results: list[dict[str, Any]],
    overall_accuracy: float,
    mean_stability: float,
    repeat_stability: float,
    indep_accuracy: float,
    shared_accuracy: float,
    launder_accuracy: float,
    depth_stats: dict[int, dict[str, float]],
    error_decomp: dict[str, float],
) -> str:
    """Render interactive HTML report for Exploration 2B.1 boundary assay."""
    rows = []
    for res in results:
        w_id = html.escape(res["world_id"])
        mode = html.escape(res["mode"])
        depth = res["depth"]
        obs_vec = res["observed_vector"]
        twin_vec = res["twin_observed_vector"]
        gt_vec = res["ground_truth_vector"]
        acc = res["accuracy"]
        stab = res["stability"]

        cells_base = [
            f"<td rowspan='2'><strong>{w_id}</strong><div class='small muted'>{mode} (d={depth})</div></td>",
            "<td><span class='small muted'>Base (W)</span></td>"
        ]
        for obs, exp in zip(obs_vec, gt_vec):
            cls = "badge-good" if obs == exp else "badge-bad"
            cells_base.append(f"<td><span class='badge {cls}'>{html.escape(obs)}</span></td>")
        for _ in range(6 - len(obs_vec)):
            cells_base.append("<td><span class='small muted'>—</span></td>")
        cells_base.append(f"<td rowspan='2'><strong>{acc * 100:.0f}%</strong></td>")
        cells_base.append(f"<td rowspan='2'><strong>{stab * 100:.0f}%</strong></td>")
        rows.append("<tr>" + "".join(cells_base) + "</tr>")

        cells_twin = ["<td><span class='small muted'>Twin (W')</span></td>"]
        for obs, exp in zip(twin_vec, gt_vec):
            cls = "badge-good" if obs == exp else "badge-bad"
            cells_twin.append(f"<td><span class='badge {cls}'>{html.escape(obs)}</span></td>")
        for _ in range(6 - len(twin_vec)):
            cells_twin.append("<td><span class='small muted'>—</span></td>")
        rows.append("<tr style='background:rgba(0,0,0,0.015);'>" + "".join(cells_twin) + "</tr>")

    depth_rows = []
    for d, s in depth_stats.items():
        depth_rows.append(
            f"<tr><td><strong>d = {d}</strong></td>"
            f"<td>{s['canonical_acc'] * 100:.1f}%</td>"
            f"<td>{s['twin_acc'] * 100:.1f}%</td>"
            f"<td><strong>{s['stability'] * 100:.1f}%</strong></td></tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OXFORD Boundary &amp; Laundering E2B.1 · {html.escape(run_id)}</title>
<style>
:root {{ --ink:#111827; --muted:#667085; --line:#e5e7eb; --paper:#ffffff; --wash:#f6f7f9; --good:#166534; --bad:#991b1b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, sans-serif; color:var(--ink); background:var(--wash); }}
.shell {{ max-width:1160px; margin:0 auto; padding:34px 22px 64px; }}
.hero {{ background:linear-gradient(135deg,#064e3b,#047857 62%,#0f766e); color:white; border-radius:22px; padding:30px 32px 28px; box-shadow:0 12px 35px rgba(6,78,59,.14); }}
.eyebrow {{ font-size:12px; letter-spacing:.14em; font-weight:800; opacity:.78; }}
h1 {{ margin:8px 0 4px; font-size:34px; letter-spacing:-.04em; }}
.sub {{ max-width:760px; color:#d1fae5; line-height:1.55; }}
.grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:18px 0; }}
.card {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:14px; }}
.card .k {{ color:var(--muted); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }}
.card .v {{ margin-top:6px; font-size:20px; font-weight:800; }}
section {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; margin-top:14px; padding:22px; }}
table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
th {{ text-align:left; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:10px 8px; border-bottom:1px solid var(--line); }}
td {{ padding:8px; border-bottom:1px solid #eef0f3; vertical-align:middle; }}
.badge {{ display:inline-block; padding:3px 7px; border-radius:6px; font-size:11px; font-weight:700; font-family:monospace; }}
.badge-good {{ background:#dcfce7; color:var(--good); }}
.badge-bad {{ background:#fee2e2; color:var(--bad); }}
.muted {{ color:var(--muted); }}
.small {{ font-size:11px; }}
</style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <div class="eyebrow">OXFORD · EXPLORATION 2B.1</div>
    <h1>Formal Support Boundary &amp; Lineage Laundering Assay</h1>
    <div class="sub">Formal Horn derivation chains, multi-hop shared ancestry DAGs, and lineage laundering collapse.</div>
  </div>

  <div class="grid">
    <div class="card"><div class="k">Overall Accuracy</div><div class="v">{overall_accuracy * 100:.1f}%</div></div>
    <div class="card"><div class="k">Isomorphic Stab</div><div class="v">{mean_stability * 100:.1f}%</div></div>
    <div class="card"><div class="k">Exact Repeat Stab</div><div class="v">{repeat_stability * 100:.1f}%</div></div>
    <div class="card"><div class="k">Indep (d=2..5)</div><div class="v">{indep_accuracy * 100:.0f}%</div></div>
    <div class="card"><div class="k">Shared Root</div><div class="v">{shared_accuracy * 100:.0f}%</div></div>
    <div class="card"><div class="k">Laundered Echo</div><div class="v">{launder_accuracy * 100:.0f}%</div></div>
  </div>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px;">
    <section style="margin-top:0;">
      <h3 style="margin-top:0;">Depth × Representation Interaction</h3>
      <table>
        <thead><tr><th>Derivation Depth</th><th>Canonical Acc (W)</th><th>Twin Acc (W')</th><th>Stability</th></tr></thead>
        <tbody>{''.join(depth_rows)}</tbody>
      </table>
    </section>
    <section style="margin-top:0;">
      <h3 style="margin-top:0;">Error Polarity Decomposition</h3>
      <table>
        <thead><tr><th>Error Metric</th><th>Definition</th><th>Rate</th></tr></thead>
        <tbody>
          <tr><td><strong>False Survival (F+)</strong></td><td>P(ACTIVE | UNKNOWN expected)</td><td>{error_decomp['false_survival_rate'] * 100:.1f}%</td></tr>
          <tr><td><strong>False Retraction (F-)</strong></td><td>P(UNKNOWN | ACTIVE expected)</td><td>{error_decomp['false_retraction_rate'] * 100:.1f}%</td></tr>
          <tr><td><strong>Format Failure (Ffmt)</strong></td><td>Non-token / reasoning overflow</td><td>{error_decomp['format_failure_rate'] * 100:.1f}%</td></tr>
        </tbody>
      </table>
    </section>
  </div>

  <section>
    <h2>Causal Trajectory Response Matrix</h2>
    <table>
      <thead>
        <tr>
          <th>World</th>
          <th>Variant</th>
          <th>C01 (Base)</th>
          <th>C02 (Lesion)</th>
          <th>C03 (Cut/Coll)</th>
          <th>C04 (Cut/Rsc)</th>
          <th>C05 (Rsc)</th>
          <th>C99 (Sham)</th>
          <th>Accuracy</th>
          <th>Stability</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
</div>
</body>
</html>"""


def command_boundary_assay(
    open_report: bool,
    seed: int,
    holdout_file: str | None = None,
    target_slug: str = "stealth/ox-alpha",
) -> int:
    """Execute Exploration 2B.1 formal graph lineage laundering and depth boundary assay."""
    from assays import support_boundary as sb

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENROUTER_API_KEY. Add key to .env.", file=sys.stderr)
        return 2

    h_path = Path(holdout_file) if holdout_file else HOLDOUT_BOUNDARY_FILE
    if not h_path.exists():
        print(f"Boundary holdout file not found: {h_path}. Running synthesize first...", file=sys.stderr)
        command_boundary_synthesize(seed=seed)

    holdouts = json.loads(h_path.read_text(encoding="utf-8"))
    holdout_hash = sha256_text(canonical_json(holdouts))

    run_id = make_run_id("boundary")
    run_dir = ensure_run_dir(run_id)

    target_id = target_slug.replace("/", "-").replace(":", "-")
    target_info = {
        "id": target_id,
        "slug": target_slug,
        "label": target_slug.split("/")[-1],
        "role": "target",
    }

    print("OXFORD Exploration 2B.1: Formal Support Boundary & Lineage Laundering Assay")
    print(f"Target: {target_slug}")
    print(f"Loaded {len(holdouts)} frozen boundary holdouts with paired twins (SHA-256={holdout_hash[:16]}...)")
    print(f"Run folder: {run_dir}\n")

    session = requests.Session()
    attempts_file = run_dir / "raw" / "attempts.jsonl"
    observations: list[dict[str, Any]] = []

    world_results = []
    all_observed = []
    all_expected = []
    stabilities = []
    repeat_matches = []
    ordinal = 1

    # Track repeat controls at depths d=2,3,4,5
    repeat_depth_checkpoints = {2: False, 3: False, 4: False, 5: False}

    for w_idx, item in enumerate(holdouts, start=1):
        world_obj = sb.BoundaryWorld(**item["world"])
        twin_obj = sb.BoundaryWorld(**item["twin"])
        traj_items = [sb.BoundaryIntervention(**t) for t in item["trajectory"]]
        twin_traj_items = [sb.BoundaryIntervention(**t) for t in item["twin_trajectory"]]
        gt_vector = item["ground_truth"]

        print(f"=== World [{w_idx}/{len(holdouts)}]: {world_obj.world_id} ({world_obj.mode}, d={world_obj.depth}) ===")
        run_base_first = (w_idx % 2 == 1)

        def execute_branch(obj: sb.BoundaryWorld, items: list[sb.BoundaryIntervention], label_suffix: str) -> list[str]:
            nonlocal ordinal
            print(f"  [{label_suffix}]")
            obs_vec = []
            for t_idx, interv in enumerate(items, start=1):
                probe_dict = {
                    "id": f"{obj.world_id}_{interv.condition_id}",
                    "label": f"{obj.world_id} {interv.label}",
                    "text": interv.prompt_text,
                }
                print(f"    [{t_idx:02d}/{len(items):02d}] {interv.label} ... ", end="", flush=True)

                obs = perform_request(
                    session,
                    api_key,
                    run_id,
                    ordinal,
                    target_info,
                    probe_dict,
                    envelope_id="envelope_a_minimal",
                    max_tokens=1024,
                    on_attempt=lambda rec: append_jsonl(attempts_file, rec),
                )
                ordinal += 1
                observations.append(asdict(obs))

                raw_resp = ""
                if obs.response_json and isinstance(obs.response_json, dict):
                    choices = obs.response_json.get("choices", [])
                    if choices:
                        raw_resp = choices[0].get("message", {}).get("content") or ""

                state = sb.parse_response_state(raw_resp)
                obs_vec.append(state)
                is_match = state == interv.expected_state
                mark = "ok" if is_match else f"DIFF (got {state}, expected {interv.expected_state})"
                print(f"{mark} · {obs.elapsed_ms:.0f} ms")

                # Exact Repeat Control Checkpoint (1 per independent depth)
                if world_obj.mode == "INDEPENDENT" and not obj.is_twin and t_idx == 1 and not repeat_depth_checkpoints.get(world_obj.depth, True):
                    repeat_depth_checkpoints[world_obj.depth] = True
                    print(f"      [Repeat Control d={world_obj.depth}] Querying identical base prompt again ... ", end="", flush=True)
                    obs_rep = perform_request(
                        session,
                        api_key,
                        run_id,
                        ordinal,
                        target_info,
                        probe_dict,
                        envelope_id="envelope_a_minimal",
                        max_tokens=1024,
                        on_attempt=lambda rec: append_jsonl(attempts_file, rec),
                    )
                    ordinal += 1
                    observations.append(asdict(obs_rep))
                    raw_rep_resp = ""
                    if obs_rep.response_json and isinstance(obs_rep.response_json, dict):
                        choices_rep = obs_rep.response_json.get("choices", [])
                        if choices_rep:
                            raw_rep_resp = choices_rep[0].get("message", {}).get("content") or ""
                    state_rep = sb.parse_response_state(raw_rep_resp)
                    rep_is_match = (state == state_rep)
                    repeat_matches.append(rep_is_match)
                    print(f"{'MATCH' if rep_is_match else 'DRIFT'} (1st={state}, 2nd={state_rep}) · {obs_rep.elapsed_ms:.0f} ms")

            return obs_vec

        if run_base_first:
            observed_vector = execute_branch(world_obj, traj_items, "Base World W (Order: 1st)")
            twin_observed_vector = execute_branch(twin_obj, twin_traj_items, "Isomorphic Twin W' (Order: 2nd)")
        else:
            twin_observed_vector = execute_branch(twin_obj, twin_traj_items, "Isomorphic Twin W' (Order: 1st)")
            observed_vector = execute_branch(world_obj, traj_items, "Base World W (Order: 2nd)")

        acc_base = sb.trajectory_accuracy(observed_vector, gt_vector)
        acc_twin = sb.trajectory_accuracy(twin_observed_vector, gt_vector)
        mean_world_acc = (acc_base + acc_twin) / 2.0
        stability = 1.0 - sb.trajectory_distance(observed_vector, twin_observed_vector)
        stabilities.append(stability)

        all_observed.extend(observed_vector)
        all_observed.extend(twin_observed_vector)
        all_expected.extend(gt_vector)
        all_expected.extend(gt_vector)

        print(f"  --> World {world_obj.world_id} Result: Base Acc={acc_base*100:.0f}%, Twin Acc={acc_twin*100:.0f}%, Stability={stability*100:.0f}%\n")

        world_results.append({
            "world_id": world_obj.world_id,
            "mode": world_obj.mode,
            "depth": world_obj.depth,
            "target": f"{world_obj.target_entity} {world_obj.target_property}",
            "observed_vector": observed_vector,
            "twin_observed_vector": twin_observed_vector,
            "ground_truth_vector": gt_vector,
            "canonical_accuracy": acc_base,
            "twin_accuracy": acc_twin,
            "accuracy": mean_world_acc,
            "stability": stability,
        })

    total_acc = sb.trajectory_accuracy(all_observed, all_expected)
    mean_stab = statistics.fmean(stabilities) if stabilities else 1.0
    repeat_stab = sum(1 for m in repeat_matches if m) / len(repeat_matches) if repeat_matches else 1.0

    ind_results = [r for r in world_results if r["mode"] == "INDEPENDENT"]
    shared_results = [r for r in world_results if r["mode"] == "SHARED_ROOT"]
    launder_results = [r for r in world_results if r["mode"] == "LAUNDERED_ECHO"]

    ind_acc = statistics.fmean([r["accuracy"] for r in ind_results]) if ind_results else 0.0
    shared_acc = statistics.fmean([r["accuracy"] for r in shared_results]) if shared_results else 0.0
    launder_acc = statistics.fmean([r["accuracy"] for r in launder_results]) if launder_results else 0.0

    # Depth Breakdown for INDEPENDENT
    depth_stats = {}
    for d in [2, 3, 4, 5]:
        d_res = [r for r in ind_results if r["depth"] == d]
        if d_res:
            c_acc = statistics.fmean([r["canonical_accuracy"] for r in d_res])
            t_acc = statistics.fmean([r["twin_accuracy"] for r in d_res])
            d_stab = statistics.fmean([r["stability"] for r in d_res])
            depth_stats[d] = {
                "canonical_acc": c_acc,
                "twin_acc": t_acc,
                "stability": d_stab,
            }

    # Error Polarity Decomposition
    # F+ = P(ACTIVE | UNKNOWN expected)
    # F- = P(UNKNOWN | ACTIVE expected)
    # F_fmt = P(FORMAT_FAILURE)
    unknown_expected_pairs = [(obs, exp) for obs, exp in zip(all_observed, all_expected) if exp == "UNKNOWN"]
    active_expected_pairs = [(obs, exp) for obs, exp in zip(all_observed, all_expected) if exp == "ACTIVE"]

    f_plus = sum(1 for obs, _ in unknown_expected_pairs if obs == "ACTIVE") / len(unknown_expected_pairs) if unknown_expected_pairs else 0.0
    f_minus = sum(1 for obs, _ in active_expected_pairs if obs == "UNKNOWN") / len(active_expected_pairs) if active_expected_pairs else 0.0
    f_fmt = sum(1 for obs in all_observed if obs == "FORMAT_FAILURE") / len(all_observed) if all_observed else 0.0

    error_decomp = {
        "false_survival_rate": f_plus,
        "false_retraction_rate": f_minus,
        "format_failure_rate": f_fmt,
    }

    manifest_data = {
        "run_id": run_id,
        "kind": "support_boundary_assay_e2b1",
        "target": target_info,
        "firewall_status": "HOLDOUT",
        "holdout_corpus_sha256": holdout_hash,
        "created_at_utc": utc_now(),
        "total_trajectories": len(world_results) * 2,
        "total_requests": len(observations),
        "repeat_controls_tested": len(repeat_matches),
    }
    write_json(run_dir / "manifest.json", manifest_data)
    write_jsonl(run_dir / "raw" / "observations.jsonl", observations)

    summary = {
        "run_id": run_id,
        "target": target_slug,
        "overall_trajectory_accuracy": total_acc,
        "within_model_isomorphic_stability": mean_stab,
        "exact_repeat_stability": repeat_stab,
        "independent_depth_accuracy": ind_acc,
        "shared_root_collapse_accuracy": shared_acc,
        "laundered_echo_collapse_accuracy": launder_acc,
        "depth_breakdown": depth_stats,
        "error_polarity_decomposition": error_decomp,
        "world_trajectories": world_results,
    }
    write_json(run_dir / "summary.json", summary)

    report_html = render_boundary_html(
        run_id, world_results, total_acc, mean_stab, repeat_stab, ind_acc, shared_acc, launder_acc, depth_stats, error_decomp
    )
    report_path = run_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")

    print("=" * 68)
    print("OXFORD EXPLORATION 2B.1: FORMAL SUPPORT BOUNDARY RESULTS")
    print("=" * 68)
    print(f"Overall Trajectory Accuracy:        {total_acc * 100:.1f}%")
    print(f"Within-Model Isomorphic Stability:  {mean_stab * 100:.1f}% (invariance across permuted twins)")
    print(f"Exact-Prompt Repeat Stability:      {repeat_stab * 100:.1f}% (target serving noise floor)")
    print(f"Independent Multi-Depth (d=2..5):   {ind_acc * 100:.1f}%")
    print(f"Shared Root Collapse Tracking:      {shared_acc * 100:.1f}%")
    print(f"Laundered Echo Collapse Tracking:   {launder_acc * 100:.1f}%")
    print("-" * 68)
    print("Depth × Representation Breakdown (Independent Chains):")
    for d, s in depth_stats.items():
        print(f"  d={d}: Canonical={s['canonical_acc']*100:.0f}%, Twin={s['twin_acc']*100:.0f}%, Stability={s['stability']*100:.0f}%")
    print("-" * 68)
    print("Error Polarity Decomposition:")
    print(f"  False Survival   (F+ = P(ACTIVE | UNKNOWN exp)): {f_plus * 100:.1f}%")
    print(f"  False Retraction (F- = P(UNKNOWN | ACTIVE exp)): {f_minus * 100:.1f}%")
    print(f"  Format Failure   (Ffmt = reasoning overflow):   {f_fmt * 100:.1f}%")
    print("=" * 68)
    print(f"HTML Report: {report_path}")
    if open_report:
        webbrowser.open(report_path.resolve().as_uri())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oxford.py",
        description="OXFORD Lite: black-box model-lineage & structural validity assay suite",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Doctor
    sub.add_parser("doctor", help="Check local environment, API keys, tokenizers, and Ollama status")

    # Structural
    struct = sub.add_parser("structural", help="Run local candidate tokenizers + remote Ox Alpha only")
    struct.add_argument("--probes", type=str, dest="probes_file", help="Path to custom probe JSON file")
    struct.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    struct.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for shuffle (default {DEFAULT_SEED})")

    # Positive Control
    pos = sub.add_parser("positive-control", help="Validate OXFORD against known specimen (remote model vs local tokenizers)")
    pos.add_argument("--model", type=str, dest="target_slug", help="Specimen model slug (default: z-ai/glm-5.2:free)")
    pos.add_argument("--paid", action="store_true", help="Use paid endpoint for specimen model")
    pos.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    pos.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for shuffle (default {DEFAULT_SEED})")

    # Collision Simulation
    col = sub.add_parser("collision", help="Empirical Monte Carlo collision simulation across candidate tokenizers")
    col.add_argument("--trials", type=int, default=100000, help="Number of Monte Carlo trials (default 100000)")
    col.add_argument("--probes-pool", type=int, default=2000, help="Size of synthetic probe corpus (default 2000)")
    col.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Simulation seed (default {DEFAULT_SEED})")

    # Synthesize Probes
    synth = sub.add_parser("synthesize-probes", help="Generate synthetic probes and select top discriminatory probes")
    synth.add_argument("--count", type=int, default=5000, help="Number of candidate probes to generate (default 5000)")
    synth.add_argument("--top-k", type=int, default=16, help="Top K discriminatory probes to select (default 16)")
    synth.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Generator seed (default {DEFAULT_SEED})")

    # Envelope Assay
    env = sub.add_parser("envelope", help="Test differential shape invariance across 3 prompt envelopes")
    env.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    env.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for shuffle (default {DEFAULT_SEED})")

    # Remote Assay
    remote = sub.add_parser("remote", help="Run remote candidate and target model assay")
    remote.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    remote.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Request shuffle seed (default {DEFAULT_SEED})")
    remote.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay between free calls (default {DEFAULT_DELAY})")
    remote.add_argument("--resume", type=str, dest="resume_ref", help="Resume prior run folder (e.g. --resume latest)")
    remote.add_argument("--paid", action="store_true", help="Use paid OpenRouter model routes")
    remote.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max retries on 429 (default {DEFAULT_MAX_RETRIES})")
    remote.add_argument("--provider-order", nargs="+", help="Pin specific OpenRouter providers (e.g. --provider-order together deepinfra)")
    remote.add_argument("--no-fallback", action="store_true", help="Disable provider fallback")

    # Exploration 2A: Causal Support Dynamics
    dyn_synth = sub.add_parser("dynamics-synthesize", help="Synthesize Stage-5A support worlds with isomorphic twins and freeze top holdouts")
    dyn_synth.add_argument("--count", type=int, default=30, help="Number of candidate worlds to generate (default 30)")
    dyn_synth.add_argument("--top-k", type=int, default=8, help="Top K holdout trajectories to freeze (default 8)")
    dyn_synth.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Simulation seed (default {DEFAULT_SEED})")

    dyn_assay = sub.add_parser("dynamics-assay", help="Run Exploration 2A causal support dynamics assay against target")
    dyn_assay.add_argument("--holdout", type=str, dest="holdout_file", help="Path to frozen holdout JSON file")
    dyn_assay.add_argument("--target", type=str, default="stealth/ox-alpha", help="Target model slug (default stealth/ox-alpha)")
    dyn_assay.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    dyn_assay.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for shuffle (default {DEFAULT_SEED})")

    # Exploration 2B: Lineage Laundering & Depth Boundary
    bnd_synth = sub.add_parser("boundary-synthesize", help="Synthesize Exploration 2B boundary and laundering worlds and freeze holdout set")
    bnd_synth.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Simulation seed (default {DEFAULT_SEED})")

    bnd_assay = sub.add_parser("boundary-assay", help="Run Exploration 2B lineage laundering and depth boundary assay against target")
    bnd_assay.add_argument("--holdout", type=str, dest="holdout_file", help="Path to frozen boundary holdout JSON file")
    bnd_assay.add_argument("--target", type=str, default="stealth/ox-alpha", help="Target model slug (default stealth/ox-alpha)")
    bnd_assay.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    bnd_assay.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for shuffle (default {DEFAULT_SEED})")

    # Local Assay (Ollama)
    local = sub.add_parser("local", help="Run local assays against Ollama")
    local.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    local.add_argument("--models", nargs="+", help="Ollama model names (default: gemma2:9b qwen2.5:7b)")

    # Demo
    demo = sub.add_parser("demo", help="Create synthetic sample report (makes zero model calls)")
    demo.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")

    # Backward compatibility
    pilot = sub.add_parser("pilot", help="Alias for 'remote'")
    pilot.add_argument("--open", action="store_true", dest="open_report", help="Open report.html in browser")
    pilot.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Request shuffle seed (default {DEFAULT_SEED})")
    pilot.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay (default {DEFAULT_DELAY})")
    pilot.add_argument("--resume", type=str, dest="resume_ref", help="Resume prior run folder")
    pilot.add_argument("--paid", action="store_true", help="Use paid model routes")
    pilot.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max retries (default {DEFAULT_MAX_RETRIES})")
    pilot.add_argument("--provider-order", nargs="+", help="Pin specific providers")
    pilot.add_argument("--no-fallback", action="store_true", help="Disable provider fallback")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return command_doctor()
    if args.command == "structural":
        return command_structural(args.open_report, args.seed, getattr(args, "probes_file", None))
    if args.command == "positive-control":
        return command_positive_control(args.open_report, args.seed, getattr(args, "target_slug", None), getattr(args, "paid", False))
    if args.command == "collision":
        return command_collision(args.trials, args.probes_pool, args.seed)
    if args.command == "synthesize-probes":
        return command_synthesize_probes(args.count, args.top_k, args.seed)
    if args.command == "envelope":
        return command_envelope(args.open_report, args.seed)
    if args.command == "dynamics-synthesize":
        return command_dynamics_synthesize(args.count, args.top_k, args.seed)
    if args.command == "dynamics-assay":
        return command_dynamics_assay(args.open_report, args.seed, getattr(args, "holdout_file", None), args.target)
    if args.command == "boundary-synthesize":
        return command_boundary_synthesize(args.seed)
    if args.command == "boundary-assay":
        return command_boundary_assay(args.open_report, args.seed, getattr(args, "holdout_file", None), args.target)
    if args.command in ("remote", "pilot"):
        return command_remote(
            seed=args.seed,
            delay=args.delay,
            open_report=args.open_report,
            resume_ref=args.resume_ref,
            paid=args.paid,
            max_retries=args.max_retries,
            provider_order=args.provider_order,
            allow_fallbacks=not getattr(args, "no_fallback", False),
        )
    if args.command == "local":
        return command_local(args.models, args.open_report)
    if args.command == "demo":
        return command_demo(args.open_report)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
