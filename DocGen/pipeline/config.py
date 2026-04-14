"""
Shared constants and path utility functions used across the pipeline.

Centralises mode-folder naming, template path resolution, noise-ratio
parsing, and related helpers so that every module uses the same logic.
"""

import math
import os
import re
from typing import Any, Dict, Optional, Tuple


MISC_CHARACTERS = set('!@#$%^&*()-_+=~`{}[]\\|:;\'"<,>./?…•–—―·°±×÷€£¥©®™§¶†‡')


def get_output_root(base_dir: str) -> str:
    """Return the root directory for generated artifacts (templates, variations, documents).

    ``base_dir`` is the DocGen directory. Outputs are stored under the sibling ``DocSets``
    folder at the repository root, preserving the internal layout (``templates/``,
    ``variations/``, ``documents/``, etc.).
    """
    docgen = os.path.abspath(base_dir)
    repo_root = os.path.dirname(docgen)
    return os.path.join(repo_root, "DocSets")


def get_repo_root(base_dir: str) -> str:
    """Return the Rosetta repository root (parent directory of DocGen ``base_dir``)."""
    docgen = os.path.abspath(base_dir)
    return os.path.dirname(docgen)


def load_repo_dotenv(base_dir: str) -> None:
    """Load ``.env`` from ``<repo>/Config/.env``, then ``<DocGen>/config/.env``, then cwd.

    Uses python-dotenv's default (does not override variables already set in the
    process environment). Repository-level ``Config/.env`` is loaded first so it
    takes precedence over legacy ``DocGen/config/.env`` and a local ``.env`` in
    the current working directory.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    docgen = os.path.abspath(base_dir)
    for path in (
        os.path.join(get_repo_root(base_dir), "Config", ".env"),
        os.path.join(docgen, "config", ".env"),
    ):
        if os.path.isfile(path):
            load_dotenv(path)
    load_dotenv()


_LLM_CONFIG_DOTENV_DONE = False

_DEFAULT_LOCAL_LLM_MODEL = "openai/gpt-oss-120b"
_DEFAULT_CLOUD_PARSING_MODEL = "gpt-4o-mini"


def _ensure_llm_config_dotenv() -> None:
    """Load repo ``.env`` before reading ``ROSETTA_LOCAL_LLM_*`` (some import paths skip ``template_generator``)."""
    global _LLM_CONFIG_DOTENV_DONE
    if _LLM_CONFIG_DOTENV_DONE:
        return
    docgen_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_repo_dotenv(docgen_root)
    _LLM_CONFIG_DOTENV_DONE = True


def is_local_llm_configured() -> bool:
    """True when ``ROSETTA_LOCAL_LLM_BASE_URL`` is set; pipeline uses that OpenAI-compatible server for parsing/variation LLM calls."""
    _ensure_llm_config_dotenv()
    return bool(os.getenv("ROSETTA_LOCAL_LLM_BASE_URL", "").strip())


def normalize_openai_compat_base_url(url: str) -> str:
    """Ensure a trailing ``/v1`` segment for OpenAI-compatible HTTP APIs (Ollama, LM Studio, vLLM, etc.)."""
    u = (url or "").strip().rstrip("/")
    if not u:
        raise ValueError("OpenAI-compatible base URL is empty")
    if u.endswith("/v1"):
        return u
    return f"{u}/v1"


def get_local_llm_base_url() -> Optional[str]:
    """Normalized base URL when ``ROSETTA_LOCAL_LLM_BASE_URL`` is set; otherwise ``None``."""
    _ensure_llm_config_dotenv()
    raw = os.getenv("ROSETTA_LOCAL_LLM_BASE_URL", "").strip()
    if not raw:
        return None
    return normalize_openai_compat_base_url(raw)


def get_local_llm_api_key() -> str:
    """API key sent to the local OpenAI-compatible server (often ignored); default ``EMPTY``."""
    _ensure_llm_config_dotenv()
    key = (os.getenv("ROSETTA_LOCAL_LLM_API_KEY") or "EMPTY").strip()
    return key if key else "EMPTY"


def get_configured_local_llm_model() -> str:
    """Model id for the OpenAI-compatible server when ``ROSETTA_LOCAL_LLM_BASE_URL`` is set."""
    _ensure_llm_config_dotenv()
    model = (os.getenv("ROSETTA_LOCAL_LLM_MODEL") or _DEFAULT_LOCAL_LLM_MODEL).strip()
    return model if model else _DEFAULT_LOCAL_LLM_MODEL


def get_cloud_parsing_fallback_model() -> str:
    """OpenAI model used for parsing/variation calls when no local server is configured (``ROSETTA_CLOUD_PARSING_MODEL``)."""
    _ensure_llm_config_dotenv()
    model = (os.getenv("ROSETTA_CLOUD_PARSING_MODEL") or _DEFAULT_CLOUD_PARSING_MODEL).strip()
    return model if model else _DEFAULT_CLOUD_PARSING_MODEL


def get_parsing_llm_model() -> str:
    """Model id for all former \"local LLM\" chat completions (local server if configured, else OpenAI fallback)."""
    if is_local_llm_configured():
        return get_configured_local_llm_model()
    return get_cloud_parsing_fallback_model()


def create_parsing_llm_openai_client():
    """``OpenAI`` client for parsing / variation / narrative-analysis LLM calls: local-compatible server or OpenAI cloud fallback."""
    import openai

    _ensure_llm_config_dotenv()
    if is_local_llm_configured():
        base = get_local_llm_base_url()
        if not base:
            raise ValueError("ROSETTA_LOCAL_LLM_BASE_URL is set but invalid")
        return openai.OpenAI(base_url=base, api_key=get_local_llm_api_key())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when ROSETTA_LOCAL_LLM_BASE_URL is not set "
            "(parsing and variation steps fall back to the OpenAI API)"
        )
    return openai.OpenAI(api_key=api_key)


def get_local_llm_model() -> str:
    """Same as :func:`get_parsing_llm_model` (backward-compatible name)."""
    return get_parsing_llm_model()


def create_local_llm_openai_client():
    """Same as :func:`create_parsing_llm_openai_client` (backward-compatible name)."""
    return create_parsing_llm_openai_client()


def get_ground_truth_data_path(base_dir: str, dataset_folder_name: str) -> str:
    """Return ``<repo>/GroundTruth/<dataset_folder_name>/`` for bundled datasets (e.g. MINIDEV)."""
    return os.path.join(get_repo_root(base_dir), "GroundTruth", dataset_folder_name)


def get_ground_truth_table_sample_description_dir(base_dir: str, dataset_folder_name: str) -> str:
    """Return ``<repo>/GroundTruth/<dataset>/table_sample_data/description/``."""
    return os.path.join(
        get_ground_truth_data_path(base_dir, dataset_folder_name),
        "table_sample_data",
        "description",
    )


def get_ground_truth_column_descriptors_enhanced_path(base_dir: str) -> str:
    """Return ``<repo>/GroundTruth/column_descriptors_enhanced.json``."""
    return os.path.join(get_repo_root(base_dir), "GroundTruth", "column_descriptors_enhanced.json")


def get_ground_truth_column_descriptors_path(base_dir: str) -> str:
    """Return ``<repo>/GroundTruth/column_descriptors.json`` (base descriptors for setup)."""
    return os.path.join(get_repo_root(base_dir), "GroundTruth", "column_descriptors.json")


def get_ground_truth_context_cache_path(base_dir: str, dataset_folder_name: str) -> str:
    """Per-dataset LLM context cache under ``GroundTruth/<dataset>/``."""
    return os.path.join(get_ground_truth_data_path(base_dir, dataset_folder_name), "context_cache.json")


def table_sample_entry_path(
    base_dir: str, dataset_folder_name: str, database: str, table: str, index: int
) -> str:
    """Path to one row sample text file (e.g. ``table0.txt``)."""
    return os.path.join(
        get_ground_truth_table_sample_description_dir(base_dir, dataset_folder_name),
        database,
        table,
        f"{table}{index}.txt",
    )


def get_mode_folder_name(null_mode: str, binary_mode: str) -> str:
    """Return the canonical folder name for a null/binary mode combination."""
    if null_mode == "explicit" and binary_mode == "explicit":
        return "null_binary_explicit"
    elif null_mode == "explicit" and binary_mode == "implicit":
        return "null_explicit_binary_implicit"
    elif null_mode == "implicit" and binary_mode == "explicit":
        return "null_implicit_binary_explicit"
    else:
        return "null_binary_implicit"


def get_noise_ratio_folder_name(data_noise_x: int, data_noise_y: int) -> str:
    """Return the folder name encoding a data:noise ratio, e.g. '1_data_0_noise'."""
    return f"{data_noise_x}_data_{data_noise_y}_noise"


def get_sentence_template_path(base_dir: str, null_mode: str, binary_mode: str, db_name: str, table_name: str) -> str:
    """Return the full path for a sentence template JSON file."""
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    return os.path.join(get_output_root(base_dir), "templates", mode_folder, "sentence_templates", db_name, f"{table_name}_template.json")


def get_narrative_template_path(base_dir: str, null_mode: str, binary_mode: str, data_noise_x: int, data_noise_y: int, db_name: str, table_name: str) -> str:
    """Return the full path for a narrative template JSON file."""
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    ratio_folder = get_noise_ratio_folder_name(data_noise_x, data_noise_y)
    return os.path.join(get_output_root(base_dir), "templates", mode_folder, "narrative_templates", ratio_folder, db_name, f"{table_name}_template.json")


def get_sentence_templates_dir(base_dir: str, null_mode: str, binary_mode: str) -> str:
    """Return the directory that holds all sentence template folders."""
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    return os.path.join(get_output_root(base_dir), "templates", mode_folder, "sentence_templates")


def variation_templates_exist(
    base_dir: str, null_mode: str, binary_mode: str, db_name: str, table_name: str,
) -> bool:
    """Return True if a sentence-variation JSON already exists under any ratio folder."""
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    variations_root = os.path.join(get_output_root(base_dir), "variations", mode_folder)
    if not os.path.isdir(variations_root):
        return False
    target = f"{table_name}_sentence_templates.json"
    for ratio_entry in os.listdir(variations_root):
        candidate = os.path.join(variations_root, ratio_entry, db_name, target)
        if os.path.isfile(candidate):
            return True
    return False


def get_narrative_templates_dir(base_dir: str, null_mode: str, binary_mode: str, data_noise_x: int, data_noise_y: int) -> str:
    """Return the directory for narrative templates at a given noise ratio."""
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    ratio_folder = get_noise_ratio_folder_name(data_noise_x, data_noise_y)
    return os.path.join(get_output_root(base_dir), "templates", mode_folder, "narrative_templates", ratio_folder)


def parse_data_noise_ratio_str(ratio: Optional[str], default_x: int = 1, default_y: int = 0) -> Tuple[int, int]:
    """Parse 'X:Y' from template metadata; fall back to defaults on missing or invalid input."""
    if not ratio or not isinstance(ratio, str):
        return (default_x, default_y)
    parts = ratio.strip().split(':')
    if len(parts) != 2:
        return (default_x, default_y)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (default_x, default_y)


def parse_ratio_folder_name(folder_name: str) -> Optional[Tuple[int, int]]:
    """Parse e.g. '1_data_1_noise' -> (1, 1)."""
    m = re.match(r'^(\d+)_data_(\d+)_noise$', folder_name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def find_narrative_template_path_scan(
    base_dir: str, null_mode: str, binary_mode: str, db_name: str, table_name: str
) -> Optional[str]:
    """If a narrative file exists under any ratio subfolder, return its path (first match)."""
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    root = os.path.join(get_output_root(base_dir), "templates", mode_folder, "narrative_templates")
    if not os.path.isdir(root):
        return None
    target = f"{table_name}_template.json"
    for entry in sorted(os.listdir(root)):
        candidate = os.path.join(root, entry, db_name, target)
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_backfill_noise_xy_and_narrative_path(
    base_dir: str,
    null_mode: str,
    binary_mode: str,
    sentence_data: Dict[str, Any],
    default_x: int = 1,
    default_y: int = 0,
) -> Tuple[int, int, Optional[str]]:
    """
    Resolve data:noise X/Y and the narrative JSON path for backfill.

    Uses sentence template's data_noise_ratio when the narrative file exists there; otherwise
    if a narrative file exists under any other ratio folder (e.g. stale 1:0 on sentence but
    file under 1_data_1_noise), uses that folder's X/Y and path.
    """
    db = sentence_data.get('database', '')
    table = sentence_data.get('table', '')
    sx, sy = parse_data_noise_ratio_str(sentence_data.get('data_noise_ratio'), default_x, default_y)

    path_from_sentence: Optional[str] = None
    if sy > 0:
        path_from_sentence = get_narrative_template_path(
            base_dir, null_mode, binary_mode, sx, sy, db, table
        )

    scanned_path = find_narrative_template_path_scan(base_dir, null_mode, binary_mode, db, table)

    if path_from_sentence and os.path.isfile(path_from_sentence):
        return sx, sy, path_from_sentence

    if scanned_path and os.path.isfile(scanned_path):
        ratio_folder = os.path.basename(os.path.dirname(os.path.dirname(scanned_path)))
        parsed = parse_ratio_folder_name(ratio_folder)
        if parsed:
            rx, ry = parsed
            return rx, ry, scanned_path
        return sx, sy, scanned_path

    if sy > 0:
        return sx, sy, path_from_sentence
    return sx, sy, None


def compute_expected_transition_count(num_data_sentences: int, data_noise_x: int, data_noise_y: int) -> int:
    """
    Compute the expected number of transition (noise) sentences for a given ratio.

    When Y=0, returns 0 (no transitions).
    When Y>0, returns max(Y, ceil(N * Y / X)) where N = num_data_sentences.
    This ensures at least Y transitions even when N < X.
    """
    if data_noise_y == 0:
        return 0
    proportional = math.ceil(num_data_sentences * data_noise_y / data_noise_x)
    return max(data_noise_y, proportional)
