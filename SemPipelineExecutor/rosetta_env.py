"""
Load API keys and shared paths for SemPipelineExecutor.

``OPENAI_API_KEY`` (and other vars) are read from ``<repo>/Config/.env``,
then legacy ``DocGen/config/.env``, then the process environment / cwd ``.env``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

EXECUTOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXECUTOR_ROOT.parent


def load_rosetta_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (
        REPO_ROOT / "Config" / ".env",
        REPO_ROOT / "DocGen" / "config" / ".env",
    ):
        if path.is_file():
            load_dotenv(path)
    load_dotenv()


def evaluation_results_dir(
    system_name: str,
    null_param: Optional[str],
    noise_param: Optional[str],
    num_documents: int,
) -> Path:
    """``Rosetta/Results/{system}/{null_mode}/{noise_ratio}/{N}_documents/``

    Mirrors DocSets layout: ``documents/{mode_folder}/{ratio_folder}/...``.
    When ``noise_param`` is omitted, uses ``1_data_0_noise`` (data-only, no narrative noise).
    When ``null_param`` is omitted, uses ``default``.
    """
    mode_folder = null_param if null_param else "default"
    ratio_folder = noise_param if noise_param else "1_data_0_noise"
    out = REPO_ROOT / "Results" / system_name / mode_folder / ratio_folder / f"{int(num_documents)}_documents"
    out.mkdir(parents=True, exist_ok=True)
    return out
