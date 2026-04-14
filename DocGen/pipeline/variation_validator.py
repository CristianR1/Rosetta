"""
Validate and correct variation template placeholders across all generated
variation JSONs.

**Detection** (inherited from the original pipeline validation logic):

1. Placeholder gaps — non-static entries whose variations are missing the
   expected ``[COLUMN]`` placeholder.
2. Misgeneration — static entries whose source sentence carried a
   ``(Hash: …)`` tag, meaning the sentence was about a real data column but the
   parsing logic could not locate the data value.

**Correction** (new):

For every detected misgeneration the script regenerates the template entry
*in place* using a **placeholder-first** LLM prompt (the LLM is asked to
include ``[COLUMN_NAME]`` directly in the sentence instead of generating a
concrete value that must later be parsed out).  The null/binary mode is
derived from the folder name of the config that is currently being processed
so variations, counter-variations, and null-variations are generated with
the correct encoding.

For orphaned columns that could not be matched to any existing static
entry, a brand-new template entry is **inserted** at the end of the
templates list.  The original sentence is first generated *plainly* (with a
concrete sample value, mirroring the original pipeline), and then the
placeholder-first prompt produces the ``template_pattern`` and variation
banks.

All other template entries in each variation JSON are preserved verbatim.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent))

from .variation_generator import VariationBankGenerator
from .template_system import DocumentTemplateSystem
from .data_loader import load_column_descriptors
from .config import get_output_root, load_repo_dotenv

load_repo_dotenv(str(_script_dir.parent))


HASH_TAG_RE = re.compile(r'\s*\(Hash:\s*[0-9a-fA-F]+\)\s*$')


def safe_print(text: str):
    """Print text, replacing non-encodable characters with ASCII approximations."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        print(encoded)


def parse_modes_from_config(config_folder: str) -> tuple:
    """Return (null_mode, binary_mode) from a config folder name."""
    mapping = {
        "null_binary_explicit": ("explicit", "explicit"),
        "null_explicit_binary_implicit": ("explicit", "implicit"),
        "null_implicit_binary_explicit": ("implicit", "explicit"),
        "null_binary_implicit": ("implicit", "implicit"),
    }
    return mapping.get(config_folder, ("implicit", "implicit"))


def extract_path_parts(json_path: Path, variations_root: Path) -> dict:
    """Return config, noise_ratio, database, table from a variation file path."""
    rel = json_path.relative_to(variations_root)
    parts = rel.parts
    return {
        "config": parts[0],
        "noise_ratio": parts[1],
        "database": parts[-2],
        "table": parts[-1].replace("_sentence_templates.json", ""),
    }


def resolve_template_paths(json_path: Path, variations_root: Path) -> tuple:
    """Derive sentence-template and narrative-template paths."""
    p = extract_path_parts(json_path, variations_root)
    results_root = variations_root.parent
    sentence_tmpl = (results_root / "templates" / p["config"] / "sentence_templates"
                     / p["database"] / f"{p['table']}_template.json")
    narrative_tmpl = (results_root / "templates" / p["config"] / "narrative_templates"
                      / p["noise_ratio"] / p["database"] / f"{p['table']}_template.json")
    return sentence_tmpl, narrative_tmpl


def parse_hashed_sentences_from_sentence_template(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for sent in data.get("generated_sentences", []):
        m = HASH_TAG_RE.search(sent)
        if m:
            hash_val = m.group().strip().lstrip("(").split()[-1].rstrip(")")
            clean = HASH_TAG_RE.sub("", sent).strip()
            result[hash_val] = clean
    return result


def parse_hashed_sentences_from_narrative(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    narrative = data.get("narrative", [])
    narrative_str = "\n\n".join(narrative) if isinstance(narrative, list) else str(narrative)
    result = {}
    for segment in narrative_str.split("|"):
        segment = segment.strip()
        if not segment:
            continue
        m = HASH_TAG_RE.search(segment)
        if m:
            hash_val = m.group().strip().lstrip("(").split()[-1].rstrip(")")
            clean = HASH_TAG_RE.sub("", segment).strip()
            result[hash_val] = clean
    return result


def _normalise(text: str) -> set:
    text = re.sub(r'[^\w\s]', " ", text.lower())
    return {w for w in text.split() if len(w) > 2}


def _texts_similar(text1: str, text2: str, threshold: float = 0.55) -> bool:
    w1, w2 = _normalise(text1), _normalise(text2)
    if not w1 or not w2:
        return False
    return len(w1 & w2) / len(w1 | w2) >= threshold


def build_placeholder(field_name: str) -> str:
    return f"[{field_name.upper()}]"


def placeholder_present(sentence: str, placeholder: str) -> bool:
    return placeholder.lower() in sentence.lower()


def classify_template(entry: dict) -> str:
    if entry.get("is_static", False):
        return "static"
    has_variations = bool(entry.get("variations"))
    has_counter = bool(entry.get("counter_variations"))
    has_null = bool(entry.get("null_variations"))
    if has_variations and has_counter and has_null:
        return "nullable_binary"
    if has_variations and has_counter and not has_null:
        counter_text = " ".join(entry["counter_variations"]).lower()
        if "null" in counter_text:
            return "null"
        return "binary"
    return "standard"


def extract_table_db_from_path(json_path: Path, variations_root: Path) -> tuple:
    rel = json_path.relative_to(variations_root)
    parts = rel.parts
    database = parts[-2] if len(parts) >= 2 else "unknown_db"
    table = parts[-1].replace("_sentence_templates.json", "")
    return database, table


def validate_file(json_path: Path, variations_root: Path) -> list:
    problems = []
    database, table = extract_table_db_from_path(json_path, variations_root)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, entry in enumerate(data.get("templates", [])):
        classification = classify_template(entry)
        if classification in ("nullable_binary", "binary", "static"):
            continue
        primary_fields = entry.get("primary_data_fields", [])
        if not primary_fields:
            continue
        field_name = primary_fields[0]
        ph = build_placeholder(field_name)
        variations = entry.get("variations", [])
        missing = [{"index": vi, "sentence": v}
                   for vi, v in enumerate(variations) if not placeholder_present(v, ph)]
        if missing:
            problems.append({
                "database": database, "table": table,
                "field_name": field_name, "classification": classification,
                "placeholder": ph, "template_index": idx,
                "original_sentence": entry.get("original", "")[:120],
                "template_pattern": entry.get("template_pattern", "")[:120],
                "total_variations": len(variations),
                "missing_count": len(missing), "missing_variations": missing,
            })
    return problems


def fix_placeholder_gaps(
    json_path: Path,
    issues: list,
    vgen: VariationBankGenerator,
    doc_context: str,
) -> int:
    """Regenerate variations that are missing the expected placeholder.

    For each issue, regenerates replacement variations using the existing
    template_pattern and splices them into the correct indices.

    Returns the number of variations fixed.
    """
    if not issues:
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_fixed = 0
    for issue in issues:
        idx = issue["template_index"]
        entry = data["templates"][idx]
        field_name = issue["field_name"]
        placeholder = issue["placeholder"]
        template_pattern = entry.get("template_pattern", "")
        missing_indices = [m["index"] for m in issue["missing_variations"]]

        if not template_pattern or placeholder.lower() not in template_pattern.lower():
            safe_print(f"    WARNING: template_pattern missing placeholder for {field_name} — skipping")
            continue

        safe_print(f"    Regenerating {len(missing_indices)} variation(s) for {field_name}...")

        for mi in missing_indices:
            prompt = f"""Rewrite this sentence while keeping exactly the same meaning and the placeholder {placeholder} in the same position:

Original: {template_pattern}

RULES:
1. The placeholder {placeholder} MUST appear EXACTLY ONCE
2. Keep the same meaning and information
3. Use different wording/structure
4. Write natural, professional prose
5. Return ONLY the rewritten sentence

Rewritten:"""

            system_msg = (
                "You are a sentence rewriter. Preserve meaning and placeholders exactly."
            )

            for _attempt in range(vgen.max_retries):
                response = vgen.query_local_llm(prompt, system_msg)
                if response:
                    result = response.strip()
                    result = re.sub(r'^\d+[\.\)]\s*', '', result)
                    result = re.sub(r'^[-•*]\s*', '', result)
                    if placeholder.lower() in result.lower() and placeholder not in result:
                        result = re.sub(
                            re.escape(placeholder), placeholder, result,
                            count=1, flags=re.IGNORECASE,
                        )
                    if placeholder in result and len(result) > 15:
                        entry["variations"][mi] = result
                        total_fixed += 1
                        break

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    json_str = json_str.replace('\\"[', '[').replace(']\\"', ']')
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    return total_fixed


def detect_misgeneration(json_path: Path, variations_root: Path) -> list:
    """Detect columns whose template entry lacks a proper primary placeholder.

    A column is "orphaned" when it appears in ``hash_to_column`` but no
    non-static template entry lists it in ``primary_data_fields``.

    For every orphaned column the function locates the **exact** template
    entry whose ``original`` text matches the source sentence from
    ``generated_sentences`` (hash tag stripped).  The search covers *all*
    entries — static and non-static alike — using exact string matching.

    Returns a list of dicts.  Entries with ``template_index >= 0`` are
    correctable in-place; entries with ``template_index == -1`` could not be
    located (truly missing).
    """
    problems = []
    database, table = extract_table_db_from_path(json_path, variations_root)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hash_to_col = data.get("hash_to_column", {})
    tbd_cols = {c.lower() for c in data.get("tbd_columns", [])}
    templates = data.get("templates", [])
    if not hash_to_col:
        return problems

    handled_columns: set = set()
    for entry in templates:
        if not entry.get("is_static", False):
            for fn in entry.get("primary_data_fields", []):
                handled_columns.add(fn)

    all_hashed = set(hash_to_col.values())
    orphaned = all_hashed - handled_columns - {c for c in all_hashed if c.lower() in tbd_cols}
    if not orphaned:
        return problems

    sent_tmpl, narr_tmpl = resolve_template_paths(json_path, variations_root)
    hashed_sentences: dict = {}
    hashed_sentences.update(parse_hashed_sentences_from_sentence_template(sent_tmpl))
    hashed_sentences.update(parse_hashed_sentences_from_narrative(narr_tmpl))

    orphaned_col_sources: dict = {}
    for h, col in hash_to_col.items():
        if col in orphaned and h in hashed_sentences:
            orphaned_col_sources.setdefault(col, []).append(hashed_sentences[h])

    original_to_idx: dict = {}
    for idx, entry in enumerate(templates):
        orig = entry.get("original", "")
        if orig:
            original_to_idx[orig] = idx

    matched_cols: set = set()
    for col in sorted(orphaned):
        src_sents = orphaned_col_sources.get(col, [])
        matched_idx = -1
        matched_src = ""

        for s in src_sents:
            if s in original_to_idx:
                matched_idx = original_to_idx[s]
                matched_src = s[:120]
                break

        if matched_idx == -1:
            for s in src_sents:
                for idx, entry in enumerate(templates):
                    if _texts_similar(entry.get("original", ""), s):
                        matched_idx = idx
                        matched_src = s[:120]
                        break
                if matched_idx >= 0:
                    break

        if col in matched_cols:
            continue
        matched_cols.add(col)

        if matched_idx >= 0:
            orig_text = templates[matched_idx].get("original", "")[:120]
            problems.append({
                "database": database, "table": table,
                "template_index": matched_idx,
                "original_sentence": orig_text,
                "matched_column": col,
                "source_sentence": matched_src,
                "issue_type": "misgeneration",
            })
        else:
            problems.append({
                "database": database, "table": table,
                "template_index": -1,
                "original_sentence": "",
                "matched_column": col,
                "source_sentence": src_sents[0][:120] if src_sents else "",
                "issue_type": "misgeneration_unmatched",
            })

    return problems


def load_data_fields_from_sentence_template(sent_tmpl_path: Path) -> dict:
    """Load the original_data dict from a sentence template file."""
    if not sent_tmpl_path.is_file():
        return {}
    with open(sent_tmpl_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("original_data", {})


def load_field_metadata(database: str, table: str, data_fields: dict,
                        base_dir: Path) -> tuple:
    """Return (field_metadata, field_data_types) via DocumentTemplateSystem."""
    dts = DocumentTemplateSystem.__new__(DocumentTemplateSystem)
    dts._null_mode = "implicit"
    dts._binary_mode = "implicit"
    dts.variation_generator = None
    dts.templates = []
    dts.consistency_mapping = {}
    dts.document_context = ""
    dts.local_llm_client = None

    old_cwd = os.getcwd()
    try:
        pipeline_dir = base_dir / "pipeline"
        if pipeline_dir.is_dir():
            os.chdir(pipeline_dir)
        field_metadata, _field_values, field_data_types = (
            dts.identify_binary_null_fields(database, table, data_fields, 1000)
        )
    finally:
        os.chdir(old_cwd)
    return field_metadata, field_data_types


def get_column_descriptor(column_descriptors: dict, database: str,
                          table: str, column: str) -> str:
    """Extract a human-readable descriptor for a column."""
    table_desc = column_descriptors.get(database, {}).get(table, {})
    info = table_desc.get(column, {})
    if isinstance(info, dict):
        return info.get("descriptor", f"Column: {column}")
    return f"Column: {column}"


def generate_sentence_with_placeholder(
    vgen: VariationBankGenerator,
    column_name: str,
    descriptor: str,
    field_type: str,
    null_mode: str,
    binary_mode: str,
    context: str = "",
) -> str:
    """Ask the LLM to generate a sentence that already contains ``[COLUMN]``."""
    placeholder = f"[{column_name.upper()}]"
    natural_name = column_name.replace("_", " ")
    context_line = f"Context: {context}" if context else "Context: professional data report"

    mode_guidance = ""
    if field_type == "BINARY":
        if binary_mode == "explicit":
            mode_guidance = (
                f"\n6. The placeholder {placeholder} stands for a binary 0 or 1 value."
                "\n7. Do NOT convert to yes/no or true/false — keep the literal 0/1 concept."
            )
        else:
            mode_guidance = (
                f"\n6. The placeholder {placeholder} stands for a binary state."
                "\n7. Express the concept so the placeholder can be replaced by natural "
                "language like 'is' / 'is not' phrasing."
            )
    elif field_type in ("NULL", "NULLABLE_BINARY"):
        if null_mode == "explicit":
            mode_guidance = (
                f"\n6. The placeholder {placeholder} may hold NULL for missing values."
                "\n7. Do NOT use 'not specified' — keep the literal NULL concept."
            )
        else:
            mode_guidance = (
                f"\n6. The placeholder {placeholder} may hold a missing-value indicator."
            )

    prompt = f"""Generate a single well-formed, natural sentence about the field "{natural_name}" \
that includes the placeholder {placeholder} where the actual data value would appear.

{context_line}

RULES:
1. The placeholder {placeholder} MUST appear EXACTLY ONCE in the sentence
2. The sentence must read naturally with the placeholder standing in for a real value
3. Mention the concept of "{natural_name}" naturally in the sentence
4. Write one sentence of about 15-25 words of natural, professional prose
5. Do NOT mention databases, columns, fields, tables, or rows{mode_guidance}

Return ONLY the sentence, no extra text.

Sentence:"""

    system_message = (
        "You are a sentence generator that creates natural, professional sentences "
        "incorporating a bracketed placeholder for a data value. "
        "Always use natural field names with spaces, never underscores."
    )

    for _attempt in range(vgen.max_retries):
        response = vgen.query_local_llm(prompt, system_message)
        if response:
            result = response.strip()
            result = re.sub(r'^\d+[\.\)]\s*', '', result)
            result = re.sub(r'^[-•*]\s*', '', result)
            if placeholder.lower() in result.lower() and len(result) > 15:
                if placeholder not in result:
                    result = re.sub(
                        re.escape(placeholder), placeholder, result, count=1,
                        flags=re.IGNORECASE,
                    )
                return result

    return f"The {natural_name} is {placeholder}."


def generate_plain_original_sentence(
    vgen: VariationBankGenerator,
    column_name: str,
    sample_value: str,
    descriptor: str,
    context: str = "",
) -> str:
    """Generate a natural sentence with an actual *sample_value* embedded.

    This mirrors what the original pipeline would have produced had the
    parsing logic succeeded.  The sentence is used as the ``original`` field
    for newly inserted template entries.
    """
    safe_print(f"    Generating plain sentence with sample value '{sample_value[:40]}' ...")
    sentence = vgen.generate_sentence_for_field_value(
        column_name, str(sample_value), context or "professional data report",
        natural_mode=False,
    )
    if sentence and len(sentence) > 10:
        safe_print(f"    Plain original: {sentence[:100]}")
        return sentence
    fallback = f"The {column_name.replace('_', ' ')} is {sample_value}."
    safe_print(f"    Plain original (fallback): {fallback}")
    return fallback


def correct_misgenerated_entry(
    column_name: str,
    field_type: str,
    null_mode: str,
    binary_mode: str,
    document_context: str,
    descriptor: str,
    data_fields: dict,
    vgen: VariationBankGenerator,
    plain_original: str = "",
) -> dict:
    """Generate a fully corrected template entry dict for *column_name*.

    If *plain_original* is provided it is used as the ``original`` field in
    the returned dict (used for newly inserted entries whose original
    sentence was generated with a concrete value).  Otherwise the
    placeholder-bearing base sentence is used.

    Returns a dict ready to be merged into the variation JSON ``templates``
    list (same schema as existing entries).
    """
    placeholder = f"[{column_name.upper()}]"
    context = document_context or "professional data report"

    safe_print(f"    Generating base sentence with placeholder {placeholder} ...")
    base_sentence = generate_sentence_with_placeholder(
        vgen, column_name, descriptor, field_type, null_mode, binary_mode, context,
    )
    safe_print(f"    Base: {base_sentence[:100]}")

    null_replacement_phrase = "NULL" if null_mode == "explicit" else "not specified"
    variations = []
    counter_variations = []
    null_variations = []
    field_data_types_entry: dict = {}

    if field_type == "STANDARD":
        safe_print(f"    Generating STANDARD variations ...")
        variations = vgen.generate_structural_variations_standard(
            base_sentence, column_name, context)

    elif field_type == "BINARY":
        safe_print(f"    Generating BINARY variations ...")
        variations = vgen.generate_structural_variations_standard_with_style(
            base_sentence, column_name, context, original_sentence=base_sentence)
        safe_print(f"    Generating BINARY counter variations ...")
        counter_variations = vgen.generate_structural_variations_binary_counter(
            base_sentence, column_name, "1", context)
        field_data_types_entry[column_name] = "bool"

    elif field_type == "NULL":
        safe_print(f"    Generating NULL (non-null side) variations ...")
        variations = vgen.generate_structural_variations_standard(
            base_sentence, column_name, context)
        safe_print(f"    Generating NULL (null side) counter variations ...")
        null_base = vgen.generate_sentence_for_field_value(
            column_name, null_replacement_phrase, context,
            natural_mode=False, original_sentence=base_sentence)
        counter_variations = vgen.generate_null_variations_null(
            null_base, column_name, null_replacement_phrase, context)

    elif field_type == "NULLABLE_BINARY":
        safe_print(f"    Generating NULLABLE_BINARY all three banks ...")
        variations = vgen.generate_structural_variations_standard_with_style(
            base_sentence, column_name, context, original_sentence=base_sentence)
        counter_variations = vgen.generate_structural_variations_binary_counter(
            base_sentence, column_name, "1", context)
        null_base = vgen.generate_sentence_for_field_value(
            column_name, null_replacement_phrase, context,
            natural_mode=False, original_sentence=base_sentence)
        null_variations = vgen.generate_null_variations_null(
            null_base, column_name, null_replacement_phrase, context)
        field_data_types_entry[column_name] = "bool"

    valid_variations = [v for v in variations if placeholder.lower() in v.lower()]
    if len(valid_variations) < len(variations):
        safe_print(f"    Filtered {len(variations) - len(valid_variations)} variations "
                    f"missing placeholder")
    variations = valid_variations or variations[:1]

    words = re.findall(r'\b[a-zA-Z]{4,}\b', base_sentence.lower())
    stop_words = {
        'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were',
        'will', 'would', 'could', 'should',
    }
    field_words = {
        'code', 'name', 'number', 'type', 'status', 'count', 'identifier',
        'date', 'time', 'year', 'percent', 'percentage', 'total', 'amount', 'value',
    }
    filtered = [w for w in words if w not in stop_words and w not in field_words and len(w) > 3]
    key_words = list(set(filtered))[:8]

    lexical_sets: dict = {}
    if key_words:
        safe_print(f"    Generating lexical sets for: {key_words}")
        lexical_sets = vgen.generate_lexical_variations(key_words, context)

    return {
        "original": plain_original if plain_original else base_sentence,
        "template_pattern": base_sentence,
        "primary_data_fields": [column_name],
        "foreign_data_fields": [],
        "variations": variations,
        "counter_variations": counter_variations,
        "null_variations": null_variations,
        "lexical_sets": lexical_sets,
        "field_data_types": field_data_types_entry,
        "is_static": False,
    }


def patch_variation_file(json_path: Path, corrections: list,
                         insertions: list | None = None):
    """Apply replacements and/or insertions to the variation JSON file.

    *corrections*: list of ``{"template_index": int, "new_entry": dict}``
        — replaces the template at the given index.
    *insertions*: list of ``{"new_entry": dict, "column": str}``
        — for each insertion, one static entry is **removed** and the new
          data-bearing entry is appended at the end, keeping the total
          template count (and therefore the data-to-noise ratio) constant.
          The column is also removed from ``tbd_columns`` if present.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for corr in corrections:
        idx = corr["template_index"]
        data["templates"][idx] = corr["new_entry"]

    corrected_indices = {c["template_index"] for c in corrections}

    for ins in (insertions or []):
        removed = False
        for i, entry in enumerate(data["templates"]):
            if i in corrected_indices:
                continue
            if entry.get("is_static", False):
                data["templates"].pop(i)
                corrected_indices = {
                    (idx - 1 if idx > i else idx) for idx in corrected_indices
                }
                removed = True
                break
        if not removed:
            safe_print(f"    WARNING: no static entry available to swap — "
                        f"appending without removal for column '{ins.get('column', '?')}'")

        data["templates"].append(ins["new_entry"])
        col = ins.get("column", "")
        if col:
            tbd = data.get("tbd_columns", [])
            data["tbd_columns"] = [c for c in tbd if c != col]

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    json_str = json_str.replace('\\"[', '[').replace(']\\"', ']')
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def find_variations_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    docgen = script_dir.parent
    candidate = Path(get_output_root(str(docgen))) / "variations"
    if candidate.is_dir():
        return candidate
    cwd_docsets = Path.cwd() / "DocSets" / "variations"
    if cwd_docsets.is_dir():
        return cwd_docsets
    cwd_legacy = Path.cwd() / "DGS" / "results" / "variations"
    if cwd_legacy.is_dir():
        return cwd_legacy
    print("ERROR: Could not locate DocSets/variations (or legacy DGS/results/variations).")
    sys.exit(1)


def find_base_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent
    if (candidate / "data").is_dir():
        return candidate
    cwd_candidate = Path.cwd() / "DGS"
    if (cwd_candidate / "data").is_dir():
        return cwd_candidate
    print("WARNING: Could not locate the DGS data directory.")
    return candidate


def main():
    variations_root = find_variations_root()
    base_dir = find_base_dir()
    safe_print(f"Scanning  : {variations_root}")
    safe_print(f"Base dir  : {base_dir}\n")

    column_descriptors = load_column_descriptors(str(base_dir))

    json_files = sorted(variations_root.rglob("*_sentence_templates.json"))
    if not json_files:
        safe_print("No variation template JSON files found.")
        return

    all_placeholder_issues: list = []
    all_misgen: list = []
    files_scanned = 0
    total_corrected = 0
    total_inserted = 0
    total_placeholder_fixed = 0

    metadata_cache: dict = {}

    for jf in json_files:
        files_scanned += 1
        path_parts = extract_path_parts(jf, variations_root)
        db = path_parts["database"]
        tbl = path_parts["table"]

        ph_issues = validate_file(jf, variations_root)
        all_placeholder_issues.extend(ph_issues)

        misgen = detect_misgeneration(jf, variations_root)
        all_misgen.extend(misgen)

        correctable = [m for m in misgen if m["issue_type"] == "misgeneration"]
        unmatched = [m for m in misgen if m["issue_type"] == "misgeneration_unmatched"]

        if not correctable and not unmatched and not ph_issues:
            continue

        null_mode, binary_mode = parse_modes_from_config(path_parts["config"])
        n_correct = len(correctable)
        n_insert = len(unmatched)
        n_ph_gaps = len(ph_issues)
        safe_print(f"\n{'=' * 80}")
        safe_print(f"  Processing {db}/{tbl}  [{path_parts['config']}]")
        safe_print(f"  null_mode={null_mode}  binary_mode={binary_mode}")
        safe_print(f"  misgenerations={n_correct}  insertable={n_insert}  placeholder_gaps={n_ph_gaps}")
        safe_print(f"{'=' * 80}")

        with open(jf, "r", encoding="utf-8") as f:
            var_data = json.load(f)
        doc_context = var_data.get("document_context", "professional data report")

        vgen = VariationBankGenerator(
            num_variations=15, null_mode=null_mode, binary_mode=binary_mode)

        data_fields = {}
        field_metadata = {}
        if correctable or unmatched:
            sent_tmpl, _narr = resolve_template_paths(jf, variations_root)
            data_fields = load_data_fields_from_sentence_template(sent_tmpl)
            if not data_fields:
                safe_print(f"  WARNING: no original_data in sentence template — skipping misgenerations")
                correctable = []
                unmatched = []
            else:
                cache_key = (db, tbl)
                if cache_key not in metadata_cache:
                    try:
                        fm, fdt = load_field_metadata(db, tbl, data_fields, base_dir)
                        metadata_cache[cache_key] = (fm, fdt)
                    except Exception as e:
                        safe_print(f"  WARNING: failed to load field metadata: {e}")
                        metadata_cache[cache_key] = ({}, {})
                field_metadata, field_data_types = metadata_cache[cache_key]

        corrections: list = []
        for issue in correctable:
            col = issue["matched_column"]
            idx = issue["template_index"]
            safe_print(f"\n  [{idx}] Correcting column: {col}")

            ft = field_metadata.get(col, "STANDARD")
            descriptor = get_column_descriptor(column_descriptors, db, tbl, col)
            safe_print(f"    field_type={ft}  descriptor={descriptor[:80]}")

            sample_value = str(data_fields.get(col, ""))
            if not sample_value:
                safe_print(f"    WARNING: no sample value for '{col}' — using column name")
                sample_value = col

            plain_sent = generate_plain_original_sentence(
                vgen, col, sample_value, descriptor, doc_context,
            )

            new_entry = correct_misgenerated_entry(
                column_name=col,
                field_type=ft,
                null_mode=null_mode,
                binary_mode=binary_mode,
                document_context=doc_context,
                descriptor=descriptor,
                data_fields=data_fields,
                vgen=vgen,
                plain_original=plain_sent,
            )
            corrections.append({"template_index": idx, "new_entry": new_entry})
            safe_print(f"    Generated {len(new_entry['variations'])} variations, "
                        f"{len(new_entry['counter_variations'])} counter, "
                        f"{len(new_entry['null_variations'])} null")
            time.sleep(0.3)

        insertions: list = []
        for issue in unmatched:
            col = issue["matched_column"]
            safe_print(f"\n  [NEW] Inserting column: {col}")

            ft = field_metadata.get(col, "STANDARD")
            descriptor = get_column_descriptor(column_descriptors, db, tbl, col)
            safe_print(f"    field_type={ft}  descriptor={descriptor[:80]}")

            sample_value = str(data_fields.get(col, ""))
            if not sample_value:
                safe_print(f"    WARNING: no sample value for '{col}' — using column name")
                sample_value = col

            plain_sent = generate_plain_original_sentence(
                vgen, col, sample_value, descriptor, doc_context,
            )

            new_entry = correct_misgenerated_entry(
                column_name=col,
                field_type=ft,
                null_mode=null_mode,
                binary_mode=binary_mode,
                document_context=doc_context,
                descriptor=descriptor,
                data_fields=data_fields,
                vgen=vgen,
                plain_original=plain_sent,
            )
            insertions.append({"new_entry": new_entry, "column": col})
            safe_print(f"    Generated {len(new_entry['variations'])} variations, "
                        f"{len(new_entry['counter_variations'])} counter, "
                        f"{len(new_entry['null_variations'])} null")
            time.sleep(0.3)

        if corrections or insertions:
            patch_variation_file(jf, corrections, insertions)
            total_corrected += len(corrections)
            total_inserted += len(insertions)
            safe_print(f"\n  Patched {len(corrections)} / Inserted {len(insertions)}"
                        f" entries in {jf.name}")
            for ins in insertions:
                for m in all_misgen:
                    if (m["issue_type"] == "misgeneration_unmatched"
                            and m["matched_column"] == ins["column"]
                            and m["database"] == db and m["table"] == tbl):
                        m["issue_type"] = "misgeneration_inserted"

        if ph_issues:
            safe_print(f"\n  Fixing {len(ph_issues)} placeholder gap(s)...")
            fixed = fix_placeholder_gaps(jf, ph_issues, vgen, doc_context)
            total_placeholder_fixed += fixed
            safe_print(f"  Fixed {fixed} variation(s) with missing placeholders")
            for issue in ph_issues:
                issue["fixed"] = True

    correctable_total = sum(1 for m in all_misgen if m["issue_type"] == "misgeneration")
    inserted_total = sum(1 for m in all_misgen if m["issue_type"] == "misgeneration_inserted")
    still_unmatched = sum(1 for m in all_misgen if m["issue_type"] == "misgeneration_unmatched")

    placeholder_fixed_count = sum(1 for p in all_placeholder_issues if p.get("fixed"))
    placeholder_remaining = len(all_placeholder_issues) - placeholder_fixed_count

    safe_print(f"\n{'=' * 90}")
    safe_print("CORRECTION SUMMARY")
    safe_print(f"{'=' * 90}")
    safe_print(f"Files scanned           : {files_scanned}")
    safe_print(f"Placeholder gaps        : {len(all_placeholder_issues)}")
    safe_print(f"  - fixed                   : {placeholder_fixed_count}")
    safe_print(f"  - remaining               : {placeholder_remaining}")
    safe_print(f"Variations fixed        : {total_placeholder_fixed}")
    safe_print(f"Misgeneration detected  : {len(all_misgen)}")
    safe_print(f"  - corrected (matched)     : {correctable_total}")
    safe_print(f"  - inserted  (unmatched)   : {inserted_total}")
    safe_print(f"  - remaining (unresolved)  : {still_unmatched}")
    safe_print(f"Entries corrected       : {total_corrected}")
    safe_print(f"Entries inserted        : {total_inserted}")
    safe_print(f"{'=' * 90}")

    unfixed_placeholder = [p for p in all_placeholder_issues if not p.get("fixed")]
    if unfixed_placeholder:
        grouped = defaultdict(list)
        for p in unfixed_placeholder:
            grouped[(p["database"], p["table"])].append(p)

        safe_print(f"\n{'=' * 90}")
        safe_print("PLACEHOLDER VALIDATION REPORT")
        safe_print(f"{'=' * 90}")
        for (database, table), issues in sorted(grouped.items()):
            safe_print(f"\n{'-' * 90}")
            safe_print(f"  Database : {database}")
            safe_print(f"  Table    : {table}")
            safe_print(f"  Issues   : {len(issues)}")
            safe_print(f"{'-' * 90}")
            for issue in issues:
                safe_print(f"\n  Field          : {issue['field_name']}")
                safe_print(f"  Classification : {issue['classification']}")
                safe_print(f"  Placeholder    : {issue['placeholder']}")
                safe_print(f"  Original       : {issue['original_sentence']}...")
                safe_print(f"  Missing        : {issue['missing_count']}/{issue['total_variations']} variations")
                for m in issue["missing_variations"]:
                    safe_print(f"    [{m['index']:>2}] {m['sentence'][:110]}...")
        safe_print(f"\n{'=' * 90}")
        safe_print(f"SUMMARY: {len(all_placeholder_issues)} field(s) across "
                    f"{len(grouped)} table(s) have placeholder gaps.")
        safe_print(f"{'=' * 90}")

    still_unresolved = [m for m in all_misgen
                        if m["issue_type"] == "misgeneration_unmatched"]
    if still_unresolved:
        ug = defaultdict(list)
        for p in still_unresolved:
            ug[(p["database"], p["table"])].append(p)

        safe_print(f"\n{'=' * 90}")
        safe_print("UNRESOLVED ORPHANED COLUMNS (no sample data — could not insert)")
        safe_print(f"{'=' * 90}")
        for (database, table), issues in sorted(ug.items()):
            cols = ", ".join(i["matched_column"] for i in issues)
            safe_print(f"  {database}/{table}: {cols}")
        safe_print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
