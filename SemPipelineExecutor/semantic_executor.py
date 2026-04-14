"""
Framework-agnostic semantic execution planner.

Transforms the flat ``semantic_pipeline`` from pipelines.json into an ordered
execution plan with multi-head routing, subquery binding, and NL instruction
generation that Lotus, DocETL, and Palimpzest backends consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Alias / table resolution
# ---------------------------------------------------------------------------

_ALIAS_RE_FROM = re.compile(
    r"\bFROM\s+(\w+)(?:\s+AS\s+(\w+))?", re.IGNORECASE
)
_ALIAS_RE_JOIN = re.compile(
    r"(?:INNER\s+|LEFT\s+|RIGHT\s+)?JOIN\s+(\w+)(?:\s+AS\s+(\w+))?",
    re.IGNORECASE,
)
_ALIAS_DOT_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\.")


def build_alias_table_map(sql: str, tables: list[str]) -> dict[str, str]:
    """Map every alias and bare table name to its canonical table from *tables*.

    Returns ``{alias_lower: table_lower, table_lower: table_lower}``.
    """
    sql_norm = re.sub(r"\s+", " ", sql).strip()
    amap: dict[str, str] = {}
    fm = _ALIAS_RE_FROM.search(sql_norm)
    if fm:
        t, a = fm.group(1).lower(), (fm.group(2) or fm.group(1)).lower()
        amap[a] = t
        amap[t] = t
    for m in _ALIAS_RE_JOIN.finditer(sql_norm):
        t, a = m.group(1).lower(), (m.group(2) or m.group(1)).lower()
        amap[a] = t
        amap[t] = t
    tables_lower = {t.lower() for t in tables}
    for t in tables_lower:
        amap.setdefault(t, t)
    return amap


def _extract_aliases_from_predicate(pred: str) -> set[str]:
    """Return the set of alias prefixes (``t1``, ``t2``, …) found in *pred*."""
    return {m.group(1).lower() for m in _ALIAS_DOT_RE.finditer(pred)}


def _split_and_at_depth0(clause: str) -> list[str]:
    """Split *clause* on ``AND`` at parenthesis depth 0, respecting BETWEEN."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    between_pending = False
    i = 0
    while i < len(clause):
        c = clause[i]
        if c == "(":
            depth += 1
            buf.append(c); i += 1; continue
        if c == ")":
            depth -= 1
            buf.append(c); i += 1; continue
        if depth == 0:
            rest = clause[i:].upper()
            if rest.startswith("BETWEEN") and (i == 0 or not clause[i - 1].isalnum()):
                after = i + 7
                if after >= len(clause) or not clause[after].isalnum():
                    between_pending = True
            if rest.startswith("AND") and (i == 0 or not clause[i - 1].isalnum()):
                after = i + 3
                if after >= len(clause) or not clause[after].isalnum():
                    if between_pending:
                        between_pending = False
                        buf.append(clause[i:i + 3]); i += 3; continue
                    else:
                        parts.append("".join(buf).strip())
                        buf = []; i += 3; continue
        buf.append(c); i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Subquery context
# ---------------------------------------------------------------------------

class SubqueryContext:
    """Stores bound subquery results keyed by ``$N``."""

    def __init__(self):
        self._bindings: dict[str, object] = {}

    def bind(self, var: str, value: object):
        self._bindings[var] = value

    def get(self, var: str, default=None):
        return self._bindings.get(var, default)

    def substitute(self, text: str) -> str:
        """Replace ``$0``, ``$1``, … in *text* with bound values."""
        out = text
        for var, val in self._bindings.items():
            out = out.replace(var, str(val))
        return out

    def __contains__(self, var: str) -> bool:
        return var in self._bindings

    def __repr__(self):
        return f"SubqueryContext({self._bindings})"


# ---------------------------------------------------------------------------
# Planned step dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlannedStep:
    """One step in the framework-agnostic execution plan."""
    kind: str               # sem_filter, sem_join, sem_agg, sem_extract, etc.
    op: dict                # original op dict from JSON
    instruction: str = ""   # NL instruction (filled by predicate_to_instruction)
    head_ids: list[int] = field(default_factory=list)   # which head(s) this targets
    merge_pair: tuple[int, int] | None = None           # for joins: (left_head, right_head)
    is_final: bool = False
    group_scoped: bool = False  # run per-cluster after sem_cluster_by
    subquery_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def _head_for_aliases(
    aliases: set[str],
    alias_map: dict[str, str],
    table_to_head: dict[str, int],
) -> int | None:
    """Return the head id for a set of aliases, or None if ambiguous/unknown."""
    head_ids: set[int] = set()
    for a in aliases:
        table = alias_map.get(a)
        if table and table in table_to_head:
            head_ids.add(table_to_head[table])
    if len(head_ids) == 1:
        return head_ids.pop()
    return None


def plan_semantic_execution(entry: dict) -> dict:
    """Build a framework-agnostic execution plan from a pipelines.json entry.

    Returns::

        {
            "subquery_plans": [  # ordered, run first
                {"pipeline": <subq dict>, "steps": [PlannedStep, …]},
                …
            ],
            "main_steps": [PlannedStep, …],
            "tables": ["t1", "t2", …],
            "alias_map": {…},
            "table_to_head": {"customers": 0, "yearmonth": 1, …},
        }
    """
    semantic = entry.get("semantic")
    if not semantic:
        return {"subquery_plans": [], "main_steps": [], "tables": [],
                "alias_map": {}, "table_to_head": {}}

    sql = entry.get("SQL", "")
    tables_str = entry.get("tables", "")
    tables = [t.strip() for t in tables_str.split(",") if t.strip()]
    alias_map = build_alias_table_map(sql, tables)

    table_to_head: dict[str, int] = {}
    for idx, t in enumerate(tables):
        table_to_head[t.lower()] = idx

    # --- subquery plans ---
    subquery_plans: list[dict] = []
    for sq in semantic.get("subquery_pipelines", []):
        steps = _plan_pipeline_ops(
            sq.get("cumulative_ops", []),
            sq.get("final_ops", []),
            alias_map, table_to_head, tables,
            is_subquery=True,
        )
        subquery_plans.append({"pipeline": sq, "steps": steps})

    # --- main pipeline ---
    main_pipe = semantic.get("main_pipeline", {})
    main_steps = _plan_pipeline_ops(
        main_pipe.get("cumulative_ops", []),
        main_pipe.get("final_ops", []),
        alias_map, table_to_head, tables,
        is_subquery=False,
    )

    return {
        "subquery_plans": subquery_plans,
        "main_steps": main_steps,
        "tables": tables,
        "alias_map": alias_map,
        "table_to_head": table_to_head,
    }


def _plan_pipeline_ops(
    cumulative_ops: list[dict],
    final_ops: list[dict],
    alias_map: dict[str, str],
    table_to_head: dict[str, int],
    tables: list[str],
    *,
    is_subquery: bool,
) -> list[PlannedStep]:
    """Convert raw op dicts into PlannedStep list with routing metadata."""
    steps: list[PlannedStep] = []
    merged_head: int | None = None  # after a join, everything targets this
    in_group_scope = False
    num_heads = len(tables)

    for op in cumulative_ops:
        kind = op.get("op_type", "")

        if kind == "subquery_result":
            steps.append(PlannedStep(
                kind=kind, op=op,
                subquery_refs=[op.get("subquery_ref", "")],
            ))
            continue

        if kind == "sem_filter":
            cond = op.get("condition", "")
            aliases = _extract_aliases_from_predicate(cond)

            if merged_head is not None or num_heads <= 1:
                target = merged_head if merged_head is not None else 0
                steps.append(PlannedStep(
                    kind=kind, op=op,
                    head_ids=[target],
                    group_scoped=in_group_scope,
                ))
            else:
                # Multi-head: route or split
                parts = _split_and_at_depth0(cond)
                if len(parts) <= 1 or not aliases:
                    target = _head_for_aliases(aliases, alias_map, table_to_head)
                    hids = [target] if target is not None else list(range(num_heads))
                    steps.append(PlannedStep(
                        kind=kind, op=op, head_ids=hids,
                        group_scoped=in_group_scope,
                    ))
                else:
                    # Split conjunctive parts by alias
                    per_head: dict[int, list[str]] = {}
                    for part in parts:
                        pa = _extract_aliases_from_predicate(part)
                        hid = _head_for_aliases(pa, alias_map, table_to_head)
                        if hid is None:
                            hid = 0
                        per_head.setdefault(hid, []).append(part)
                    for hid, preds in per_head.items():
                        merged_cond = " AND ".join(preds)
                        split_op = dict(op)
                        split_op["condition"] = merged_cond
                        steps.append(PlannedStep(
                            kind=kind, op=split_op, head_ids=[hid],
                            group_scoped=in_group_scope,
                        ))
            continue

        if kind == "sem_join":
            if num_heads >= 2 and merged_head is None:
                left_id = 0
                right_id = 1
                steps.append(PlannedStep(
                    kind=kind, op=op,
                    merge_pair=(left_id, right_id),
                    head_ids=[left_id, right_id],
                ))
                merged_head = left_id
                num_heads -= 1
            elif merged_head is not None and num_heads >= 2:
                next_id = merged_head + 1
                for hid in range(len(tables)):
                    if hid != merged_head:
                        next_id = hid
                        break
                steps.append(PlannedStep(
                    kind=kind, op=op,
                    merge_pair=(merged_head, next_id),
                    head_ids=[merged_head, next_id],
                ))
                num_heads -= 1
            else:
                steps.append(PlannedStep(
                    kind=kind, op=op,
                    head_ids=[merged_head or 0],
                ))
            continue

        if kind == "sem_cluster_by":
            target = merged_head if merged_head is not None else 0
            steps.append(PlannedStep(
                kind=kind, op=op, head_ids=[target],
            ))
            in_group_scope = True
            continue

        if kind == "sem_topk":
            target = merged_head if merged_head is not None else 0
            steps.append(PlannedStep(
                kind=kind, op=op, head_ids=[target],
                group_scoped=False,  # TopK operates on full result, not per-group
            ))
            continue

        # Fallback: sem_map, sem_dedup, etc.
        target = merged_head if merged_head is not None else 0
        steps.append(PlannedStep(
            kind=kind, op=op, head_ids=[target],
            group_scoped=in_group_scope,
        ))

    # Final ops
    for op in final_ops:
        kind = op.get("op_type", "")
        target = merged_head if merged_head is not None else 0
        steps.append(PlannedStep(
            kind=kind, op=op, head_ids=[target],
            is_final=True,
            group_scoped=in_group_scope,
        ))

    return steps


# ---------------------------------------------------------------------------
# Predicate → natural-language instruction
# ---------------------------------------------------------------------------

_CAST_RE = re.compile(r"::(?:text|integer|bigint|real|float|numeric|boolean|date|timestamp)\b", re.I)
_PAREN_OUTER_RE = re.compile(r"^\((.+)\)$", re.S)


def _clean_predicate(raw: str) -> str:
    """Strip Postgres casts, outer parens, excess whitespace."""
    text = _CAST_RE.sub("", raw)
    text = re.sub(r"\s+", " ", text).strip()
    m = _PAREN_OUTER_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text


def _humanize_column(ref: str) -> str:
    """``t1.customer_id`` → ``customer id``."""
    parts = ref.rsplit(".", 1)
    col = parts[-1] if parts else ref
    return col.replace("_", " ").strip()


def predicate_to_instruction(
    op: dict,
    entry: dict,
    ctx: SubqueryContext | None = None,
    llm_client=None,
) -> str:
    """Convert a semantic op dict into a human-readable NL instruction.

    Joins **always** use the legacy template (no LLM). All other operation
    types use the LLM when *llm_client* is provided, falling back to
    heuristic templates otherwise.
    """
    kind = op.get("op_type", "")
    question = entry.get("question", "")
    evidence = entry.get("evidence", "")

    # --- Joins: always legacy translation, never LLM ---
    if kind == "sem_join":
        jc = op.get("join_condition", "")
        clean = _clean_predicate(jc)
        parts = re.findall(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", clean)
        if parts:
            _, lc, _, rc = parts[0]
            return (f"The {_humanize_column(lc)} described in one document is the "
                    f"same as the {_humanize_column(rc)} described in the other document.")
        return f"Documents share a matching key: {clean}" if clean else "Documents share a matching key."

    # --- Metadata steps ---
    if kind == "subquery_result":
        ref = op.get("subquery_ref", "")
        return f"Use subquery result {ref}"

    if kind == "sem_cluster_by":
        keys = op.get("group_keys", [])
        cols = ", ".join(_humanize_column(k) for k in keys)
        return f"Group by {cols}" if cols else "Group the documents"

    # --- All other ops: prefer LLM, fall back to templates ---
    if kind == "sem_filter":
        cond = op.get("condition", "")
        clean = _clean_predicate(cond)
        if ctx:
            clean = ctx.substitute(clean)
        if llm_client:
            return _llm_translate_predicate(clean, question, evidence, "FILTER", llm_client)
        return _template_filter(clean)

    if kind == "sem_agg":
        expr = op.get("agg_expression", "")
        clean = _clean_predicate(expr)
        if ctx:
            clean = ctx.substitute(clean)
        if llm_client:
            return _llm_translate_predicate(clean, question, evidence, "AGGREGATE", llm_client)
        return _template_agg(clean)

    if kind == "sem_extract":
        col = op.get("extract_column", "")
        clean = _clean_predicate(col)
        if llm_client:
            return _llm_translate_extract(clean, question, evidence, llm_client)
        return f"The {_humanize_column(col)}"

    if kind == "sem_topk":
        sort_keys = op.get("sort_keys", [])
        limit = op.get("limit_count", 1)
        raw = ""
        if sort_keys:
            raw = _clean_predicate(sort_keys[0])
        if llm_client and raw:
            return _llm_translate_predicate(
                f"TOP {limit} ORDER BY {raw}", question, evidence, "RANK", llm_client)
        if sort_keys:
            key_text = raw
            direction = "ascending" if "ASC" in key_text.upper() else "descending"
            col = re.sub(r"\s+(ASC|DESC).*$", "", key_text, flags=re.I).strip()
            col_clean = _humanize_column(col)
            if limit == 1:
                word = "lowest" if direction == "ascending" else "highest"
                return f"The one with {word} {col_clean}"
            return f"Top {limit} by {col_clean} {direction}"
        return f"Top {limit}"

    if kind == "sem_map":
        expr = op.get("expression", "")
        clean = _clean_predicate(expr)
        if llm_client:
            return _llm_translate_predicate(clean, question, evidence, "MAP", llm_client)
        return clean

    return _clean_predicate(str(op))


def _template_filter(clean: str) -> str:
    """Build a heuristic NL filter instruction from cleaned SQL."""
    m = re.match(r"(\w[\w .]*?)\s*(=|!=|<>|>=|<=|>|<|LIKE|NOT LIKE|IN|NOT IN)\s*(.+)", clean, re.I)
    if m:
        col, oper, val = m.group(1).strip(), m.group(2).upper(), m.group(3).strip().strip("'\"")
        col_h = _humanize_column(col)
        op_text = {
            "=": "is", "!=": "is not", "<>": "is not",
            ">": "is greater than", "<": "is less than",
            ">=": "is at least", "<=": "is at most",
            "LIKE": "contains", "NOT LIKE": "does not contain",
            "IN": "is one of", "NOT IN": "is not one of",
        }.get(oper, oper)
        return f"The {col_h} {op_text} {val}"
    return clean


def _template_agg(clean: str) -> str:
    m = re.match(r"(COUNT|SUM|AVG|MAX|MIN)\s*\((.+)\)", clean, re.I)
    if m:
        func, inner = m.group(1).capitalize(), m.group(2).strip()
        col_h = _humanize_column(inner)
        if func == "Count":
            return f"The total count of {col_h}"
        return f"The {func.lower()} of {col_h}"
    return clean


_LLM_TRANSLATE_CACHE: dict[tuple, str] = {}

# ---------------------------------------------------------------------------
# System prompt: predicate-focused translation
# ---------------------------------------------------------------------------
_FEW_SHOT_SYSTEM = """\
You translate a single SQL predicate into a clear COMMAND or QUESTION that \
tells an LLM exactly what to do. The Question is the *overall* question the \
pipeline solves — use it only for context (readable column names, domain). \
Your job is to translate the PREDICATE ITSELF into an actionable instruction.

RULES:
1. Remove ALL SQL artifacts: casts (::text, CAST … AS …), aliases (T1., T2.), \
   NULLIF, unbalanced parens.
2. Translate EXACTLY what the predicate computes or checks — no more, no less. \
   Do not add conditions or details that are not in the predicate.
3. CRITICAL: Preserve ALL arithmetic operations (division, multiplication, etc.). \
   If the predicate divides by 12, say "divide by 12". If it multiplies by 100, \
   say "multiply by 100". These details are essential for correct computation.
4. For FILTER predicates: phrase as a condition check. \
   (e.g. "Does the segment equal SME?", "Is the account from the year 1995?")
5. For AGGREGATE predicates: phrase as an imperative command starting with \
   "Calculate", "Compute", "Count", "Find", or as a question "What is...?". \
   If the predicate is a standalone COUNT(column), phrase as "Count how many …" \
   or "How many … are there?". For complex aggregates (CASE/WHEN, ratios, \
   differences), describe the arithmetic faithfully including all divisions.
6. For RANK / TOP-K predicates: phrase as "Find the top N by..." or \
   "Which one has the highest/lowest...?"
7. For MAP predicates: phrase as "Extract..." or "Compute..."
8. ONE sentence. No SQL, no code. Output ONLY the instruction text."""

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            'Question: "What is the difference in the annual average consumption '
            'between SME and LAM segments?"\n'
            'Evidence: ""\n'
            'Predicate: "CAST(SUM(CASE WHEN T1.Segment = \'SME\' THEN T2.Consumption '
            "ELSE 0 END) AS REAL) / NULLIF(COUNT(T1.CustomerID), 0) - "
            "CAST(SUM(CASE WHEN T1.Segment = 'LAM' THEN T2.Consumption "
            'ELSE 0 END) AS REAL) / NULLIF(COUNT(T1.CustomerID), 0)"'
        ),
    },
    {
        "role": "assistant",
        "content": "Calculate the difference between the average consumption of SME customers and the average consumption of LAM customers.",
    },
    {
        "role": "user",
        "content": (
            'Question: "What is the age of the patient?"\n'
            'Evidence: "age can be computed from birthday"\n'
            "Predicate: \"EXTRACT(YEAR FROM CURRENT_TIMESTAMP) - EXTRACT(YEAR FROM T1.Birthday)\""
        ),
    },
    {
        "role": "assistant",
        "content": "Compute the age of the patient from their birthday.",
    },
    {
        "role": "user",
        "content": (
            'Question: "What is the ratio of male to female patients with abnormal uric acid?"\n'
            'Evidence: "normal uric acid for male <=8.0; for female <=6.5"\n'
            "Predicate: \"CAST(SUM(CASE WHEN T2.UA <= 8.0 AND T1.SEX = 'M' THEN 1 "
            "ELSE 0 END) AS REAL) / NULLIF(SUM(CASE WHEN T2.UA <= 6.5 AND T1.SEX = 'F' "
            'THEN 1 ELSE 0 END), 0)"'
        ),
    },
    {
        "role": "assistant",
        "content": "Calculate the ratio of male patients with uric acid at most 8.0 to female patients with uric acid at most 6.5.",
    },
    {
        "role": "user",
        "content": (
            'Question: "How many students are female?"\n'
            'Evidence: ""\n'
            'Predicate: "COUNT(T1.member_id)"'
        ),
    },
    {
        "role": "assistant",
        "content": "Count how many members there are.",
    },
    {
        "role": "user",
        "content": (
            'Question: "What is the name of the youngest driver?"\n'
            'Evidence: ""\n'
            'Predicate: "TOP 1 ORDER BY T1.dob DESC"'
        ),
    },
    {
        "role": "assistant",
        "content": "Find the one with the most recent date of birth.",
    },
    {
        "role": "user",
        "content": (
            'Question: "Among the accounts opened in 1995, what is the ratio of female '
            'to male account holders?"\n'
            'Evidence: "opened in 1995 refers to year of A_DATE = 1995"\n'
            "Predicate: \"(t1.a_date >= '1995-01-01') AND (t1.a_date < '1996-01-01')\""
        ),
    },
    {
        "role": "assistant",
        "content": "Check if the account was opened in the year 1995.",
    },
    {
        "role": "user",
        "content": (
            'Question: "What is the average monthly consumption of SME customers?"\n'
            'Evidence: ""\n'
            'Predicate: "AVG(T2.Consumption) / 12"'
        ),
    },
    {
        "role": "assistant",
        "content": "Calculate the average consumption and divide by 12 to get the monthly average.",
    },
    {
        "role": "user",
        "content": (
            'Question: "What percentage of transactions were successful?"\n'
            'Evidence: ""\n'
            'Predicate: "CAST(SUM(CASE WHEN status = \'success\' THEN 1 ELSE 0 END) AS REAL) * 100.0 / COUNT(*)"'
        ),
    },
    {
        "role": "assistant",
        "content": "Calculate the count of successful transactions divided by the total count, then multiply by 100 to get the percentage.",
    },
]

# ---------------------------------------------------------------------------
# Extract-specific prompt: simple column description
# ---------------------------------------------------------------------------
_EXTRACT_SYSTEM = """\
You translate a SQL column expression into a short COMMAND that tells an LLM \
what value to extract from a document. Use the Question only for context. \
Do NOT restate the full question.

RULES:
1. Remove SQL artifacts: aliases (T1., T2.), casts, SUBSTR, EXTRACT wrappers.
2. Start with "Extract" and describe WHAT to extract. \
   Examples: "Extract the first name of the patient", \
   "Extract the year portion of the transaction date".
3. If the column is simple (e.g. T1.CustomerID) just say "Extract the customer ID".
4. ONE short phrase. No SQL, no code. Output ONLY the instruction."""

_EXTRACT_EXAMPLES = [
    {"role": "user", "content": 'Question: "What is the first name of the student?"\nColumn: "T1.first_name"'},
    {"role": "assistant", "content": "Extract the first name of the student."},
    {"role": "user", "content": 'Question: "What year were the transactions made?"\nColumn: "SUBSTR(T2.Date, 1, 4)"'},
    {"role": "assistant", "content": "Extract the year portion of the date."},
    {"role": "user", "content": 'Question: "What is the customer ID?"\nColumn: "T2.CustomerID"'},
    {"role": "assistant", "content": "Extract the customer ID."},
    {"role": "user", "content": 'Question: "What country is the account holder from?"\nColumn: "T2.Country"'},
    {"role": "assistant", "content": "Extract the country."},
]


def _llm_translate_predicate(
    cleaned: str, question: str, evidence: str, op_type: str, client,
) -> str:
    cache_key = (cleaned, question[:80], op_type)
    if cache_key in _LLM_TRANSLATE_CACHE:
        return _LLM_TRANSLATE_CACHE[cache_key]

    user_msg = (
        f'Question: "{question}"\n'
        f'Evidence: "{evidence}"\n'
        f'Predicate: "{cleaned}"'
    )

    messages = [
        {"role": "system", "content": _FEW_SHOT_SYSTEM},
        *_FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=120,
            temperature=0.1,
        )
        result = resp.choices[0].message.content.strip().strip('"').strip("'")
        _LLM_TRANSLATE_CACHE[cache_key] = result
        return result
    except Exception:
        return cleaned


def _llm_translate_extract(
    cleaned_col: str, question: str, evidence: str, client,
) -> str:
    """Dedicated LLM call for extract ops — produces a short noun phrase."""
    cache_key = (cleaned_col, question[:80], "EXTRACT")
    if cache_key in _LLM_TRANSLATE_CACHE:
        return _LLM_TRANSLATE_CACHE[cache_key]

    user_msg = (
        f'Question: "{question}"\n'
        f'Column: "{cleaned_col}"'
    )
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        *_EXTRACT_EXAMPLES,
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=60,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip().strip('"').strip("'")
        _LLM_TRANSLATE_CACHE[cache_key] = result
        return result
    except Exception:
        return f"The {_humanize_column(cleaned_col)}"


# ---------------------------------------------------------------------------
# Convenience: enrich all steps in a plan with NL instructions
# ---------------------------------------------------------------------------

def enrich_plan_instructions(
    plan: dict,
    entry: dict,
    ctx: SubqueryContext | None = None,
    llm_client=None,
):
    """Fill ``step.instruction`` for every PlannedStep in *plan* (in-place).

    Prefers cached ``semantic_nl`` from the entry when available. Falls back
    to ``predicate_to_instruction`` (heuristic or LLM) otherwise.
    """
    cached = _build_instruction_cache(entry)
    op_idx = 0

    for sq_plan in plan.get("subquery_plans", []):
        for step in sq_plan.get("steps", []):
            if not step.instruction:
                instr = cached.get(op_idx)
                if instr:
                    step.instruction = instr
                else:
                    step.instruction = predicate_to_instruction(
                        step.op, entry, ctx, llm_client)
            if ctx:
                step.instruction = ctx.substitute(step.instruction)
            op_idx += 1

    for step in plan.get("main_steps", []):
        if not step.instruction:
            instr = cached.get(op_idx)
            if instr:
                step.instruction = instr
            else:
                step.instruction = predicate_to_instruction(
                    step.op, entry, ctx, llm_client)
        if ctx:
            step.instruction = ctx.substitute(step.instruction)
        op_idx += 1


def _build_instruction_cache(entry: dict) -> dict[int, str]:
    """Build op_index -> instruction mapping from cached semantic_nl."""
    nl = entry.get("semantic_nl")
    if not nl:
        return {}
    return {row["op_index"]: row["instruction"] for row in nl}
