"""
Parse mini_dev_sqlite_gold.sql and generate SQLite EXPLAIN QUERY PLAN
execution trees for each query.

Each line in the gold SQL file is:   <sql_query> TAB <db_id>
Databases live at:  MINIDEV/dev_databases/<db_id>/<db_id>.sqlite

Outputs (under SemPipelineBuilder):
  pipelines/explain_trees.json
  pipelines/explain_trees_summary.txt
"""

import json
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
GOLD_SQL_FILE = REPO_ROOT / "MINIDEV" / "mini_dev_sqlite_gold.sql"
DB_DIR = REPO_ROOT / "MINIDEV" / "dev_databases"
OUTPUT_DIR = ROOT_DIR / "pipelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_gold_sql(path: Path) -> list[dict]:
    """Return [{line_no, db_id, query}, ...] from the tab-separated file."""
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split("\t")
            if len(parts) < 2:
                print(f"[WARN] Line {line_no}: unexpected format, skipping.")
                continue
            entries.append({
                "line_no": line_no,
                "query":   parts[0].strip(),
                "db_id":   parts[1].strip(),
            })
    return entries


def get_query_plan(conn: sqlite3.Connection, query: str) -> dict:
    """
    Run EXPLAIN QUERY PLAN on *query*.

    SQLite returns rows of (id, parent_id, notused, detail).
    We convert these into:
      - plan_rows : raw list of {id, parent_id, detail} dicts
      - plan_tree : indented text tree built from parent_id relationships
      - error     : non-None string if the call failed
    """
    try:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
    except sqlite3.Error as exc:
        return {"plan_rows": None, "plan_tree": None, "error": str(exc)}

    plan_rows = [
        {"id": r[0], "parent_id": r[1], "detail": r[3]}
        for r in rows
    ]
    plan_tree = _build_tree(plan_rows)
    return {"plan_rows": plan_rows, "plan_tree": plan_tree, "error": None}


def _build_tree(plan_rows: list[dict]) -> str:
    """Render plan_rows as an indented tree string using parent_id links."""
    if not plan_rows:
        return "(empty plan)"

    # Map id → row for fast lookup
    by_id = {r["id"]: r for r in plan_rows}

    # Find depth of each node by walking up through parent_ids.
    # Root nodes have parent_id == 0 (SQLite convention).
    def depth(node: dict) -> int:
        d, pid = 0, node["parent_id"]
        while pid != 0 and pid in by_id:
            d  += 1
            pid = by_id[pid]["parent_id"]
        return d

    lines = []
    for row in plan_rows:
        indent = "    " * depth(row)
        lines.append(f"{indent}--  {row['detail']}")
    return "\n".join(lines)


def main() -> None:
    entries = parse_gold_sql(GOLD_SQL_FILE)
    print(f"Parsed {len(entries)} queries from {GOLD_SQL_FILE.name}")

    # Group by db so we open each SQLite file only once.
    by_db: dict[str, list[dict]] = {}
    for entry in entries:
        by_db.setdefault(entry["db_id"], []).append(entry)

    all_results: list[dict] = []
    stats = {"total": len(entries), "success": 0, "failed": 0}

    for db_id, db_entries in by_db.items():
        sqlite_path = DB_DIR / db_id / f"{db_id}.sqlite"

        if not sqlite_path.exists():
            print(f"\n[ERROR] Database not found: {sqlite_path}")
            for entry in db_entries:
                all_results.append({**entry, "error": "SQLite file not found",
                                    "plan_rows": None, "plan_tree": None})
                stats["failed"] += 1
            continue

        print(f"\n--- {db_id}  ({len(db_entries)} queries) ---")

        conn = sqlite3.connect(str(sqlite_path))
        try:
            for entry in db_entries:
                result = get_query_plan(conn, entry["query"])
                all_results.append({
                    "line_no": entry["line_no"],
                    "db_id":   db_id,
                    "sql":     entry["query"],
                    **result,
                })
                if result["error"]:
                    stats["failed"] += 1
                    print(f"  [FAIL] line {entry['line_no']}: {result['error']}")
                else:
                    stats["success"] += 1
                    print(f"  [OK]   line {entry['line_no']}")
        finally:
            conn.close()

    # ── JSON output ────────────────────────────────────────────────────────────
    json_out = OUTPUT_DIR / "explain_trees.json"
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON  -> {json_out}")

    # ── Text summary ───────────────────────────────────────────────────────────
    txt_out = OUTPUT_DIR / "explain_trees_summary.txt"
    with open(txt_out, "w", encoding="utf-8") as fh:
        for row in all_results:
            fh.write("=" * 80 + "\n")
            fh.write(f"Line : {row['line_no']}\n")
            fh.write(f"DB   : {row['db_id']}\n")
            fh.write(f"SQL  : {row['sql']}\n")
            if row["error"]:
                fh.write(f"ERROR: {row['error']}\n")
            else:
                fh.write("\nExecution Tree (EXPLAIN QUERY PLAN):\n")
                fh.write(row["plan_tree"])
                fh.write("\n")
            fh.write("\n")
    print(f"Text  -> {txt_out}")

    print(f"\nDone. {stats['success']}/{stats['total']} succeeded "
          f"({stats['failed']} failed).")


if __name__ == "__main__":
    main()
