"""
PostgreSQL Explain Pipeline

Default output directory: ``SemPipelineBuilder/pipelines/``
(``explain_trees_postgres.json``, ``explain_trees_postgres_summary.txt``).

End-to-end pipeline that:
1. Spins up PostgreSQL in Docker
2. Loads database from PostgreSQL dump (MINIDEV_postgresql) OR migrates from SQLite
3. Runs EXPLAIN (FORMAT JSON) on gold queries
4. Parses execution trees into physical-operator DAGs
5. Stores results for downstream semantic validation

Usage:
    # Load from MINIDEV_postgresql SQL dump (default):
    python postgres_explain_pipeline.py [--start-docker] [--stop-docker]

    # Migrate from SQLite instead:
    python postgres_explain_pipeline.py --sqlite-source [--start-docker] [--stop-docker]

    # Skip loading (assume database already exists):
    python postgres_explain_pipeline.py --skip-migration
"""

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psycopg2

from explain_parser import (
    format_aggregate_info,
    format_plan_tree,
    format_semantic_pipeline,
    get_operator_summary,
    parse_explain_json,
)


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
GOLD_SQL_FILE = REPO_ROOT / "MINIDEV" / "mini_dev_postgresql_gold.sql"
DB_DIR = REPO_ROOT / "MINIDEV" / "dev_databases"
MINIDEV_POSTGRESQL_DIR = REPO_ROOT / "MINIDEV_postgresql"
DEFAULT_SQL_DUMP = MINIDEV_POSTGRESQL_DIR / "BIRD_dev.sql"
# Pipeline artifacts live under SemPipelineBuilder/pipelines/ (not repo-root pipeline_data/)
OUTPUT_DIR = ROOT_DIR / "pipelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"

DOCKER_COMPOSE_FILE = ROOT_DIR / "docker-compose.yml"


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def wait_for_postgres(
    host: str = PG_HOST,
    port: int = PG_PORT,
    user: str = PG_USER,
    password: str = PG_PASSWORD,
    timeout: int = 60,
    interval: float = 2.0
) -> bool:
    """Wait for PostgreSQL to become ready."""
    print(f"Waiting for PostgreSQL at {host}:{port}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if is_port_open(host, port):
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname="postgres",
                    connect_timeout=5
                )
                conn.close()
                print("PostgreSQL is ready!")
                return True
            except psycopg2.Error:
                pass
        
        time.sleep(interval)
    
    print(f"Timeout waiting for PostgreSQL after {timeout}s")
    return False


def start_docker_postgres() -> bool:
    """Start PostgreSQL using docker-compose. Removes existing container first for a fresh start."""
    if not DOCKER_COMPOSE_FILE.exists():
        print(f"docker-compose.yml not found at {DOCKER_COMPOSE_FILE}")
        return False

    def run_compose(args: list[str]) -> tuple[bool, str]:
        last_err = ""
        for cmd in [["docker-compose", "-f", str(DOCKER_COMPOSE_FILE)] + args, ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE)] + args]:
            try:
                r = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
                if r.returncode != 0:
                    last_err = r.stderr or r.stdout or f"exit code {r.returncode}"
                    continue
                return True, ""
            except FileNotFoundError:
                last_err = f"Command not found: {cmd[0]}"
                continue
        return False, last_err

    print("Removing existing container (if any)...")
    subprocess.run(
        ["docker", "rm", "-f", "sdps_postgres"],
        capture_output=True,
    )
    run_compose(["down", "-v"])

    print("Starting PostgreSQL container...")
    ok, err = run_compose(["up", "-d"])
    if not ok:
        print(f"Failed to start Docker container: {err}")
        return False
    print("Docker container started.")
    return True


def load_postgresql_dump(
    sql_file: Path,
    pg_host: str = PG_HOST,
    pg_port: int = PG_PORT,
    pg_user: str = PG_USER,
    pg_password: str = PG_PASSWORD,
    db_name: str = "bird_dev",
    drop_existing: bool = True,
) -> bool:
    """
    Load a PostgreSQL dump (e.g., BIRD_dev.sql) into a database.
    
    Uses psql for efficient loading of large SQL files.
    """
    if not sql_file.exists():
        print(f"[ERROR] SQL dump not found: {sql_file}")
        return False
    
    env = os.environ.copy()
    env["PGPASSWORD"] = pg_password
    
    try:
        admin_conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            dbname="postgres"
        )
        admin_conn.autocommit = True
        cursor = admin_conn.cursor()
        
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone() is not None
        
        if exists and drop_existing:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,)
            )
            cursor.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            print(f"  Dropped existing database: {db_name}")
        
        if not exists or drop_existing:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"  Created database: {db_name}")
        
        admin_conn.close()
        
        print(f"  Loading SQL dump: {sql_file.name} (this may take a while for large files)...")
        
        result = subprocess.run(
            [
                "psql",
                "-h", pg_host,
                "-p", str(pg_port),
                "-U", pg_user,
                "-d", db_name,
                "-f", str(sql_file),
                "-v", "ON_ERROR_STOP=0",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"  [WARN] psql reported errors (e.g. missing roles); schema/data may still have loaded: {result.stderr[:300]}")
        
        print(f"  Loaded {sql_file.name} (schema and data)")
        return True
        
    except psycopg2.Error as e:
        print(f"[ERROR] Database error: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] psql not found. Ensure PostgreSQL client tools are installed.")
        return False


def stop_docker_postgres() -> bool:
    """Stop PostgreSQL using docker-compose."""
    if not DOCKER_COMPOSE_FILE.exists():
        print(f"docker-compose.yml not found at {DOCKER_COMPOSE_FILE}")
        return False
    
    print("Stopping PostgreSQL container...")
    try:
        subprocess.run(
            ["docker-compose", "-f", str(DOCKER_COMPOSE_FILE), "down"],
            cwd=str(ROOT_DIR),
            check=True,
            capture_output=True,
            text=True
        )
        print("Docker container stopped.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop Docker container: {e.stderr}")
        return False
    except FileNotFoundError:
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "down"],
                cwd=str(ROOT_DIR),
                check=True,
                capture_output=True,
                text=True
            )
            print("Docker container stopped.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e2:
            print(f"Failed to stop Docker container: {e2}")
            return False


def _sqlite_type_to_pg(stype: str) -> str:
    """Map SQLite type to PostgreSQL type."""
    s = (stype or "").upper()
    if "INT" in s:
        return "BIGINT"
    if "REAL" in s or "FLOAT" in s or "DOUB" in s:
        return "DOUBLE PRECISION"
    if "BLOB" in s:
        return "BYTEA"
    return "TEXT"


def migrate_all_databases(
    db_dir: Path,
    pg_host: str,
    pg_port: int,
    pg_user: str,
    pg_password: str,
    drop_existing: bool = True,
) -> dict:
    """Migrate SQLite databases from db_dir to PostgreSQL. Returns {db_id: {success, tables_migrated, rows_migrated}}."""
    results = {}
    if not db_dir.exists():
        return results
    for subdir in sorted(db_dir.iterdir()):
        if not subdir.is_dir():
            continue
        db_id = subdir.name
        sqlite_path = subdir / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            continue
        stats = {"success": False, "tables_migrated": 0, "rows_migrated": 0}
        try:
            admin_conn = psycopg2.connect(
                host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname="postgres"
            )
            admin_conn.autocommit = True
            cur = admin_conn.cursor()
            cur.execute(f'DROP DATABASE IF EXISTS "{db_id}"')
            cur.execute(f'CREATE DATABASE "{db_id}"')
            cur.close()
            admin_conn.close()
            pg_conn = psycopg2.connect(
                host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=db_id
            )
            sqlite_conn = sqlite3.connect(str(sqlite_path))
            for row in sqlite_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ):
                table = row[0]
                try:
                    schema = sqlite_conn.execute(f"PRAGMA table_info({table})").fetchall()
                    col_defs = [f'"{c[1]}" {_sqlite_type_to_pg(c[2])}' for c in schema]
                    create_sql = f'CREATE TABLE "{table}" (' + ", ".join(col_defs) + ")"
                    pg_cur = pg_conn.cursor()
                    pg_cur.execute(create_sql)
                    data = sqlite_conn.execute(f'SELECT * FROM "{table}"').fetchall()
                    if data:
                        cols = [f'"{c[1]}"' for c in schema]
                        placeholders = ",".join(["%s"] * len(schema))
                        insert_sql = f'INSERT INTO "{table}" ({",".join(cols)}) VALUES ({placeholders})'
                        for r in data:
                            pg_cur.execute(insert_sql, r)
                        stats["rows_migrated"] += len(data)
                    stats["tables_migrated"] += 1
                except Exception as e:
                    print(f"    [WARN] Table {table}: {e}")
            sqlite_conn.close()
            pg_conn.commit()
            pg_conn.close()
            stats["success"] = True
        except Exception as e:
            print(f"  [ERROR] {db_id}: {e}")
        results[db_id] = stats
    return results


def run_analyze(
    db_names: list[str],
    pg_host: str = PG_HOST,
    pg_port: int = PG_PORT,
    pg_user: str = PG_USER,
    pg_password: str = PG_PASSWORD,
) -> None:
    """Run ANALYZE on all tables in each database so plans use real statistics."""
    for db_name in db_names:
        try:
            conn = psycopg2.connect(
                host=pg_host,
                port=pg_port,
                user=pg_user,
                password=pg_password,
                dbname=db_name,
            )
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("ANALYZE")
            cur.close()
            conn.close()
            print(f"  ANALYZE {db_name}: done")
        except psycopg2.Error as e:
            print(f"  [WARN] ANALYZE {db_name}: {e}")


def configure_postgres_for_logical_plan(cur) -> None:
    """Reduce plan noise: eliminate parallelism, hash joins, bitmap scans, materialize, memoize, join reordering."""
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET enable_hashjoin = OFF")
    cur.execute("SET enable_memoize = OFF")
    cur.execute("SET enable_bitmapscan = OFF")
    cur.execute("SET enable_material = OFF")
    cur.execute("SET join_collapse_limit = 1")
    cur.execute("SET from_collapse_limit = 1")


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
                "query": parts[0].strip(),
                "db_id": parts[1].strip(),
            })
    return entries


def get_explain_json(
    conn,
    query: str
) -> tuple[Optional[list], Optional[str]]:
    """
    Run EXPLAIN (FORMAT JSON, VERBOSE) on a query.
    VERBOSE adds Output/expressions per node (e.g. sum(x), avg(y) for Aggregate).
    
    Returns:
        (explain_json, error) - explain_json is the parsed JSON, error is None on success
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN (FORMAT JSON, VERBOSE) {query}")
        result = cursor.fetchone()
        cursor.close()
        
        if result and result[0]:
            return result[0], None
        return None, "Empty EXPLAIN result"
    
    except psycopg2.Error as e:
        return None, str(e)


def run_explain_pipeline(
    entries: list[dict],
    pg_host: str = PG_HOST,
    pg_port: int = PG_PORT,
    pg_user: str = PG_USER,
    pg_password: str = PG_PASSWORD,
    single_db_name: Optional[str] = None,
) -> list[dict]:
    """
    Run EXPLAIN on all queries and parse execution plans.
    
    Args:
        entries: List of {line_no, db_id, query} dicts
        pg_*: PostgreSQL connection parameters
        single_db_name: If set, run all queries against this database (e.g., when
            loading from BIRD_dev.sql which has all tables in one DB)
    
    Returns:
        List of result dicts with execution plan DAGs
    """
    results = []
    by_db: dict[str, list[dict]] = {}
    for entry in entries:
        by_db.setdefault(entry["db_id"], []).append(entry)
    
    stats = {"total": len(entries), "success": 0, "failed": 0, "db_missing": 0}
    connections: dict[str, Optional[psycopg2.extensions.connection]] = {}
    
    if single_db_name:
        db_names_to_connect = {single_db_name}
        db_id_to_conn = single_db_name
    else:
        db_names_to_connect = set(by_db.keys())
        db_id_to_conn = None
    
    for db_name in db_names_to_connect:
        try:
            conn = psycopg2.connect(
                host=pg_host,
                port=pg_port,
                user=pg_user,
                password=pg_password,
                dbname=db_name
            )
            cur = conn.cursor()
            configure_postgres_for_logical_plan(cur)
            cur.close()
            if single_db_name:
                for db_id in by_db.keys():
                    connections[db_id] = conn
            else:
                connections[db_name] = conn
            print(f"Connected to database: {db_name}")
        except psycopg2.Error as e:
            if single_db_name:
                for db_id in by_db.keys():
                    connections[db_id] = None
            else:
                connections[db_name] = None
            print(f"[ERROR] Cannot connect to database {db_name}: {e}")
    
    for db_id, db_entries in by_db.items():
        conn = connections.get(db_id)
        
        if conn is None:
            for entry in db_entries:
                results.append({
                    "line_no": entry["line_no"],
                    "db_id": db_id,
                    "sql": entry["query"],
                    "explain_json": None,
                    "plan_dag": None,
                    "semantic_sequence": None,
                    "error": f"Database not found: {db_id}",
                })
                stats["db_missing"] += 1
            continue
        
        print(f"\n--- {db_id} ({len(db_entries)} queries) ---")
        
        for entry in db_entries:
            explain_json, error = get_explain_json(conn, entry["query"])
            
            if error:
                results.append({
                    "line_no": entry["line_no"],
                    "db_id": db_id,
                    "sql": entry["query"],
                    "explain_json": None,
                    "plan_dag": None,
                    "semantic_pipeline": None,
                    "error": error,
                })
                stats["failed"] += 1
                print(f"  [FAIL] line {entry['line_no']}: {error[:80]}...")
            else:
                plan_dag = parse_explain_json(explain_json, entry["query"])
                semantic_pipeline = plan_dag.get("semantic_pipeline")
                results.append({
                    "line_no": entry["line_no"],
                    "db_id": db_id,
                    "sql": entry["query"],
                    "explain_json": explain_json,
                    "plan_dag": plan_dag,
                    "semantic_pipeline": semantic_pipeline,
                    "error": None,
                })
                stats["success"] += 1
                print(f"  [OK]   line {entry['line_no']} - {len(plan_dag['nodes'])} operators")
    
    for conn in connections.values():
        if conn:
            conn.close()
    
    print(f"\n{'='*60}")
    print(f"Results: {stats['success']}/{stats['total']} succeeded")
    print(f"  - Failed: {stats['failed']}")
    print(f"  - DB missing: {stats['db_missing']}")
    
    return results


def write_json_output(results: list[dict], output_path: Path) -> None:
    """Write results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"JSON output -> {output_path}")


def write_summary_output(results: list[dict], output_path: Path) -> None:
    """Write human-readable summary to text file."""
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write("=" * 80 + "\n")
            fh.write(f"Line : {row['line_no']}\n")
            fh.write(f"DB   : {row['db_id']}\n")
            fh.write(f"SQL  : {row['sql']}\n")
            
            if row["error"]:
                fh.write(f"ERROR: {row['error']}\n")
            else:
                fh.write("\nExecution Plan (PostgreSQL):\n")
                if row["plan_dag"]:
                    fh.write(format_plan_tree(row["plan_dag"]))
                    fh.write("\n\n")
                    
                    fh.write(f"Operator Sequence: {row['plan_dag']['operator_sequence']}\n")

                    # Write semantic pipeline
                    sem_pipeline = row.get("semantic_pipeline")
                    if sem_pipeline:
                        fh.write(f"\nSemantic Pipeline:\n")
                        fh.write(format_semantic_pipeline(sem_pipeline))
                        fh.write("\n")

                    summary = get_operator_summary(row["plan_dag"])
                    fh.write(f"Total Operators: {summary['total_operators']}\n")
                    fh.write(f"Operator Counts: {summary['operator_counts']}\n")
                else:
                    fh.write("(no plan available)\n")
            
            fh.write("\n")
    
    print(f"Summary output -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PostgreSQL Explain Pipeline for MINIDEV benchmark"
    )
    parser.add_argument(
        "--start-docker",
        action="store_true",
        help="Start PostgreSQL Docker container before running"
    )
    parser.add_argument(
        "--stop-docker",
        action="store_true",
        help="Stop PostgreSQL Docker container after running"
    )
    parser.add_argument(
        "--sql-source",
        type=Path,
        default=DEFAULT_SQL_DUMP,
        help=f"Path to PostgreSQL dump SQL file (default: {DEFAULT_SQL_DUMP}). "
             "When set, loads from this file instead of migrating from SQLite."
    )
    parser.add_argument(
        "--sqlite-source",
        action="store_true",
        help="Use SQLite migration instead of PostgreSQL dump (from MINIDEV/dev_databases)"
    )
    parser.add_argument(
        "--skip-migration",
        action="store_true",
        help="Skip database loading (assume database already exists)"
    )
    parser.add_argument(
        "--host",
        default=PG_HOST,
        help=f"PostgreSQL host (default: {PG_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PG_PORT,
        help=f"PostgreSQL port (default: {PG_PORT})"
    )
    parser.add_argument(
        "--user",
        default=PG_USER,
        help=f"PostgreSQL user (default: {PG_USER})"
    )
    parser.add_argument(
        "--password",
        default=PG_PASSWORD,
        help=f"PostgreSQL password (default: {PG_PASSWORD})"
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=GOLD_SQL_FILE,
        help=f"Path to gold SQL file (default: {GOLD_SQL_FILE})"
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DB_DIR,
        help=f"Path to SQLite databases directory (default: {DB_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Path to output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--db-name",
        default="bird_dev",
        help="Database name when loading from PostgreSQL dump (default: bird_dev)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PostgreSQL Explain Pipeline")
    print("=" * 60)
    
    if args.start_docker:
        if not start_docker_postgres():
            print("Failed to start Docker. Exiting.")
            return 1
    
    if not wait_for_postgres(args.host, args.port, args.user, args.password):
        print("PostgreSQL is not available. Exiting.")
        return 1
    
    single_db_name: Optional[str] = None
    
    if not args.skip_migration:
        if args.sqlite_source:
            print("\n" + "=" * 60)
            print("Step 1: Migrating SQLite databases to PostgreSQL")
            print("=" * 60)
            
            if not args.db_dir.exists():
                print(f"[WARN] Database directory not found: {args.db_dir}")
                print("       Skipping migration. Make sure databases exist in PostgreSQL.")
            else:
                migration_stats = migrate_all_databases(
                    db_dir=args.db_dir,
                    pg_host=args.host,
                    pg_port=args.port,
                    pg_user=args.user,
                    pg_password=args.password,
                    drop_existing=True,
                )
                
                print(f"\nMigration Summary:")
                for db_id, stats in migration_stats.items():
                    status = "OK" if stats["success"] else "FAILED"
                    print(f"  {db_id}: {status} - {stats['tables_migrated']} tables, {stats['rows_migrated']} rows")
        else:
            print("\n" + "=" * 60)
            print("Step 1: Loading from PostgreSQL dump (MINIDEV_postgresql)")
            print("=" * 60)
            
            if not load_postgresql_dump(
                sql_file=args.sql_source,
                pg_host=args.host,
                pg_port=args.port,
                pg_user=args.user,
                pg_password=args.password,
                db_name=args.db_name,
                drop_existing=True,
            ):
                print("Failed to load PostgreSQL dump. Exiting.")
                return 1
            single_db_name = args.db_name
    else:
        print("\n[INFO] Skipping database loading (--skip-migration)")
        single_db_name = None if args.sqlite_source else args.db_name
    
    print("\n" + "=" * 60)
    print("Step 2: Parsing gold SQL file")
    print("=" * 60)
    
    if not args.gold_file.exists():
        print(f"[ERROR] Gold file not found: {args.gold_file}")
        return 1
    
    entries = parse_gold_sql(args.gold_file)
    print(f"Parsed {len(entries)} queries from {args.gold_file.name}")
    
    unique_dbs = set(e["db_id"] for e in entries)
    print(f"Unique databases: {sorted(unique_dbs)}")
    
    print("\n" + "=" * 60)
    print("Step 3: Running ANALYZE (build statistics for accurate plans)")
    print("=" * 60)
    dbs_to_analyze = [single_db_name] if single_db_name else sorted(unique_dbs)
    run_analyze(
        db_names=dbs_to_analyze,
        pg_host=args.host,
        pg_port=args.port,
        pg_user=args.user,
        pg_password=args.password,
    )
    
    print("\n" + "=" * 60)
    print("Step 4: Running EXPLAIN on queries")
    print("=" * 60)
    
    results = run_explain_pipeline(
        entries=entries,
        pg_host=args.host,
        pg_port=args.port,
        pg_user=args.user,
        pg_password=args.password,
        single_db_name=single_db_name,
    )
    
    print("\n" + "=" * 60)
    print("Step 5: Writing output files")
    print("=" * 60)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    json_output = args.output_dir / "explain_trees_postgres.json"
    write_json_output(results, json_output)
    
    summary_output = args.output_dir / "explain_trees_postgres_summary.txt"
    write_summary_output(results, summary_output)
    
    success_count = sum(1 for r in results if r["error"] is None)
    failed_count = len(results) - success_count
    
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Total queries: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"\nOutputs:")
    print(f"  JSON:    {json_output}")
    print(f"  Summary: {summary_output}")
    
    if args.stop_docker:
        print()
        stop_docker_postgres()
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
