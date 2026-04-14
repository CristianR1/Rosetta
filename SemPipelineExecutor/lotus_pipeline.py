import pandas as pd
import lotus
import os
import re
import json
import sqlite3
import argparse
import time
import csv
from pathlib import Path
from lotus.models import LM
from abc import ABC, abstractmethod
import math

from rosetta_env import load_rosetta_env

load_rosetta_env()
lm = LM(model="gpt-4o-mini", max_tokens=1000)

# For GroupOperation (semantic clustering), configure retrieval model and vector store
from lotus.models import SentenceTransformersRM
from lotus.vector_store import FaissVS

rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

from pipeline_sources import DATABASE_TABLES as database_tables
from build_pipelines import load_or_build_pipelines, prepare_semantic_execution
from semantic_executor import (
    PlannedStep, SubqueryContext, enrich_plan_instructions,
    predicate_to_instruction,
)

class Operation(ABC):
    def __init__(self, instruction: str):
        self.instruction = instruction
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    @abstractmethod
    def modifies_docset(self) -> bool:
        pass

class FilterOperation(Operation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "contents" not in df.columns:
            return df
        result = df.sem_filter("{contents} includes information that indicates {instruction}".format(contents="{contents}", instruction=self.instruction))
        return result
    def modifies_docset(self) -> bool:
        return True

class ExtractOperation(Operation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "contents" not in df.columns:
            return df
        output_cols = {"extraction": "Extract ONLY: {instruction}. Return just the value, nothing else.".format(instruction=self.instruction)}
        result = df.sem_extract(["contents"], output_cols, extract_quotes=False)
        return result
    def modifies_docset(self) -> bool:
        return False

class RankOperation(Operation):
    def __init__(self, instruction: str, k: int = 1):
        super().__init__(instruction)
        self.k = k
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "contents" not in df.columns:
            return df
        result = df.sem_topk("In {contents}, {instruction}".format(contents = "{contents}", instruction=self.instruction), self.k)
        return result
    def modifies_docset(self) -> bool:
        return True

def _content_columns(df: pd.DataFrame) -> list:
    """Return list of column names that hold document content (contents, contents:left, contents:right, etc.)."""
    cols = []
    if "contents" in df.columns:
        cols.append("contents")
    for c in df.columns:
        if isinstance(c, str) and c != "contents" and "contents" in c.lower():
            cols.append(c)
    return cols


def normalize_joined_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    After a lotus join, the DataFrame has contents:left and contents:right instead of contents.
    Merge those into a single 'contents' column so downstream Filter/Extract/Rank/Aggregate work.
    In a join chain, merge all content-like columns (contents:left, contents:right, etc.) into one.
    Also set canonical filename/filepath from :left so interim save and row identity work.
    """
    if df is None or df.empty:
        return df
    # Work on a copy with a clean RangeIndex to avoid alignment/reindex issues
    df = df.copy().reset_index(drop=True)
    if "contents" in df.columns:
        return df
    content_cols = _content_columns(df)
    if not content_cols:
        return df
    # Build a single 'contents' series by concatenating all content-like columns
    base_col = content_cols[0]
    series = df[base_col].astype(str).reset_index(drop=True)
    for c in content_cols[1:]:
        if c in df.columns:
            add = df[c].astype(str).reset_index(drop=True)
            series = series + "\n\n" + add
    df["contents"] = series
    # Canonical filename/filepath from left side so interim save and identity work
    if "filename" not in df.columns and "filename:left" in df.columns:
        df["filename"] = df["filename:left"].astype(str)
    if "filepath" not in df.columns and "filepath:left" in df.columns:
        df["filepath"] = df["filepath:left"].astype(str)
    # Drop prior right-side columns to avoid column collisions in subsequent joins
    right_cols = [c for c in df.columns if isinstance(c, str) and c.endswith(":right")]
    if right_cols:
        df = df.drop(columns=right_cols)
    return df


def build_lotus_join_instruction(instruction: str, left_df: pd.DataFrame, right_df: pd.DataFrame) -> str:
    """
    questions.json uses semantic instructions: "one document" / "the other document" (no left/right).
    Lotus maps to sem_join by labeling inputs as First/Second so the instruction stays role-based.
    After a join, left_df may have contents:left/contents:right; we treat a single 'contents' or first content column.
    """
    left_cols = _content_columns(left_df)
    right_cols = _content_columns(right_df)
    if left_cols and right_cols:
        return "First document: {contents:left}. Second document: {contents:right}. " + instruction.strip()
    return instruction

class JoinOperation(Operation):
    def __init__(self, instruction: str, right_df: pd.DataFrame = None):
        super().__init__(instruction)
        self.right_df = right_df
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or (self.right_df is not None and self.right_df.empty):
            return df if not df.empty else pd.DataFrame()
        if self.right_df is not None:
            join_instruction = build_lotus_join_instruction(self.instruction, df, self.right_df)
            print(f"Well Formed Instruction: {join_instruction}")
            result = df.sem_join(self.right_df, join_instruction)
        else:
            result = df.copy()
        return result
    def modifies_docset(self) -> bool:
        return True

class AggregateOperation(Operation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "contents" not in df.columns:
            return df
        result = df.sem_agg("Aggregate {instruction} across {contents}. Return ONLY the resulting numerical value without any other text or formatting".format(instruction=self.instruction,contents = "{contents}"))
        return result
    def modifies_docset(self) -> bool:
        return False

class GroupOperation(Operation):
    """
    Group documents by semantic similarity: index the content column for similarity retrieval,
    then cluster so that documents with similar conditions are binarily separated (two clusters).
    Requires lotus.settings to have rm (e.g. SentenceTransformersRM) and vs (e.g. FaissVS) configured.
    """
    # Default index dir for semantic grouping (contents column)
    GROUP_INDEX_DIR = "lotus_group_contents_index"

    def __init__(self, instruction: str, group_col: str = None):
        super().__init__(instruction)
        self.group_col = group_col

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if self.group_col and self.group_col in df.columns:
            return df.groupby(self.group_col).apply(lambda x: x)
        # Semantic grouping: use "contents" (or first text-like column), index then cluster
        content_col = "contents" if "contents" in df.columns else None
        if content_col is None:
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                    content_col = c
                    break
        if content_col is None:
            raise ValueError("GroupOperation: no 'contents' or text column found in DataFrame")
        try:
            rm, vs = lotus.settings.rm, lotus.settings.vs
        except Exception:
            rm, vs = None, None
        if rm is None or vs is None:
            raise ValueError(
                "GroupOperation requires a retrieval model (RM) and vector store (VS). "
                "Configure with e.g. lotus.settings.configure(lm=..., rm=SentenceTransformersRM(...), vs=FaissVS())"
            )
        n_clusters = min(2, max(1, len(df)))
        index_dir = self.GROUP_INDEX_DIR
        if df.attrs.get("index_dirs", {}).get(content_col) != index_dir:
            df = df.sem_index(content_col, index_dir)
        return df.sem_cluster_by(content_col, n_clusters)
    def modifies_docset(self) -> bool:
        return True

class DocumentManager:
    def __init__(self, data_root: str, table_names: list, max_docs: int = 0):
        self.data_root = Path(data_root)
        self.table_names = table_names
        self.max_docs = max_docs
        self.data_paths = []
        for table in self.table_names:
            database_name = database_tables.get(table)
            self.data_paths.append(self.data_root / database_name / table)
        for path in self.data_paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def load_documents(self, source_path: str = None) -> pd.DataFrame:
        """Load documents as DataFrame, respecting max_docs limit.
        
        Files are sorted numerically by the number in the filename (e.g., customers1.txt,
        customers2.txt, ..., customers10.txt) to match SQL's ORDER BY ROWID ordering.
        """
        path = Path(source_path) if source_path else self.data_paths[0]
        
        def _numeric_sort_key(p: Path) -> int:
            """Extract numeric portion from filename for proper ordering."""
            import re
            match = re.search(r'(\d+)', p.stem)
            return int(match.group(1)) if match else 0
        
        documents = []
        for txt_file in sorted(path.glob("*.txt"), key=_numeric_sort_key):
            with open(txt_file, 'r', encoding='utf-8') as f:
                documents.append({"filename": txt_file.name, "contents": f.read(), "filepath": str(txt_file)})
            if self.max_docs > 0 and len(documents) >= self.max_docs:
                break
        return pd.DataFrame(documents)

class Pipeline:
    def __init__(self, doc_manager: DocumentManager, verbose: bool = False):
        self.doc_manager = doc_manager
        self.pipelines = []
        self.results = []
        self.verbose = verbose
        self.operations = []
    def log(self, message: str):
        if self.verbose:
            print(message)

    def _parse_ops_from_segment(self, segment_str: str, collect_ops: list | None = None) -> list:
        """Parse a single segment string ('TYPE - instruction' separated by &&) into a list of Operation objects.
        
        New format uses && to separate operations within a segment (not commas).
        If collect_ops is provided, OP operations are appended there instead of self.operations.
        """
        operations = []
        if not segment_str or not isinstance(segment_str, str):
            return operations
        
        ops_target = collect_ops if collect_ops is not None else self.operations
        
        # Split by && for operations within segment
        parts = [p.strip() for p in segment_str.split("&&") if p.strip()]
        for part in parts:
            if " - " in part:
                op_type, instruction = part.split(" - ", 1)
                op_type = op_type.strip().upper()
                instruction = instruction.strip()
                if op_type == "FILTER":
                    operations.append(FilterOperation(instruction))
                elif op_type == "EXTRACT":
                    operations.append(ExtractOperation(instruction))
                elif op_type == "RANK":
                    operations.append(RankOperation(instruction))
                elif op_type in ("JOIN", "LEFT JOIN", "RIGHT JOIN"):
                    operations.append(JoinOperation(instruction))
                elif op_type == "AGGREGATE":
                    operations.append(AggregateOperation(instruction))
                elif op_type == "GROUP":
                    operations.append(GroupOperation(instruction))
                elif op_type == "OP":
                    for op in instruction.split(","):
                        ops_target.append(op.strip())
            elif part.strip().startswith("OP:"):
                op_part = part.strip()[3:].strip()
                for op in op_part.split(","):
                    ops_target.append(op.strip())
        return operations

    def _parse_single_pipeline(self, format_str: str) -> dict:
        """Parse one pipeline segment (trunk/branches/operations). Returns {trunk, branches, operations}."""
        ops_for_this = []
        trunk_ops = []
        branches = []
        
        format_str = format_str.strip()
        if " / " not in format_str:
            trunk_ops = self._parse_ops_from_segment(format_str, collect_ops=ops_for_this)
            return {"trunk": trunk_ops, "branches": [], "operations": ops_for_this}
        
        first_split_idx = format_str.find(" / ")
        trunk_str = format_str[:first_split_idx].strip()
        branches_str = format_str[first_split_idx + 3:].strip()
        
        trunk_ops = self._parse_ops_from_segment(trunk_str)
        remaining = branches_str
        
        while remaining:
            remaining = remaining.strip()
            if not remaining:
                break
            if remaining.startswith("OP:") or remaining.startswith("&& OP:"):
                if remaining.startswith("&& "):
                    remaining = remaining[3:]
                op_part = remaining[3:].strip()
                for op in op_part.split(","):
                    ops_for_this.append(op.strip())
                break
            pipe_idx = remaining.find("|")
            if pipe_idx == -1:
                if remaining.strip().startswith("OP:") or " OP:" in remaining:
                    if remaining.strip().startswith("OP:"):
                        op_part = remaining.strip()[3:].strip()
                        for op in op_part.split(","):
                            ops_for_this.append(op.strip())
                    elif "&& OP:" in remaining:
                        parts = remaining.split("&& OP:")
                        if parts[0].strip():
                            branch_ops = self._parse_ops_from_segment(parts[0].strip())
                            if branch_ops:
                                branches.append(branch_ops)
                        if len(parts) > 1:
                            for op in parts[1].split(","):
                                ops_for_this.append(op.strip())
                else:
                    branch_ops = self._parse_ops_from_segment(remaining)
                    if branch_ops:
                        branches.append(branch_ops)
                break
            branch_content = remaining[:pipe_idx].strip()
            remaining = remaining[pipe_idx + 1:].strip()
            if remaining.startswith("/"):
                remaining = remaining[1:].strip()
            elif remaining.startswith(" / "):
                remaining = remaining[3:].strip()
            if branch_content:
                branch_ops = self._parse_ops_from_segment(branch_content)
                branches.append(branch_ops)
            else:
                branches.append([])
        
        return {"trunk": trunk_ops, "branches": branches, "operations": ops_for_this}

    def parse_format(self, format_str: str):
        """
        Parse format string with tree-like splits for BFS-style execution.
        
        Multi-pipeline: ' |-| ' separates pipelines (one per SELECT expression).
        Each pipeline: trunk / branch1 | / branch2 | && OP: post_ops
        """
        self.pipelines = []
        self.pipeline_groups = []
        self.operations = []
        
        format_str = format_str.strip()
        if not format_str:
            return
        
        # Split on |-| for multi-pipeline (multi-answer) format
        segments = [s.strip() for s in format_str.split(" |-| ") if s.strip()]
        if not segments:
            segments = [format_str]
        
        for seg in segments:
            group = self._parse_single_pipeline(seg)
            self.pipeline_groups.append(group)
            self.pipelines.append({"trunk": group["trunk"], "branches": group["branches"]})
        
        if len(self.pipeline_groups) == 1:
            self.operations = self.pipeline_groups[0]["operations"]
        
        total = sum(len(g["trunk"]) + sum(len(b) for b in g["branches"]) for g in self.pipeline_groups)
        self.log(f"Parsed {len(self.pipeline_groups)} pipeline(s), {total} total ops")
    def _run_ops_on_df(self, df: pd.DataFrame, operations: list, table_dfs: list = None, is_join_phase: bool = False) -> tuple:
        """Run a list of operations on df sequentially (BFS-style).
        
        For join phase, table_dfs contains all loaded tables and joins are performed sequentially.
        Returns (result_df, failed_flag).
        """
        if is_join_phase and table_dfs is not None:
            for idx, op in enumerate(operations):
                if len(table_dfs) < 2:
                    break
                op_name = type(op).__name__
                self.log(f"Executing {op_name}: {op.instruction}")
                try:
                    op.right_df = table_dfs[1]
                    df = op.execute(table_dfs[0])
                    if hasattr(df, 'to_pandas'):
                        df = df.to_pandas()
                    df = normalize_joined_df(df)
                    table_dfs[:] = [df] + table_dfs[2:]
                    self.log(f"{op_name} complete - {len(df)} documents in joined result")
                    if len(df) == 0:
                        return df, True
                    self.results.append({"operation": op_name, "instruction": op.instruction, "doc_count": len(df), "modifies_docset": True})
                except Exception as e:
                    self.log(f"Join operation failed with error: {e}. Marking pipeline as failed.")
                    return pd.DataFrame(), True
            out = table_dfs[0] if table_dfs else pd.DataFrame()
            return (normalize_joined_df(out) if not out.empty else out), False
        
        # Sequential execution of non-join operations
        for idx, operation in enumerate(operations):
            op_name = type(operation).__name__
            try:
                if "contents" in df.columns:
                    self.log(f"{len(df['contents'])} documents remaining ...")
                    if len(df['contents']) == 0 and idx != 0:
                        raise ValueError("No documents remaining")
            except Exception:
                self.log(f"Exiting early, pipeline failure on operation {idx}")
                break
            self.log(f"[{idx+1}/{len(operations)}] Executing {op_name}: {operation.instruction}")
            df = operation.execute(df)
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            self.log(f"[{idx+1}/{len(operations)}] {op_name} complete - {len(df)} documents remaining")
            
            # Log non-modifying operation results (like Extract, Aggregate)
            if not operation.modifies_docset():
                self.log(f"Operation Result:")
                for col in df.columns:
                    if col not in ["filename", "contents", "filepath"]:
                        self.log(f"  Column '{col}':")
                        for i, val in enumerate(df[col].values):
                            self.log(f"    [{i}] {val}")
            
            self.results.append({
                "operation": op_name,
                "instruction": operation.instruction,
                "doc_count": len(df),
                "modifies_docset": operation.modifies_docset()
            })
        return df, False

    def execute(self) -> list:
        """Execute pipelines using BFS-style execution.
        
        Multi-pipeline: outer loop over pipeline_groups (split by |-|).
        Per group: trunk + branches -> dfs. Returns list of {dfs, operations, trunk_df} per group.
        """
        all_group_results = []
        pipeline_groups = getattr(self, "pipeline_groups", None)
        if not pipeline_groups:
            pipeline_groups = [{"trunk": p["trunk"], "branches": p["branches"], "operations": self.operations} for p in self.pipelines]
        
        for group_idx, group in enumerate(pipeline_groups):
            trunk_ops = [op for op in group["trunk"] if type(op).__name__ != "GroupOperation"]
            branches = [[op for op in b if type(op).__name__ != "GroupOperation"] for b in group["branches"]]
            
            table_dfs = []
            for path in self.doc_manager.data_paths:
                self.log(f"Loading documents from {path}")
                df = self.doc_manager.load_documents(str(path))
                table_dfs.append(df)
                self.log(f"Loaded {len(df)} documents from {path}")
            
            self.results = [{"operation": "initial", "doc_count": sum(len(d) for d in table_dfs), "tables": len(table_dfs)}]
            
            # Separate join operations from other trunk operations
            join_idx = 0
            while join_idx < len(trunk_ops) and type(trunk_ops[join_idx]).__name__ == 'JoinOperation':
                join_idx += 1
            join_ops = trunk_ops[:join_idx]
            rest_trunk_ops = trunk_ops[join_idx:]
            
            # joins
            ext = False
            if join_ops:
                df_trunk, ext = self._run_ops_on_df(None, join_ops, table_dfs=table_dfs, is_join_phase=True)
                df_after_join = df_trunk.copy() if not ext and df_trunk is not None else None
            else:
                df_trunk = table_dfs[0] if table_dfs else pd.DataFrame()
                df_after_join = None
            # Percent count denominator: count after last join (if joins) else original doc set
            percent_count_denom_df = df_after_join if df_after_join is not None else (table_dfs[0] if table_dfs else None)
            
            if ext:
                self.log("Exiting early, pipeline failure on join phase")
                all_group_results.append({"dfs": [], "operations": group["operations"], "trunk_df": None})
                continue
            
            if len(df_trunk) > 0:
                self.log(f"DataFrame columns: {list(df_trunk.columns)}")
            
            if rest_trunk_ops:
                df_trunk, ext = self._run_ops_on_df(df_trunk, rest_trunk_ops, is_join_phase=False)
                if ext:
                    all_group_results.append({"dfs": [], "operations": group["operations"], "trunk_df": None, "percent_count_denom_df": None})
                    continue
            
            if not branches:
                group_dfs = [df_trunk]
                self.log(f"Pipeline group {group_idx + 1} complete (no branches) - {len(df_trunk)} documents")
            else:
                self.trunk_df = df_trunk
                group_dfs = []
                for branch_idx, branch_ops in enumerate(branches):
                    df_branch = df_trunk.copy()
                    if not branch_ops:
                        group_dfs.append(df_branch)
                        self.log(f"Branch {branch_idx + 1} (empty) - passed through trunk result")
                        continue
                    self.log(f"Branch {branch_idx + 1}/{len(branches)} starting from trunk state ({len(df_branch)} documents)")
                    df_branch, ext = self._run_ops_on_df(df_branch, branch_ops, is_join_phase=False)
                    if not ext:
                        group_dfs.append(df_branch)
                        self.log(f"Branch {branch_idx + 1} complete - {len(df_branch)} documents in result")
            
            all_group_results.append({
                "dfs": group_dfs,
                "operations": group["operations"],
                "trunk_df": self.trunk_df if branches else df_trunk,
                "percent_count_denom_df": percent_count_denom_df,
            })
        
        return all_group_results

from pipeline_sources import extract_tables_from_sql


# ---------------------------------------------------------------------------
# Semantic plan executor (multi-head DAG, subquery-first)
# ---------------------------------------------------------------------------

def _step_to_lotus_op(
    step: PlannedStep,
    ctx: SubqueryContext,
    entry: dict,
    right_df: pd.DataFrame | None = None,
) -> Operation | None:
    """Map a PlannedStep to a concrete Lotus Operation instance."""
    instr = step.instruction
    if ctx:
        instr = ctx.substitute(instr)
    kind = step.kind

    if kind == "sem_filter":
        return FilterOperation(instr)
    if kind == "sem_join":
        return JoinOperation(instr, right_df=right_df)
    if kind == "sem_agg":
        return AggregateOperation(instr)
    if kind == "sem_extract":
        return ExtractOperation(instr)
    if kind == "sem_topk":
        limit = step.op.get("limit_count", 1)
        return RankOperation(instr, k=limit)
    if kind == "sem_cluster_by":
        keys = step.op.get("group_keys", [])
        col = keys[0] if keys else None
        return GroupOperation(instr, group_col=col)
    if kind == "sem_map":
        return ExtractOperation(instr)
    return None


def _run_one_op(op: Operation, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Execute one Lotus Operation, normalizing output."""
    if verbose:
        print(f"    [{type(op).__name__}] {op.instruction}")
    result = op.execute(df)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    if verbose:
        print(f"    -> {len(result)} rows")
    return result


def _run_subquery_pipeline(
    sq_plan: dict,
    entry: dict,
    doc_manager: DocumentManager,
    ctx: SubqueryContext,
    verbose: bool = False,
) -> object:
    """Execute one subquery pipeline and return its scalar result."""
    pipeline_dict = sq_plan["pipeline"]
    steps: list[PlannedStep] = sq_plan["steps"]
    subquery_var = pipeline_dict.get("subquery_var", "")

    # Subqueries typically operate on a single table; load all available
    tables = [t.strip() for t in entry.get("tables", "").split(",") if t.strip()]
    if not tables:
        return None
    df = doc_manager.load_documents(str(doc_manager.data_paths[0]))

    if verbose:
        print(f"  Subquery {subquery_var}: {len(steps)} steps on {len(df)} docs")

    for step in steps:
        if step.kind == "subquery_result":
            continue
        op = _step_to_lotus_op(step, ctx, entry)
        if op is None:
            continue
        df = _run_one_op(op, df, verbose=verbose)
        df = normalize_joined_df(df) if "contents" not in df.columns else df

    # Extract scalar: prefer _output, then extraction, then count
    if "_output" in df.columns and len(df) > 0:
        val = df["_output"].iloc[0]
    elif "extraction" in df.columns and len(df) > 0:
        val = df["extraction"].iloc[0]
    else:
        val = len(df)

    if verbose:
        print(f"  Subquery {subquery_var} -> {val}")
    return val


class UnsupportedOperationError(Exception):
    """Raised when a pipeline operation is not supported."""
    pass


def _plan_contains_unsupported_op(plan: dict) -> str | None:
    """Check if plan contains any unsupported operations. Returns op name or None.
    
    Note: sem_cluster_by is now supported with RM/VS configured.
    sem_topk is now also supported via RankOperation.
    """
    unsupported: set[str] = set()  # All operations now supported
    
    main_steps = plan.get("main_steps", [])
    for step in main_steps:
        if step.kind in unsupported:
            return step.kind
    
    for sq_plan in plan.get("subquery_plans", []):
        for step in sq_plan.get("steps", []):
            if hasattr(step, 'kind') and step.kind in unsupported:
                return step.kind
    
    return None


def execute_semantic_plan(
    plan: dict,
    entry: dict,
    doc_manager: DocumentManager,
    verbose: bool = False,
) -> tuple:
    """Execute a full semantic plan: subqueries first, then multi-head main pipeline.

    Returns a result tuple suitable for metric comparison.
    """
    # Early check for unsupported operations
    unsupported_op = _plan_contains_unsupported_op(plan)
    if unsupported_op:
        raise UnsupportedOperationError(
            f"{unsupported_op} is not supported by lotus - skipping query"
        )
    
    ctx = SubqueryContext()
    enrich_plan_instructions(plan, entry, ctx)

    # ---- 1. Subqueries ----
    for sq_plan in plan.get("subquery_plans", []):
        var = sq_plan["pipeline"].get("subquery_var", "")
        val = _run_subquery_pipeline(sq_plan, entry, doc_manager, ctx, verbose)
        if var:
            ctx.bind(var, val)
            if verbose:
                print(f"  Bound {var} = {val}")

    # Re-enrich with bound context so $N substitution takes effect
    enrich_plan_instructions(plan, entry, ctx)

    # ---- 2. Main pipeline heads ----
    tables = plan.get("tables", [])
    table_to_head = plan.get("table_to_head", {})
    heads: dict[int, pd.DataFrame] = {}
    for idx, table in enumerate(tables):
        if idx < len(doc_manager.data_paths):
            heads[idx] = doc_manager.load_documents(str(doc_manager.data_paths[idx]))
            if verbose:
                print(f"  Head {idx} ({table}): {len(heads[idx])} docs")

    if not heads:
        return ()

    main_steps: list[PlannedStep] = plan.get("main_steps", [])

    # Track per-cluster DFs after sem_cluster_by
    cluster_groups: dict[int, list[pd.DataFrame]] | None = None
    
    # Collect results from final operations (agg/extract) separately
    final_results: list[pd.DataFrame] = []

    for step in main_steps:
        kind = step.kind

        if kind == "subquery_result":
            ref = step.op.get("subquery_ref", "")
            if ref not in ctx:
                if verbose:
                    print(f"  [WARN] subquery {ref} not bound")
            continue

        if kind == "sem_join" and step.merge_pair:
            left_id, right_id = step.merge_pair
            left_df = heads.get(left_id, pd.DataFrame())
            right_df = heads.get(right_id, pd.DataFrame())
            op = _step_to_lotus_op(step, ctx, entry, right_df=right_df)
            if op is None:
                continue
            joined = _run_one_op(op, left_df, verbose=verbose)
            joined = normalize_joined_df(joined)
            heads[left_id] = joined
            heads.pop(right_id, None)
            cluster_groups = None
            continue

        # Determine target heads
        target_ids = step.head_ids if step.head_ids else list(heads.keys())

        if step.group_scoped and cluster_groups:
            # Apply operation to every cluster group independently
            for hid in target_ids:
                if hid not in cluster_groups:
                    continue
                new_groups = []
                for grp_df in cluster_groups[hid]:
                    op = _step_to_lotus_op(step, ctx, entry)
                    if op is None:
                        new_groups.append(grp_df)
                        continue
                    result = _run_one_op(op, grp_df, verbose=verbose)
                    new_groups.append(result)
                cluster_groups[hid] = new_groups
            continue

        # For non-group-scoped ops (like TopK), merge cluster groups first if present
        if cluster_groups and not step.group_scoped:
            for hid in list(cluster_groups.keys()):
                if cluster_groups[hid]:
                    heads[hid] = pd.concat(cluster_groups[hid], ignore_index=True)
            cluster_groups = None  # Clear after merging

        for hid in target_ids:
            if hid not in heads:
                continue
            op = _step_to_lotus_op(step, ctx, entry)
            if op is None:
                continue
            result = _run_one_op(op, heads[hid], verbose=verbose)

            if kind == "sem_cluster_by":
                # After clustering, split into groups
                cluster_col = None
                for c in result.columns:
                    if "cluster" in str(c).lower():
                        cluster_col = c
                        break
                if cluster_col:
                    groups = [g.reset_index(drop=True) for _, g in result.groupby(cluster_col)]
                    if cluster_groups is None:
                        cluster_groups = {}
                    cluster_groups[hid] = groups
                    if verbose:
                        print(f"    Clustered into {len(groups)} groups")
                heads[hid] = result
            elif step.is_final:
                # Final operations (agg/extract) collect results without replacing heads
                # so subsequent finals can also run on the original data
                final_results.append(result)
            else:
                heads[hid] = result

    # ---- 3. Collect results ----
    # If we have cluster groups, merge them back
    if cluster_groups:
        for hid, groups in cluster_groups.items():
            if groups:
                heads[hid] = pd.concat(groups, ignore_index=True)

    # Get the final DF (should be single head after joins)
    final_df = list(heads.values())[0] if heads else pd.DataFrame()

    def _is_divide_by_zero_ratio(v) -> bool:
        """Check if value is a ratio with 0 denominator like '10:0', '5:0', 'Infinity'."""
        if v is None:
            return False
        s = str(v).strip()
        # Check for patterns like "N:0" or "N/0" or "Infinity"
        if s.lower() in ("infinity", "inf", "-infinity", "-inf", "nan"):
            return True
        if re.match(r"^\d+(\.\d+)?[:/]0$", s):
            return True
        return False

    # Extract answer(s) from final operation results
    values: list = []
    
    # If no final results were collected and main DF is empty, return empty
    if not final_results and final_df.empty:
        return ()
    
    # Collect values from all final operation results
    for result_df in final_results:
        if result_df.empty:
            continue
        if "_output" in result_df.columns:
            for v in result_df["_output"]:
                if v is not None:
                    if _is_divide_by_zero_ratio(v):
                        continue
                    values.append(v)
        elif "extraction" in result_df.columns:
            for v in result_df["extraction"]:
                if v is not None:
                    if _is_divide_by_zero_ratio(v):
                        continue
                    values.append(v)
    
    # Fallback: if no final ops collected but we have data in heads, check there
    if not values and not final_results:
        if "_output" in final_df.columns and len(final_df) > 0:
            for v in final_df["_output"]:
                if v is not None:
                    if _is_divide_by_zero_ratio(v):
                        continue
                    values.append(v)
        elif "extraction" in final_df.columns and len(final_df) > 0:
            for v in final_df["extraction"]:
                if v is not None:
                    if _is_divide_by_zero_ratio(v):
                        continue
                    values.append(v)

    return tuple(str(v) for v in values) if values else ()


def limit_sql_to_num_documents(sql: str, num_documents: int) -> str:
    tables = extract_tables_from_sql(sql)
    tables = [t for t in tables if database_tables.get(t)]
    if not tables or num_documents <= 0:
        return sql
    tables_set = {t.lower() for t in tables}
    n = len(sql)
    depth = 0
    out = []
    i = 0
    while i < n:
        c = sql[i]
        if c == '(':
            depth += 1
            out.append(c)
            i += 1
            continue
        if c == ')':
            depth -= 1
            out.append(c)
            i += 1
            continue
        if depth > 0:
            out.append(c)
            i += 1
            continue
        found_kw = False
        kw_end = i
        for kw in ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FROM', 'JOIN']:
            parts = kw.split()
            j = i
            for p in parts:
                while j < n and sql[j] in ' \t':
                    j += 1
                if j + len(p) > n or sql[j:j+len(p)].upper() != p.upper():
                    break
                j += len(p)
            else:
                if j < n and (sql[j].isalnum() or sql[j] == '_'):
                    continue
                found_kw = True
                kw_end = j
                break
        if not found_kw:
            out.append(c)
            i += 1
            continue
        out.append(sql[i:kw_end])
        i = kw_end
        while i < n and sql[i] in ' \t':
            i += 1
        table_start = i
        while i < n and (sql[i].isalnum() or sql[i] == '_'):
            i += 1
        table_name = sql[table_start:i]
        if table_name.lower() not in tables_set:
            out.append(sql[table_start:i])
            while i < n and sql[i] in ' \t':
                i += 1
            if i + 2 <= n and sql[i:i+2].upper() == 'AS':
                i += 2
                while i < n and sql[i] in ' \t':
                    i += 1
                while i < n and (sql[i].isalnum() or sql[i] == '_'):
                    i += 1
            continue
        alias = table_name
        while i < n and sql[i] in ' \t':
            i += 1
        if i + 2 <= n and sql[i:i+2].upper() == 'AS':
            i += 2
            while i < n and sql[i] in ' \t':
                i += 1
            alias_start = i
            while i < n and (sql[i].isalnum() or sql[i] == '_'):
                i += 1
            alias = sql[alias_start:i]
        subq = f"(SELECT * FROM {table_name} ORDER BY ROWID LIMIT {num_documents}) AS {alias} "
        out.append(subq)
    return ''.join(out)

def execute_sql(db_path, sql: str, database_name: str):
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path / database_name / f"{database_name}.sqlite")
    cursor = conn.cursor()
    print(sql)
    cursor.execute(sql)
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return results, columns

def validate_result(final_result: list, db_path: str, database_name: str, sql: str):
    sql_results, sql_columns = execute_sql(db_path, sql, database_name)
    ground_truth = [row[0] if len(row) == 1 else row for row in sql_results]
    print(f"\nExtracted: {final_result}")
    print(f"Ground Truth: {ground_truth}")
    return final_result, ground_truth


# Pipeline run configuration
ALLOWED_DB_IDS = None  # None means all databases
ALLOWED_DIFFICULTIES = {"simple", "moderate", "challenging"}


def _normalize_for_compare(val) -> str:
    """Strip and clean to words/numerics only, lowercase, for comparison."""
    if val is None:
        return ""
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        parts = []
        for v in val:
            parts.extend(re.findall(r"[a-z0-9]+", str(v).lower()))
        return " ".join(sorted(parts))
    return " ".join(sorted(re.findall(r"[a-z0-9]+", str(val).lower())))


def _answers_match(extracted, ground_truth: list) -> bool:
    """True if extracted matches ground_truth.
    Handles: extracted as tuple (multi-pipeline), ground_truth as [(a,b)] or [a,b].
    """
    if not ground_truth:
        return not extracted
    ex = extracted if isinstance(extracted, (list, tuple)) else [extracted]
    gt = ground_truth
    if len(gt) == 1 and isinstance(gt[0], (list, tuple)) and len(ex) == len(gt[0]):
        gt = list(gt[0])
    norm_ex = _normalize_for_compare(ex)
    norm_gt = _normalize_for_compare(gt)
    if norm_ex == norm_gt:
        return True

    def _extract_numbers(val) -> list[float]:
        txt = str(val)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        out = []
        for n in nums:
            try:
                out.append(float(n))
            except Exception:
                continue
        return out

    ex_nums = _extract_numbers(ex)
    gt_nums = _extract_numbers(gt)
    if gt_nums and ex_nums:
        tol = 1e-2
        all_present = True
        for g in gt_nums:
            if not any(abs(e - g) <= tol for e in ex_nums):
                all_present = False
                break
        if all_present:
            return True
    return False


def _get_ground_truth(db_path: str, database_name: str, sql: str):
    """Return ground truth list from DB; empty list if DB missing or error."""
    if not Path(db_path).exists():
        return []
    try:
        results, _ = execute_sql(db_path, sql, database_name)
        return [row[0] if len(row) == 1 else row for row in results]
    except Exception:
        return []


def aggregate_metrics_to_csv(rows: list, output_dir: str = "./pipeline_data/results/lotus", num_skipped_unsupported: int = 0) -> str:
    """Write per-question metrics and summary to CSV; return path to written file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "lotus_metrics.csv"
    if not rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question_number", "question_id", "db_id", "difficulty", "num_documents",
                        "altered_sql", "execution_time_seconds", "llm_total_tokens", "llm_total_cost",
                        "correct", "extracted", "ground_truth"])
        return str(filepath)
    fieldnames = ["question_number", "question_id", "db_id", "difficulty", "num_documents",
                 "altered_sql", "execution_time_seconds", "llm_total_tokens", "llm_total_cost",
                 "correct", "extracted", "ground_truth"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_export = {k: r.get(k) for k in fieldnames}
            row_export["extracted"] = str(r.get("extracted", ""))[:500]
            row_export["ground_truth"] = str(r.get("ground_truth", ""))[:500]
            writer.writerow(row_export)
    total_time = sum(r.get("execution_time_seconds") or 0 for r in rows)
    total_tokens = sum(r.get("llm_total_tokens") or 0 for r in rows)
    total_cost = sum(r.get("llm_total_cost") or 0 for r in rows)
    num_correct = sum(1 for r in rows if r.get("correct") is True)
    summary_path = output_path / "lotus_metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_questions", len(rows)])
        w.writerow(["total_execution_time_seconds", round(total_time, 4)])
        w.writerow(["total_llm_tokens", total_tokens])
        w.writerow(["total_llm_cost", round(total_cost, 6)])
        w.writerow(["num_correct", num_correct])
        w.writerow(["accuracy", round(num_correct / len(rows), 4) if rows else 0])
        w.writerow(["num_skipped_unsupported", num_skipped_unsupported])
    return str(filepath)


def _filter_entries_by_ops(entries: list, ops: list[str]) -> list:
    """Keep only entries whose semantic_nl contains at least one of the requested op types."""
    if not ops:
        return entries
    target = set()
    alias_map = {
        "filter": "sem_filter", "join": "sem_join", "group": "sem_cluster_by",
        "agg": "sem_agg", "extract": "sem_extract", "topk": "sem_topk",
    }
    for o in ops:
        mapped = alias_map.get(o.lower(), o)
        target.add(mapped)
    filtered = []
    for e in entries:
        nl = e.get("semantic_nl", [])
        entry_ops = {row["op_type"] for row in nl}
        if entry_ops & target:
            filtered.append(e)
    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_documents", type=int, default=10, help="Limit each table in SQL to this many rows (ORDER BY ROWID) to align with document set size")
    parser.add_argument("--semantic", action="store_true", help="Use semantic plan execution instead of format strings")
    parser.add_argument("--ops", nargs="*", default=[], help="Only run entries containing these ops: filter, join, group, agg, extract, topk")
    parser.add_argument("--null-param", default=None, help="Null representation directory (e.g. null_binary_explicit)")
    parser.add_argument("--noise-param", default=None, help="Noise ratio directory (e.g. 1_data_1_noise)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of questions to run (0 = all)")
    args = parser.parse_args()
    num_documents = args.num_documents
    use_semantic = args.semantic

    _DEMO_DIR = Path(__file__).resolve().parent
    MINIDEV_PATH = _DEMO_DIR / "MINIDEV"
    DB_PATH = MINIDEV_PATH / "dev_databases"

    if args.null_param and args.noise_param:
        data_root = _DEMO_DIR / "pipeline_data" / args.null_param / args.noise_param / "Text"
    elif args.null_param:
        data_root = _DEMO_DIR / "pipeline_data" / args.null_param / "Text"
    else:
        data_root = _DEMO_DIR / "pipeline_data" / "data"

    pipeline_data = load_or_build_pipelines()
    d = pipeline_data["entries"]
    if not d:
        print("Failed to load question sequence")
    else:
        # Keep original pipelines.json order (by line_no)
        filtered_entries = [
            e for e in d
            if (ALLOWED_DB_IDS is None or e["db_id"] in ALLOWED_DB_IDS)
            and e["difficulty"] in ALLOWED_DIFFICULTIES
        ]

        if args.ops:
            filtered_entries = _filter_entries_by_ops(filtered_entries, args.ops)
        if args.limit > 0:
            filtered_entries = filtered_entries[:args.limit]

        print(f"[lotus] Running {len(filtered_entries)} entries | data_root={data_root}")

        metrics_rows = []
        num_skipped_unsupported = 0
        for question_number, entry in enumerate(filtered_entries, start=1):
            tables_list = entry["tables"].split(", ")
            database_name = database_tables.get(tables_list[0])
            if not database_name:
                continue
            doc_manager = DocumentManager(str(data_root), tables_list, max_docs=num_documents)

            sem_plan = prepare_semantic_execution(entry) if use_semantic else None

            lotus.settings.lm.reset_stats()
            start_time = time.perf_counter()
            execution_time_seconds = 0.0
            try:
                if sem_plan:
                    extracted = execute_semantic_plan(
                        sem_plan, entry, doc_manager, verbose=True)
                    execute_result = None
                else:
                    sequence = entry["format"]
                    pipeline = Pipeline(doc_manager, verbose=True)
                    pipeline.parse_format(sequence)
                    execute_result = pipeline.execute()
                    extracted = None
            except UnsupportedOperationError as e:
                execution_time_seconds = time.perf_counter() - start_time
                num_skipped_unsupported += 1
                print(f"\n[SKIP] Question {question_number} ({entry.get('question_id', '')}): {e}")
                gt_sql = entry.get("sql_sqlite") or entry["SQL"]
                limited_sql = limit_sql_to_num_documents(gt_sql, num_documents)
                ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
                metrics_rows.append({
                    "question_number": question_number,
                    "question_id": entry.get("question_id", ""),
                    "db_id": entry["db_id"],
                    "difficulty": entry["difficulty"],
                    "num_documents": 0,
                    "altered_sql": limited_sql,
                    "execution_time_seconds": 0.0,
                    "llm_total_tokens": 0,
                    "llm_total_cost": 0.0,
                    "correct": False,
                    "extracted": [],
                    "ground_truth": ground_truth,
                    "skipped": True,
                })
                continue
            except Exception as e:
                execution_time_seconds = time.perf_counter() - start_time
                print(f"\nError on question {question_number} ({entry.get('question_id', '')}): {e}")
                import traceback
                traceback.print_exc()
                gt_sql = entry.get("sql_sqlite") or entry["SQL"]
                limited_sql = limit_sql_to_num_documents(gt_sql, num_documents)
                ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
                metrics_rows.append({
                    "question_number": question_number,
                    "question_id": entry.get("question_id", ""),
                    "db_id": entry["db_id"],
                    "difficulty": entry["difficulty"],
                    "num_documents": 0,
                    "altered_sql": limited_sql,
                    "execution_time_seconds": round(execution_time_seconds, 4),
                    "llm_total_tokens": 0,
                    "llm_total_cost": 0.0,
                    "correct": False,
                    "extracted": [],
                    "ground_truth": ground_truth,
                })
                continue
            execution_time_seconds = time.perf_counter() - start_time

            lm_stats = lotus.settings.lm.stats
            llm_tokens = lm_stats.physical_usage.total_tokens
            llm_cost = lm_stats.physical_usage.total_cost
            if sem_plan:
                num_docs_processed = sum(len(doc_manager.load_documents(str(p))) for p in doc_manager.data_paths)
            else:
                num_docs_processed = pipeline.results[0]["doc_count"] if pipeline.results else 0

            gt_sql = entry.get("sql_sqlite") or entry["SQL"]
            limited_sql = limit_sql_to_num_documents(gt_sql, num_documents)

            # Semantic path already produced `extracted`; skip format-path OP logic
            if sem_plan:
                extracted_fmt = "[" + ", ".join(str(x) for x in extracted) + "]" if extracted else "[]"
                ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
                if DB_PATH.exists() and limited_sql:
                    print(f"\nExtracted: {extracted_fmt}")
                    print(f"Ground Truth: {ground_truth}")
                    correct = _answers_match(extracted, ground_truth)
                else:
                    correct = None

                metrics_rows.append({
                    "question_number": question_number,
                    "question_id": entry.get("question_id", ""),
                    "db_id": entry["db_id"],
                    "difficulty": entry["difficulty"],
                    "num_documents": num_docs_processed,
                    "altered_sql": limited_sql,
                    "execution_time_seconds": round(execution_time_seconds, 4),
                    "llm_total_tokens": llm_tokens,
                    "llm_total_cost": round(llm_cost, 6),
                    "correct": correct,
                    "extracted": extracted_fmt,
                    "ground_truth": ground_truth,
                })
                continue

            def _doc_count(df):
                """Get document count for a dataframe."""
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return 0
                return len(df["contents"]) if "contents" in df.columns else len(df)

            def _get_df_value(df):
                """Get the value for a dataframe: _output > extraction > document count."""
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return 0
                if '_output' in df.columns and len(df) > 0:
                    val = df['_output'].iloc[0]
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return val
                if 'extraction' in df.columns and len(df) > 0:
                    val = df['extraction'].iloc[0]
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return val
                return _doc_count(df)

            final_result = []
            for group_idx, group_result in enumerate(execute_result):
                result = group_result["dfs"]
                operations = group_result["operations"]
                trunk_df = group_result.get("trunk_df")
                percent_count_denom_df = group_result.get("percent_count_denom_df")
                has_zero_doc_df = any(
                    _doc_count(df) == 0 for df in result if df is not None
                )
                op_failed = False
                prev_was_comparison = False
                group_values = []
                
                for idx, df in enumerate(result):
                    if df is not None:
                        doc_count = _doc_count(df)
                        print(f"Group {group_idx + 1} dataframe {idx} contains {doc_count} documents")
                        if '_output' in df.columns and len(df) > 0:
                            print(f"  has _output: {df['_output'].iloc[0]}")
                
                for op_idx, operation in enumerate(operations):
                    print(f"Executing OP (group {group_idx + 1}): {operation}")
                    if operation == "ratio":
                        try:
                            if len(result) == 2:
                                v0, v1 = _get_df_value(result[0]), _get_df_value(result[1])
                                if v1 == 0:
                                    group_values.append('None')
                                else:
                                    group_values.append(f"{v0} / {v1}")
                            else:
                                raise ValueError("ratio requires exactly two results")
                        except (ValueError, KeyError, ZeroDivisionError) as e:
                            print(f"Pipeline ratio operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    elif operation == "total":
                        try:
                            if len(result) == 1:
                                group_values.append(str(_doc_count(result[0])))
                            else:
                                raise ValueError("total requires exactly one result")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline total operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    elif operation in ("percent count", "percent sum"):
                        try:
                            if operation == "percent count" and len(result) == 1:
                                num = _get_df_value(result[0])
                                denom_df = percent_count_denom_df if percent_count_denom_df is not None else trunk_df
                                denom = _doc_count(denom_df) if denom_df is not None and not (isinstance(denom_df, pd.DataFrame) and denom_df.empty) else 0
                                if denom == 0 or num == 0:
                                    group_values.append("None")
                                else:
                                    percent = (num / denom) * 100
                                    group_values.append(f"{percent}")
                            elif len(result) == 2:
                                num, denom = _get_df_value(result[0]), _get_df_value(result[1])
                                if denom == 0 or num == 0:
                                    group_values.append("None")
                                else:
                                    percent = (num / denom) * 100
                                    group_values.append(f"{percent}")
                            else:
                                raise ValueError(f"{operation} requires one result (percent count) or two results (percent sum)")
                        except (ValueError, KeyError, ZeroDivisionError) as e:
                            print(f"Pipeline {operation} operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    elif operation == "percent":
                        try:
                            if len(result) == 2:
                                num, denom = _get_df_value(result[0]), _get_df_value(result[1])
                                if denom == 0:
                                    group_values.append("None")
                                else:
                                    percent = (num / denom) * 100
                                    group_values.append(f"{percent}")
                            else:
                                raise ValueError("percent requires exactly two results")
                        except (ValueError, KeyError, ZeroDivisionError) as e:
                            print(f"Pipeline percent operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    elif operation == "percent reverse":
                        try:
                            if len(result) == 2:
                                num, denom = _get_df_value(result[0]), _get_df_value(result[1])
                                if denom == 0 or num == 0:
                                    group_values.append("None")
                                else:
                                    percent = (num / denom) * 100
                                    group_values.append(f"{percent}")
                            else:
                                raise ValueError("percent reverse requires exactly two results")
                        except (ValueError, KeyError, ZeroDivisionError) as e:
                            print(f"Pipeline percent reverse operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    elif operation == "percent forward":
                        try:
                            if len(result) == 2:
                                num, denom = _get_df_value(result[0]), _get_df_value(result[1])
                                if denom == 0:
                                    group_values.append("None")
                                else:
                                    percent = (num / denom) * 100
                                    group_values.append(f"{percent}")
                            else:
                                raise ValueError("percent forward requires exactly two results")
                        except (ValueError, KeyError, ZeroDivisionError) as e:
                            print(f"Pipeline percent forward operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    elif operation == ">":
                        try:
                            if len(result) == 2:
                                v0, v1 = _get_df_value(result[0]), _get_df_value(result[1])
                                comparison = v0 > v1
                                group_values.append(str(comparison))
                                prev_was_comparison = True
                            else:
                                raise ValueError("> requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline > operation failed: {e}")
                            prev_was_comparison = False
                            op_failed = True
                    elif operation == "<":
                        try:
                            if len(result) == 2:
                                v0, v1 = _get_df_value(result[0]), _get_df_value(result[1])
                                comparison = v0 < v1
                                group_values.append(str(comparison))
                                prev_was_comparison = True
                            else:
                                raise ValueError("< requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline < operation failed: {e}")
                            prev_was_comparison = False
                            op_failed = True
                    elif operation == "bool":
                        try:
                            if len(result) >= 1:
                                has_docs = _doc_count(result[0]) != 0
                                group_values.append("True" if has_docs else "False")
                                prev_was_comparison = True
                            else:
                                raise ValueError("bool requires at least one result")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline bool operation failed: {e}")
                            prev_was_comparison = False
                            op_failed = True
                    elif operation == "-":
                        try:
                            if len(result) == 2:
                                v0, v1 = _get_df_value(result[0]), _get_df_value(result[1])
                                difference = v0 - v1
                                group_values.append(f"{difference}")
                            else:
                                raise ValueError("- requires exactly two results")
                        except (ValueError, KeyError) as e:
                            print(f"Pipeline - operation failed: {e}")
                            op_failed = True
                        prev_was_comparison = False
                    else:
                        prev_was_comparison = False

                # if no OP produced a result for this group, check value columns
                if not group_values:
                    for df_item in result:
                        if df_item is None:
                            continue
                        if '_output' in df_item.columns and len(df_item) > 0:
                            val = df_item['_output'].iloc[0]
                            if val is not None:
                                group_values.append(str(val))
                                break
                        elif 'extraction' in df_item.columns and len(df_item) > 0:
                            val = df_item['extraction'].iloc[0]
                            if val is not None:
                                group_values.append(str(val))
                                break
                
                if op_failed and not group_values:
                    group_values.append("None")
                if not group_values and has_zero_doc_df:
                    group_values.append("None")
                
                final_result.extend(group_values)

            # Aggregate into tuple for multi-answer; single value as 1-tuple for consistency
            extracted = tuple(final_result) if final_result else ()
            extracted_fmt = "[" + ", ".join(str(x) for x in extracted) + "]" if extracted else "[]"

            ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
            if DB_PATH.exists() and limited_sql:
                print(f"\nExtracted: {extracted_fmt}")
                print(f"Ground Truth: {ground_truth}")
                correct = _answers_match(extracted, ground_truth)
            else:
                correct = None

            metrics_rows.append({
                "question_number": question_number,
                "question_id": entry.get("question_id", ""),
                "db_id": entry["db_id"],
                "difficulty": entry["difficulty"],
                "num_documents": num_docs_processed,
                "altered_sql": limited_sql,
                "execution_time_seconds": round(execution_time_seconds, 4),
                "llm_total_tokens": llm_tokens,
                "llm_total_cost": round(llm_cost, 6),
                "correct": correct,
                "extracted": extracted_fmt,
                "ground_truth": ground_truth,
            })

        from rosetta_env import evaluation_results_dir

        results_dir = evaluation_results_dir(
            "lotus", args.null_param, args.noise_param, num_documents
        )
        out_path = aggregate_metrics_to_csv(
            metrics_rows,
            output_dir=str(results_dir),
            num_skipped_unsupported=num_skipped_unsupported,
        )
        print(f"\nMetrics written to {out_path}")
        print(f"Results directory: {results_dir}")
        if metrics_rows:
            lotus.settings.lm.print_total_usage()