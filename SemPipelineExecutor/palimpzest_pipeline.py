"""
Palimpzest pipeline aligned with lotus_pipeline: && and / split logic,
trunk+branch execution, metric tracking, correctness comparison, CSV export.
Uses palimpzest operators: sem_filter, sem_map (extract), sem_join, sem_agg, sem_topk.
"""
import palimpzest as pz
import copy
import os
import re
import json
import sqlite3
import argparse
import time
import csv
import tempfile
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from rosetta_env import load_rosetta_env

load_rosetta_env()

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
    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        pass

    @abstractmethod
    def modifies_docset(self) -> bool:
        pass


class FilterOperation(Operation):
    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        return dataset.sem_filter(
            self.instruction,
            depends_on=["contents"],
        )

    def modifies_docset(self) -> bool:
        return True


class ExtractOperation(Operation):
    _extract_counter = 0

    def __init__(self, instruction: str):
        super().__init__(instruction)
        ExtractOperation._extract_counter += 1
        self._col_name = f"extraction_{ExtractOperation._extract_counter}"
        self.output_cols = [
            {
                "name": self._col_name,
                "type": str,
                "desc": f"Extract ONLY: {instruction}. Return just the value, nothing else.",
            }
        ]

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        return dataset.sem_map(self.output_cols, depends_on=["contents"])

    def modifies_docset(self) -> bool:
        return False


class UnsupportedOperationError(Exception):
    """Raised when a pipeline operation is not supported by palimpzest."""
    pass


class RankOperation(Operation):
    def __init__(self, instruction: str, k: int = 1):
        super().__init__(instruction)
        self.k = k

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        raise UnsupportedOperationError("RANK (TopK) is not supported by palimpzest semantically on a document level")

    def modifies_docset(self) -> bool:
        return True


def _join_instruction_for_palimpzest(instruction: str) -> str:
    """Use natural-language instruction as-is; questions.json uses 'one document' / 'the other document'."""
    return instruction.strip()


class JoinOperation(Operation):
    def __init__(self, instruction: str, right_dataset: pz.Dataset = None):
        super().__init__(instruction)
        self.right_dataset = right_dataset

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        if self.right_dataset is not None:
            instr = _join_instruction_for_palimpzest(self.instruction)
            return dataset.sem_join(self.right_dataset, instr)
        return dataset

    def modifies_docset(self) -> bool:
        return True


class AggregateOperation(Operation):
    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        return dataset.sem_agg(
            col={
                "name": "aggregate",
                "type": str,
                "desc": self.instruction,
            },
            agg=self.instruction,
            depends_on=["contents"],
        )

    def modifies_docset(self) -> bool:
        return False


class GroupOperation(Operation):
    def __init__(self, instruction: str, group_by: str = None):
        super().__init__(instruction)
        self.group_by = group_by

    def execute(self, dataset: pz.Dataset) -> pz.Dataset:
        raise UnsupportedOperationError("GROUP is not supported by palimpzest semantically on a document level")

    def modifies_docset(self) -> bool:
        return True


#build data_path per table
class DocumentManager:
    def __init__(self, data_root: str, table_names: list, max_docs: int = 0):
        self.data_root = Path(data_root)
        self.table_names = table_names
        self.max_docs = max_docs
        self.data_paths = []
        for table in self.table_names:
            database_name = database_tables.get(table)
            if database_name:
                self.data_paths.append(self.data_root / database_name / table)
        for path in self.data_paths:
            path.mkdir(parents=True, exist_ok=True)

    def load_documents(self, source_path: str = None) -> list:
        """Load documents as list of dicts, respecting max_docs limit.
        
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
            with open(txt_file, "r", encoding="utf-8") as f:
                documents.append(
                    {"filename": txt_file.name, "contents": f.read(), "filepath": str(txt_file)}
                )
            if self.max_docs > 0 and len(documents) >= self.max_docs:
                break
        return documents

    def load_dataset(self, source_path: str) -> pz.Dataset:
        """Load a pz.Dataset, respecting max_docs limit.
        
        If max_docs is set, writes limited docs to a temp dir and loads from there.
        Otherwise loads directly from the source path.
        """
        path = Path(source_path)
        table_name = path.name
        
        if self.max_docs > 0:
            docs = self.load_documents(str(source_path))
            temp_dir = Path(tempfile.mkdtemp(prefix=f"pz_{table_name}_"))
            for doc in docs:
                (temp_dir / doc["filename"]).write_text(doc["contents"], encoding="utf-8")
            return pz.TextFileDataset(id=f"documents-{table_name}", path=str(temp_dir))
        
        return pz.TextFileDataset(id=f"documents-{table_name}", path=str(path))

    def load_datasets(self) -> list:
        return [self.load_dataset(p) for p in self.data_paths]


def _output_to_df(output) -> pd.DataFrame:
    """Convert palimpzest run output to DataFrame (records with attributes)."""
    try:
        return output.to_df()
    except Exception:
        records = []
        for r in output:
            row = {}
            for attr in dir(r):
                if not attr.startswith("_"):
                    try:
                        v = getattr(r, attr)
                        if not callable(v):
                            row[attr] = v
                    except Exception:
                        pass
            records.append(row)
        return pd.DataFrame(records)


def _records_to_memory_vals(records) -> list:
    """Convert trunk output records to list of dicts for MemoryDataset."""
    vals = []
    for r in records:
        vals.append({
            "filename": getattr(r, "filename", ""),
            "contents": getattr(r, "contents", ""),
            "filepath": getattr(r, "filepath", ""),
        })
    return vals


MEMORY_SCHEMA = [
    {"name": "filename", "type": str, "desc": "Filename"},
    {"name": "contents", "type": str, "desc": "Document contents"},
    {"name": "filepath", "type": str, "desc": "File path"},
]


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

    def _parse_ops_from_segment(self, segment_str: str, collect_ops: bool = True, ops_target: list | None = None) -> list:
        """Parse a segment ('TYPE - instruction' separated by &&) into list of Operation objects.
        If collect_ops=True, appends OP:... to ops_target or self.operations.
        """
        operations = []
        if not segment_str or not isinstance(segment_str, str):
            return operations
        target = ops_target if ops_target is not None else self.operations
        
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
                elif op_type == "OP" and collect_ops:
                    for op in instruction.split(","):
                        target.append(op.strip())
            elif collect_ops and part.strip().startswith("OP:"):
                op_part = part.strip()[3:].strip()
                for op in op_part.split(","):
                    target.append(op.strip())
        return operations

    def _parse_single_pipeline(self, format_str: str) -> dict:
        """Parse one pipeline segment. Returns {paths, operations}."""
        ops_for_this = []
        format_str = format_str.strip()
        
        if " / " not in format_str:
            trunk_ops = self._parse_ops_from_segment(format_str, collect_ops=True, ops_target=ops_for_this)
            return {"paths": [trunk_ops], "operations": ops_for_this}
        
        first_split_idx = format_str.find(" / ")
        trunk_str = format_str[:first_split_idx].strip()
        branches_str = format_str[first_split_idx + 3:].strip()
        
        trunk_ops = self._parse_ops_from_segment(trunk_str, collect_ops=False)
        branches = []
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
                            branch_ops = self._parse_ops_from_segment(parts[0].strip(), collect_ops=False)
                            if branch_ops:
                                branches.append(branch_ops)
                        if len(parts) > 1:
                            for op in parts[1].split(","):
                                ops_for_this.append(op.strip())
                else:
                    branch_ops = self._parse_ops_from_segment(remaining, collect_ops=False)
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
                branch_ops = self._parse_ops_from_segment(branch_content, collect_ops=False)
                branches.append(branch_ops)
            else:
                branches.append([])
        
        paths = []
        if not branches:
            paths.append(trunk_ops)
        else:
            for branch_ops in branches:
                path = list(trunk_ops) + list(branch_ops)
                paths.append(path)
            if "percent count" in ops_for_this and len(branches) == 1:
                paths.append(list(trunk_ops))
        
        return {"paths": paths, "operations": ops_for_this}

    def parse_format(self, format_str: str):
        """
        Parse format string. Multi-pipeline: ' |-| ' separates pipelines (one per SELECT expression).
        Each pipeline: trunk / branch1 | / branch2 | && OP: post_ops (DFS-style paths).
        """
        self.pipelines = []
        self.pipeline_groups = []
        self.operations = []
        
        format_str = format_str.strip()
        if not format_str:
            return
        
        segments = [s.strip() for s in format_str.split(" |-| ") if s.strip()]
        if not segments:
            segments = [format_str]
        
        for seg in segments:
            group = self._parse_single_pipeline(seg)
            self.pipeline_groups.append(group)
            self.pipelines.append({"paths": group["paths"]})
        
        if len(self.pipeline_groups) == 1:
            self.operations = self.pipeline_groups[0]["operations"]
        
        total = sum(len(g["paths"]) for g in self.pipeline_groups)
        self.log(f"Parsed {len(self.pipeline_groups)} pipeline(s), {total} total paths")

    def execute(self, run_mode: str = "max_quality") -> list:
        """Execute pipelines. Multi-pipeline: returns list of {dfs, operations} per group."""
        run_kw = {"max_quality": True} if run_mode == "max_quality" else {"min_cost": True}
        all_group_results = []
        self.total_pz_time = 0.0
        self.total_pz_cost = 0.0
        self.total_pz_tokens = 0
        self._seen_pz_op_ids: set[str] = set()
        datasets = self.doc_manager.load_datasets()
        total_docs = sum(len(list(Path(p).glob("*.txt"))) for p in self.doc_manager.data_paths)
        self.results = [{"operation": "initial", "doc_count": total_docs, "tables": len(datasets)}]

        pipeline_groups = getattr(self, "pipeline_groups", None)
        if not pipeline_groups:
            pipeline_groups = [{"paths": p["paths"], "operations": self.operations} for p in self.pipelines]

        for group_idx, group in enumerate(pipeline_groups):
            paths = group["paths"]
            group_dfs = []

            for path_idx, path_ops in enumerate(paths):
                path_ops = [op for op in path_ops if type(op).__name__ != "GroupOperation"]
                path_ops = [copy.deepcopy(op) for op in path_ops]

                if not path_ops:
                    self.log(f"Group {group_idx + 1} path {path_idx + 1}: no ops, skipping")
                    continue

                join_idx = 0
                while join_idx < len(path_ops) and type(path_ops[join_idx]).__name__ == "JoinOperation":
                    join_idx += 1
                join_ops = path_ops[:join_idx]
                rest_ops = path_ops[join_idx:]

                current = None
                if not datasets:
                    self.log("No datasets to load")
                    continue
                
                if not join_ops:
                    current = datasets[0]
                    dataset_list = datasets[1:]
                else:
                    current = datasets[0]
                    dataset_list = datasets[1:]
                    for i, join_op in enumerate(join_ops):
                        if i >= len(dataset_list):
                            break
                        join_op.right_dataset = dataset_list[i]
                        self.log(f"Path {path_idx + 1} JoinOperation: {join_op.instruction}")
                        current = join_op.execute(current)
                        self.results.append({
                            "operation": "JoinOperation",
                            "instruction": join_op.instruction,
                            "modifies_docset": True,
                        })

                if current is None:
                    continue

                for op in rest_ops:
                    op_name = type(op).__name__
                    self.log(f"Path {path_idx + 1} {op_name}: {op.instruction}")
                    current = op.execute(current)
                    self.results.append({
                        "operation": op_name,
                        "instruction": op.instruction,
                        "modifies_docset": op.modifies_docset(),
                    })

                self.log(f"Running group {group_idx + 1} path {path_idx + 1} (DFS - complete path)...")
                output = current.run(**run_kw)
                df = _output_to_df(output)
                group_dfs.append(df)
                if self.results:
                    self.results[-1]["doc_count"] = len(df) if not df.empty else 0
                self.log(f"Path {path_idx + 1} complete - {len(df)} rows")

                # Accumulate execution stats per-operator, deduplicating trunk ops.
                # For a split pipeline, path 1 and path 2 both re-execute the same
                # trunk operators (for now, optimization req). Since unique_full_op_id is a deterministic content
                # hash that is identical for trunk ops across all paths, we skip any
                # operator we've already counted.
                es = getattr(output, "execution_stats", None)
                if es is not None:
                    for plan_stats in es.plan_stats.values():
                        for unique_op_id, op_stats in plan_stats.operator_stats.items():
                            if unique_op_id not in self._seen_pz_op_ids:
                                self._seen_pz_op_ids.add(unique_op_id)
                                self.total_pz_time += op_stats.total_op_time
                                self.total_pz_cost += op_stats.total_op_cost
                                self.total_pz_tokens += (
                                    op_stats.total_input_tokens + op_stats.total_output_tokens
                                )

            original_doc_count = total_docs
            all_group_results.append({
                "dfs": group_dfs,
                "operations": group["operations"],
                "original_doc_count": original_doc_count,
            })

        return all_group_results


from pipeline_sources import extract_tables_from_sql


# ---------------------------------------------------------------------------
# Semantic plan → palimpzest Dataset chain executor
# ---------------------------------------------------------------------------

_UNSUPPORTED_PZ_OPS = {"sem_topk", "sem_cluster_by"}


def _apply_pz_step(
    step: PlannedStep,
    dataset: pz.Dataset,
    ctx: SubqueryContext,
    right_dataset: pz.Dataset | None = None,
    verbose: bool = False,
    post_join: bool = False,
) -> pz.Dataset:
    """Apply a single PlannedStep to a palimpzest Dataset.
    
    Args:
        post_join: If True, the dataset has been through a join and has 
                   'contents_right' column that should be included in depends_on.
    """
    instr = step.instruction
    if ctx:
        instr = ctx.substitute(instr)
    kind = step.kind

    if kind == "sem_filter":
        # After joins, include contents_right in depends_on
        depends_cols = ["contents"]
        if post_join:
            depends_cols.append("contents_right")
        
        if verbose:
            print(f"    pz.sem_filter:")
            print(f"      Instruction: \"{instr}\"")
            print(f"      depends_on: {depends_cols}")
        
        return dataset.sem_filter(instr, depends_on=depends_cols)

    if kind == "sem_join":
        if right_dataset is not None:
            if verbose:
                print(f"    pz.sem_join:")
                print(f"      Instruction: \"{instr}\"")
            return dataset.sem_join(right_dataset, instr)
        return dataset

    if kind == "sem_agg":
        # After joins, include contents_right in depends_on
        depends_cols = ["contents"]
        if post_join:
            depends_cols.append("contents_right")
        
        if verbose:
            print(f"    pz.sem_agg:")
            print(f"      Instruction: \"{instr}\"")
            print(f"      Output column: {{'name': 'aggregate', 'type': str, 'desc': <instruction>}}")
            print(f"      depends_on: {depends_cols}")
            # Debug: show what documents the aggregation will work on
            try:
                pre_agg_output = dataset.run(max_quality=True)
                pre_agg_df = _output_to_df(pre_agg_output)
                print(f"      [PRE-AGG] {len(pre_agg_df)} input docs, cols: {list(pre_agg_df.columns)}")
                if len(pre_agg_df) > 0 and "contents" in pre_agg_df.columns:
                    sample = str(pre_agg_df["contents"].iloc[0])[:400]
                    print(f"      [CONTEXT-LEFT] {sample}...")
                if len(pre_agg_df) > 0 and "contents_right" in pre_agg_df.columns:
                    sample_right = str(pre_agg_df["contents_right"].iloc[0])[:400]
                    print(f"      [CONTEXT-RIGHT] {sample_right}...")
            except Exception as e:
                print(f"      [DEBUG] Could not preview pre-agg: {e}")
        
        return dataset.sem_agg(
            col={"name": "aggregate", "type": str, "desc": instr},
            agg=instr,
            depends_on=depends_cols,
        )

    if kind in ("sem_extract", "sem_map"):
        col_name = f"extraction_{id(step) % 10000}"
        
        # After joins, include contents_right in depends_on
        depends_cols = ["contents"]
        if post_join:
            depends_cols.append("contents_right")
        
        if verbose:
            print(f"    pz.sem_map (extract):")
            print(f"      Instruction: \"{instr}\"")
            print(f"      Output: {{'name': '{col_name}', 'desc': 'Extract ONLY: <instruction>. Return just the value.'}}")
            print(f"      depends_on: {depends_cols}")
        
        return dataset.sem_map(
            [{"name": col_name, "type": str, "desc": f"Extract ONLY: {instr}. Return just the value."}],
            depends_on=depends_cols,
        )

    if kind in _UNSUPPORTED_PZ_OPS:
        raise UnsupportedOperationError(
            f"{kind} is not supported by palimpzest semantically on a document level"
        )

    if verbose:
        print(f"    pz.passthrough ({kind}): {instr}")
    return dataset


def _run_subquery_pz(
    sq_plan: dict,
    entry: dict,
    doc_manager: "DocumentManager",
    ctx: SubqueryContext,
    verbose: bool = False,
) -> object:
    """Run a subquery pipeline to completion via palimpzest, returning scalar."""
    steps_list: list[PlannedStep] = sq_plan["steps"]
    dataset = doc_manager.load_dataset(doc_manager.data_paths[0])

    for step in steps_list:
        if step.kind == "subquery_result":
            continue
        dataset = _apply_pz_step(step, dataset, ctx, verbose=verbose)

    output = dataset.run(max_quality=True)
    df = _output_to_df(output)
    if "aggregate" in df.columns and len(df) > 0:
        return df["aggregate"].iloc[0]
    ext_cols = [c for c in df.columns if c.startswith("extraction")]
    if ext_cols and len(df) > 0:
        return df[ext_cols[-1]].iloc[0]
    return len(df)


def _plan_contains_unsupported_op(plan: dict) -> str | None:
    """Check if plan contains any unsupported operations. Returns op name or None."""
    unsupported = {"sem_cluster_by", "sem_topk"}
    
    main_steps = plan.get("main_steps", [])
    for step in main_steps:
        if step.kind in unsupported:
            return step.kind
    
    for sq_plan in plan.get("subquery_plans", []):
        for step in sq_plan.get("steps", []):
            if hasattr(step, 'kind') and step.kind in unsupported:
                return step.kind
    
    return None


def execute_semantic_plan_pz(
    plan: dict,
    entry: dict,
    doc_manager: "DocumentManager",
    verbose: bool = False,
) -> tuple:
    """Execute a semantic plan using palimpzest Dataset operators.

    Runs subqueries first, then builds the main pipeline as a chain of
    Dataset operators with multi-head support via sem_join.
    """
    # Early check for unsupported operations
    unsupported_op = _plan_contains_unsupported_op(plan)
    if unsupported_op:
        raise UnsupportedOperationError(
            f"{unsupported_op} is not supported by palimpzest - skipping query"
        )
    
    ctx = SubqueryContext()
    enrich_plan_instructions(plan, entry, ctx)

    # 1. Subqueries
    for sq_plan in plan.get("subquery_plans", []):
        var = sq_plan["pipeline"].get("subquery_var", "")
        val = _run_subquery_pz(sq_plan, entry, doc_manager, ctx, verbose)
        if var:
            ctx.bind(var, val)
            if verbose:
                print(f"  Bound {var} = {val}")

    enrich_plan_instructions(plan, entry, ctx)

    # 2. Main pipeline
    tables = plan.get("tables", [])
    main_steps: list[PlannedStep] = plan.get("main_steps", [])

    heads: dict[int, pz.Dataset] = {}
    for idx, table in enumerate(tables):
        if idx < len(doc_manager.data_paths):
            heads[idx] = doc_manager.load_dataset(doc_manager.data_paths[idx])

    if not heads:
        return ()

    # Track which heads have been joined (contain contents_right)
    joined_heads: set[int] = set()

    for step in main_steps:
        kind = step.kind

        if kind == "subquery_result":
            continue

        if kind == "sem_join" and step.merge_pair:
            left_id, right_id = step.merge_pair
            left_ds = heads.get(left_id)
            right_ds = heads.get(right_id)
            if left_ds is not None and right_ds is not None:
                heads[left_id] = _apply_pz_step(
                    step, left_ds, ctx, right_dataset=right_ds, verbose=verbose)
                heads.pop(right_id, None)
                # Mark left_id as having been joined
                joined_heads.add(left_id)
            continue

        target_ids = step.head_ids if step.head_ids else list(heads.keys())
        for hid in target_ids:
            if hid not in heads:
                continue
            # Pass post_join=True if this head has been through a join
            heads[hid] = _apply_pz_step(
                step, heads[hid], ctx, verbose=verbose, post_join=(hid in joined_heads)
            )

    # Run the final dataset
    final_ds = list(heads.values())[0] if heads else None
    if final_ds is None:
        return ()

    output = final_ds.run(max_quality=True)
    df = _output_to_df(output)

    if verbose:
        print(f"\n--- Final DataFrame Debug ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        if len(df) > 0:
            print(f"First row:\n{df.iloc[0].to_dict()}")
        print(f"Full DataFrame:\n{df.to_string(max_colwidth=80)}")
        print(f"--- End Debug ---\n")

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

    values: list = []
    
    # If DataFrame is empty, return empty result (equivalent to [None])
    if df.empty:
        return ()
    
    if "aggregate" in df.columns and len(df) > 0:
        for v in df["aggregate"]:
            if v is not None:
                if _is_divide_by_zero_ratio(v):
                    continue  # Treat as None
                values.append(v)
    else:
        ext_cols = [c for c in df.columns if c.startswith("extraction")]
        if ext_cols and len(df) > 0:
            for v in df[ext_cols[-1]]:
                if v is not None:
                    if _is_divide_by_zero_ratio(v):
                        continue
                    values.append(v)
    # No else - if no output columns, values stays empty

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
        if c == "(":
            depth += 1
            out.append(c)
            i += 1
            continue
        if c == ")":
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
        for kw in ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FROM", "JOIN"]:
            parts = kw.split()
            j = i
            for p in parts:
                while j < n and sql[j] in " \t":
                    j += 1
                if j + len(p) > n or sql[j : j + len(p)].upper() != p.upper():
                    break
                j += len(p)
            else:
                if j < n and (sql[j].isalnum() or sql[j] == "_"):
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
        while i < n and sql[i] in " \t":
            i += 1
        table_start = i
        while i < n and (sql[i].isalnum() or sql[i] == "_"):
            i += 1
        table_name = sql[table_start:i]
        if table_name.lower() not in tables_set:
            out.append(sql[table_start:i])
            while i < n and sql[i] in " \t":
                i += 1
            if i + 2 <= n and sql[i : i + 2].upper() == "AS":
                i += 2
                while i < n and sql[i] in " \t":
                    i += 1
                while i < n and (sql[i].isalnum() or sql[i] == "_"):
                    i += 1
            continue
        alias = table_name
        while i < n and sql[i] in " \t":
            i += 1
        if i + 2 <= n and sql[i : i + 2].upper() == "AS":
            i += 2
            while i < n and sql[i] in " \t":
                i += 1
            alias_start = i
            while i < n and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            alias = sql[alias_start:i]
        subq = f"(SELECT * FROM {table_name} ORDER BY ROWID LIMIT {num_documents}) AS {alias} "
        out.append(subq)
    return "".join(out)


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
    sql_results, _ = execute_sql(db_path, sql, database_name)
    ground_truth = [row[0] if len(row) == 1 else row for row in sql_results]
    print(f"\nExtracted: {final_result}")
    print(f"Ground Truth: {ground_truth}")
    return final_result, ground_truth


# Pipeline run configuration
ALLOWED_DB_IDS = None  # None means all databases
ALLOWED_DIFFICULTIES = {"simple", "moderate", "challenging"}


def _normalize_for_compare(val) -> str:
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
    if not Path(db_path).exists():
        return []
    try:
        results, _ = execute_sql(db_path, sql, database_name)
        return [row[0] if len(row) == 1 else row for row in results]
    except Exception:
        return []


def aggregate_metrics_to_csv(rows: list, output_dir: str = "./pipeline_data/results/palimpzest", num_skipped_unsupported: int = 0) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "palimpzest_metrics.csv"
    fieldnames = [
        "question_number", "question_id", "db_id", "difficulty", "num_documents",
        "altered_sql", "execution_time_seconds", "pz_execution_time_seconds",
        "llm_total_tokens", "llm_total_cost",
        "correct", "extracted", "ground_truth",
    ]
    if not rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return str(filepath)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_export = {k: r.get(k) for k in fieldnames}
            row_export["extracted"] = str(r.get("extracted", ""))[:500]
            row_export["ground_truth"] = str(r.get("ground_truth", ""))[:500]
            writer.writerow(row_export)
    total_time = sum(r.get("execution_time_seconds") or 0 for r in rows)
    total_pz_time = sum(r.get("pz_execution_time_seconds") or 0 for r in rows)
    total_tokens = sum(r.get("llm_total_tokens") or 0 for r in rows)
    total_cost = sum(r.get("llm_total_cost") or 0 for r in rows)
    num_correct = sum(1 for r in rows if r.get("correct") is True)
    summary_path = output_path / "palimpzest_metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_questions", len(rows)])
        w.writerow(["total_wall_time_seconds", round(total_time, 4)])
        w.writerow(["total_pz_execution_time_seconds", round(total_pz_time, 4)])
        w.writerow(["total_llm_tokens", total_tokens])
        w.writerow(["total_llm_cost", round(total_cost, 6)])
        w.writerow(["num_correct", num_correct])
        w.writerow(["accuracy", round(num_correct / len(rows), 4) if rows else 0])
        w.writerow(["num_skipped_unsupported", num_skipped_unsupported])
        w.writerow(["unsupported_operations", "RANK (TopK), GROUP"])
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
    parser.add_argument("--num_documents", type=int, default=10)
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

        num_skipped_unsupported = 0

        print(f"[palimpzest] Running {len(filtered_entries)} entries | data_root={data_root}")

        metrics_rows = []
        for question_number, entry in enumerate(filtered_entries, start=1):
            tables_list = entry["tables"].split(", ")
            database_name = database_tables.get(tables_list[0])
            if not database_name:
                continue
            doc_manager = DocumentManager(str(data_root), tables_list, max_docs=num_documents)
            ExtractOperation._extract_counter = 0

            sem_plan = prepare_semantic_execution(entry) if use_semantic else None

            start_time = time.perf_counter()
            execute_result = None
            semantic_extracted = None
            try:
                if sem_plan:
                    semantic_extracted = execute_semantic_plan_pz(
                        sem_plan, entry, doc_manager, verbose=False)
                else:
                    sequence = entry["format"]
                    pipeline = Pipeline(doc_manager, verbose=False)
                    pipeline.parse_format(sequence)
                    execute_result = pipeline.execute(run_mode="max_quality")
            except UnsupportedOperationError as e:
                # Gracefully skip unsupported operations without full traceback
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
                    "pz_execution_time_seconds": 0.0,
                    "llm_total_tokens": 0,
                    "llm_total_cost": 0.0,
                    "correct": False,
                    "extracted": [],
                    "ground_truth": ground_truth,
                    "skipped": True,
                    "skip_reason": str(e),
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
                pz_time = 0.0
                try:
                    if not sem_plan:
                        pz_time = pipeline.total_pz_time
                except NameError:
                    pass
                metrics_rows.append({
                    "question_number": question_number,
                    "question_id": entry.get("question_id", ""),
                    "db_id": entry["db_id"],
                    "difficulty": entry["difficulty"],
                    "num_documents": 0,
                    "altered_sql": limited_sql,
                    "execution_time_seconds": round(execution_time_seconds, 4),
                    "pz_execution_time_seconds": round(pz_time, 4),
                    "llm_total_tokens": 0,
                    "llm_total_cost": 0.0,
                    "correct": False,
                    "extracted": [],
                    "ground_truth": ground_truth,
                })
                continue

            execution_time_seconds = time.perf_counter() - start_time

            if sem_plan:
                llm_tokens = 0
                llm_cost = 0.0
                pz_execution_time = execution_time_seconds
                num_docs_processed = sum(
                    len(list(Path(p).glob("*.txt"))) for p in doc_manager.data_paths
                )
            else:
                llm_tokens = pipeline.total_pz_tokens
                llm_cost = pipeline.total_pz_cost
                pz_execution_time = pipeline.total_pz_time
                num_docs_processed = pipeline.results[0]["doc_count"] if pipeline.results else 0
            gt_sql = entry.get("sql_sqlite") or entry["SQL"]
            limited_sql = limit_sql_to_num_documents(gt_sql, num_documents)

            if sem_plan:
                extracted = semantic_extracted
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
                    "pz_execution_time_seconds": round(pz_execution_time, 4),
                    "llm_total_tokens": llm_tokens,
                    "llm_total_cost": round(llm_cost, 6),
                    "correct": correct,
                    "extracted": extracted_fmt,
                    "ground_truth": ground_truth,
                })
                continue

            def _doc_count(df):
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return 0
                return len(df["contents"]) if "contents" in df.columns else len(df)

            def _get_df_value(df):
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return 0
                if 'aggregate' in df.columns and len(df) > 0:
                    val = df['aggregate'].iloc[0]
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return val
                ext_cols = [c for c in df.columns if c.startswith("extraction")]
                if ext_cols and len(df) > 0:
                    val = df[ext_cols[-1]].iloc[0]
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
                has_zero_doc_df = any(_doc_count(df) == 0 for df in result if df is not None)
                op_failed = False
                prev_was_comparison = False
                group_values = []
                
                for idx, df in enumerate(result):
                    if df is not None:
                        doc_count = _doc_count(df)
                        print(f"Group {group_idx + 1} dataframe {idx} contains {doc_count} documents")
                        if 'aggregate' in df.columns and len(df) > 0:
                            print(f"  has aggregate: {df['aggregate'].iloc[0]}")
                
                for operation in operations:
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
                                denom = original_doc_count
                                if denom == 0 or num == 0:
                                    group_values.append("None")
                                else:
                                    percent = (num / denom) * 100
                                    group_values.append(f"{percent}")
                            elif operation == "percent count" and len(result) == 2:
                                num = _get_df_value(result[0])
                                denom = _doc_count(result[1])
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
                                raise ValueError(f"{operation} requires two results")
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
                    elif operation == "total percent":
                        continue
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

                if not group_values:
                    for df_item in result:
                        if df_item is None:
                            continue
                        if 'aggregate' in df_item.columns and len(df_item) > 0:
                            val = df_item['aggregate'].iloc[0]
                            if val is not None:
                                group_values.append(str(val))
                                break
                        ext_cols = [c for c in df_item.columns if c.startswith("extraction")]
                        if ext_cols and len(df_item) > 0:
                            val = df_item[ext_cols[-1]].iloc[0]
                            if val is not None:
                                group_values.append(str(val))
                                break
                
                if op_failed and not group_values:
                    group_values.append("None")
                if not group_values and has_zero_doc_df:
                    group_values.append("None")
                
                final_result.extend(group_values)

            extracted = tuple(final_result) if final_result else ()

            ground_truth = _get_ground_truth(str(DB_PATH), database_name, limited_sql) if DB_PATH.exists() and limited_sql else []
            if DB_PATH.exists() and limited_sql:
                print(f"\nExtracted: {extracted}")
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
                "pz_execution_time_seconds": round(pz_execution_time, 4),
                "llm_total_tokens": llm_tokens,
                "llm_total_cost": round(llm_cost, 6),
                "correct": correct,
                "extracted": extracted,
                "ground_truth": ground_truth,
            })

        from rosetta_env import evaluation_results_dir

        results_dir = evaluation_results_dir(
            "palimpzest", args.null_param, args.noise_param, num_documents
        )
        out_path = aggregate_metrics_to_csv(
            metrics_rows,
            output_dir=str(results_dir),
            num_skipped_unsupported=num_skipped_unsupported,
        )
        print(f"\nMetrics written to {out_path}")
        print(f"Results directory: {results_dir}")
