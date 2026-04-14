"""
DocETL pipeline aligned with lotus_pipeline and palimpzest_pipeline:
&& and / split logic, trunk+branch execution (DFS-style), metric tracking,
correctness comparison, CSV export. Uses DocETL YAML pipeline format:
filter, extract, reduce (aggregate/rank/group), equijoin (resolve).
"""
import os
import re
import json
import sqlite3
import argparse
import time
import csv
import shutil
import subprocess
import sys
import tempfile
import yaml
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


def _escape_for_docetl(text: str) -> str:
    """Escape literal curly braces in instruction text for Jinja2 templates.
    
    Only escapes standalone curly braces that are not part of Jinja2 syntax.
    This prevents template errors when instructions contain JSON-like content.
    """
    if not text:
        return text
    # Only escape curly braces that could cause issues
    # Don't escape if text doesn't contain problematic braces
    # For now, just return as-is since our prompts are controlled
    return text


class Operation(ABC):
    def __init__(self, instruction: str):
        self.instruction = _escape_for_docetl(instruction) if instruction else ""

    @abstractmethod
    def to_yaml_config(self, op_idx: int) -> dict:
        pass

    @abstractmethod
    def modifies_docset(self) -> bool:
        pass


class FilterOperation(Operation):
    def to_yaml_config(self, op_idx: int, post_join: bool = False, post_reduce: bool = False) -> dict:
        if post_reduce:
            # After reduce, reference the whole input object
            doc_ref = '"{{ input }}"'
        elif post_join:
            doc_ref = '"{{ input.contents_left }}" and "{{ input.contents_right }}"'
        else:
            doc_ref = '"{{ input.contents }}"'
        
        return {
            "name": f"op_{op_idx}_filter",
            "type": "filter",
            "prompt": f'Analyze the following document: {doc_ref}. If {self.instruction} within the document, respond with true',
            "output": {"schema": {"keep": "boolean"}},
        }

    def modifies_docset(self) -> bool:
        return True


class ExtractOperation(Operation):
    _extract_counter = 0

    def __init__(self, instruction: str):
        super().__init__(instruction)
        ExtractOperation._extract_counter += 1
        self._col_name = f"extraction_{ExtractOperation._extract_counter}"

    def to_yaml_config(self, op_idx: int, post_join: bool = False, post_reduce: bool = False) -> dict:
        if post_reduce:
            # After reduce, reference the whole input object
            doc_ref = "{{ input }}"
        elif post_join:
            doc_ref = "{{ input.contents_left }} | {{ input.contents_right }}"
        else:
            doc_ref = "{{ input.contents }}"
        
        return {
            "name": f"op_{op_idx}_extract",
            "type": "map",
            "prompt": f'From the document: {doc_ref}. Extract ONLY: {self.instruction}. Return just the value.',
            "output": {"schema": {"extracted_value": "str"}},
        }

    def modifies_docset(self) -> bool:
        return False


class RankOperation(Operation):
    def __init__(self, instruction: str, k: int = 1):
        super().__init__(instruction)
        self.k = k

    def to_yaml_config(self, op_idx: int, post_join: bool = False, post_reduce: bool = False) -> dict:
        if post_reduce:
            # After another reduce, just show the item
            doc_content = "{{ item }}"
        elif post_join:
            doc_content = "{{ item.contents_left }} | {{ item.contents_right }}"
        else:
            doc_content = "{{ item.contents }}"
        
        return {
            "name": f"op_{op_idx}_rank",
            "type": "reduce",
            "reduce_key": "_all",
            "prompt": f'Rank and select top {self.k} by: {self.instruction}. Documents: {{% for item in inputs %}}[{doc_content}]{{% endfor %}}. Return ONLY the identifier or value.',
            "output": {"schema": {"ranked_result": "str"}},
            "pass_through": False,
        }

    def modifies_docset(self) -> bool:
        return True


class JoinOperation(Operation):
    def to_yaml_config(self, op_idx: int) -> dict:
        # Note: equijoin uses Jinja2 {{ left.field }} and {{ right.field }} syntax
        return {
            "name": f"op_{op_idx}_join",
            "type": "equijoin",
            "comparison_prompt": f"""Compare these two documents:

Left document: {{{{ left.contents }}}}

Right document: {{{{ right.contents }}}}

Join condition: {self.instruction}

Answer "True" if these documents should be joined based on the condition, "False" otherwise.""",
            "blocking_keys": {"left": ["contents"], "right": ["contents"]},
        }

    def modifies_docset(self) -> bool:
        return True


class AggregateOperation(Operation):
    def to_yaml_config(self, op_idx: int, post_join: bool = False, post_reduce: bool = False) -> dict:
        if post_reduce:
            # After another reduce, just show the item
            doc_content = "{{ item }}"
        elif post_join:
            doc_content = "{{ item.contents_left }} | {{ item.contents_right }}"
        else:
            doc_content = "{{ item.contents }}"
        
        return {
            "name": f"op_{op_idx}_aggregate",
            "type": "reduce",
            "reduce_key": "_all",
            "prompt": f'{self.instruction}. Documents: {{% for item in inputs %}}[{doc_content}]{{% endfor %}}. Return ONLY the numeric result.',
            "output": {"schema": {"aggregated_result": "str"}},
        }

    def modifies_docset(self) -> bool:
        return False


class GroupOperation(Operation):
    def __init__(self, instruction: str, group_by: str = None):
        super().__init__(instruction)
        self.group_by = group_by or "contents"

    def to_yaml_config(self, op_idx: int, post_join: bool = False, post_reduce: bool = False) -> dict:
        if post_reduce:
            doc_content = "{{ item }}"
            reduce_key = "_all"
        elif post_join:
            doc_content = "{{ item.contents_left }} | {{ item.contents_right }}"
            reduce_key = "_all"
        else:
            doc_content = "{{ item.contents }}"
            reduce_key = [self.group_by] if isinstance(self.group_by, str) else self.group_by
        
        return {
            "name": f"op_{op_idx}_group",
            "type": "reduce",
            "reduce_key": reduce_key,
            "prompt": f'{self.instruction}. Group documents: {{% for item in inputs %}}[{doc_content}]{{% endfor %}}. Provide summary.',
            "output": {"schema": {"grouped_result": "str"}},
            "pass_through": True,
        }

    def modifies_docset(self) -> bool:
        return True


class UnsupportedOperationError(Exception):
    """Raised when a pipeline operation is not supported by DocETL."""
    pass


# Operations that DocETL cannot handle semantically
_UNSUPPORTED_DOCETL_OPS: set[str] = set()  # All operations now supported including TopK via RankOperation


def _plan_contains_unsupported_op(plan: dict) -> str | None:
    """Check if plan contains any unsupported operations. Returns op name or None."""
    unsupported = _UNSUPPORTED_DOCETL_OPS
    
    main_steps = plan.get("main_steps", [])
    for step in main_steps:
        if step.kind in unsupported:
            return step.kind
    
    for sq_plan in plan.get("subquery_plans", []):
        for step in sq_plan.get("steps", []):
            if hasattr(step, 'kind') and step.kind in unsupported:
                return step.kind
    
    return None


class DocumentManager:
    """Multi-table document manager like lotus/palimpzest."""

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

    def load_documents_as_df(self, source_path: str = None) -> pd.DataFrame:
        docs = self.load_documents(source_path)
        return pd.DataFrame(docs)

    def documents_to_json_list(self, documents: list) -> list:
        """Convert documents (list of dicts) to DocETL JSON format. Handles both load_documents output and DataFrame records."""
        out = []
        for i, d in enumerate(documents):
            d = dict(d)
            contents = d.get("contents")
            if contents is None:
                parts = [str(v) for k, v in d.items() if isinstance(v, str) and len(str(v)) > 20]
                contents = "\n\n".join(parts) if parts else json.dumps(d)
            out.append({
                "filename": d.get("filename", f"doc_{i}.txt"),
                "contents": contents,
                "filepath": d.get("filepath", ""),
            })
        return out


def _docetl_output_to_df(output_path: Path, last_op_modifies_docset: bool) -> pd.DataFrame:
    """Convert DocETL JSON output to DataFrame, normalizing column names for lotus compatibility."""
    if not output_path.exists():
        return pd.DataFrame()
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)

    # equijoin output may have left/right structure
    if "contents" not in df.columns:
        text_cols = [c for c in df.columns if "content" in c.lower() or "text" in c.lower()]
        if text_cols:
            df["contents"] = df[text_cols[0]].fillna("")
        else:
            df["contents"] = df.apply(lambda r: " ".join(str(v) for v in r.values if isinstance(v, str)), axis=1)

    extract_cols = [c for c in df.columns if "extracted" in c.lower() and "extract" in c.lower()]
    if extract_cols:
        df["extraction"] = df[extract_cols[-1]]
    elif "extracted_value" in df.columns:
        df["extraction"] = df["extracted_value"]

    if "aggregated_result" in df.columns:
        df["_output"] = df["aggregated_result"]
    elif "ranked_result" in df.columns:
        df["_output"] = df["ranked_result"]
    elif "grouped_result" in df.columns:
        df["_output"] = df["grouped_result"]

    if "keep" in df.columns:
        df = df[df["keep"] == True].copy()
        df = df.drop(columns=["keep"], errors="ignore")

    return df


def _run_docetl_pipeline(yaml_path: Path, cwd: Path, env: dict) -> tuple[subprocess.CompletedProcess, Path]:
    """Run docetl run <yaml> and return (result, output_path from pipeline config)."""
    docetl_cmd = None
    if shutil.which("docetl"):
        docetl_cmd = ["docetl", "run", str(yaml_path.name)]
    elif os.name == "nt" and shutil.which("docetl.exe"):
        docetl_cmd = ["docetl.exe", "run", str(yaml_path.name)]
    else:
        raise RuntimeError(
            "DocETL command not found. Install with: pip install docetl"
        )
    result = subprocess.run(
        docetl_cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    return result, cwd


class Pipeline:
    def __init__(self, doc_manager: DocumentManager, verbose: bool = False):
        self.doc_manager = doc_manager
        self.pipelines = []
        self.pipeline_groups = []
        self.operations = []
        self.results = []
        self.verbose = verbose

    def log(self, message: str):
        if self.verbose:
            print(f"{message}")

    def _parse_ops_from_segment(
        self, segment_str: str, collect_ops: list | None = None
    ) -> list:
        """Parse segment ('TYPE - instruction' separated by &&) into Operation objects."""
        operations = []
        if not segment_str or not isinstance(segment_str, str):
            return operations
        ops_target = collect_ops if collect_ops is not None else self.operations
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
        """Parse one pipeline segment. Returns {trunk, branches, operations} (lotus-style)."""
        ops_for_this = []
        trunk_ops = []
        branches = []
        format_str = format_str.strip()

        if " / " not in format_str:
            trunk_ops = self._parse_ops_from_segment(format_str, collect_ops=ops_for_this)
            return {"trunk": trunk_ops, "branches": [], "operations": ops_for_this}

        first_split_idx = format_str.find(" / ")
        trunk_str = format_str[:first_split_idx].strip()
        branches_str = format_str[first_split_idx + 3 :].strip()
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
            remaining = remaining[pipe_idx + 1 :].strip()
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
        """Parse format string. Multi-pipeline: |-| separates pipelines."""
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
            self.pipelines.append({"trunk": group["trunk"], "branches": group["branches"]})
        if len(self.pipeline_groups) == 1:
            self.operations = self.pipeline_groups[0]["operations"]
        total = sum(
            len(g["trunk"]) + sum(len(b) for b in g["branches"])
            for g in self.pipeline_groups
        )
        self.log(f"Parsed {len(self.pipeline_groups)} pipeline(s), {total} total ops")

    def _build_yaml_and_run(
        self,
        path_ops: list,
        table_docs: list[list],
        temp_dir: Path,
        env: dict,
    ) -> pd.DataFrame:
        """Build YAML for a single path, run DocETL, return DataFrame."""
        path_ops = [op for op in path_ops if type(op).__name__ != "GroupOperation"]
        if not path_ops:
            # No operations: return loaded documents as DataFrame (e.g. empty trunk with branches)
            docs = table_docs[0] if table_docs else []
            if isinstance(docs, pd.DataFrame):
                docs = docs.to_dict("records")
            json_list = self.doc_manager.documents_to_json_list(docs)
            df = pd.DataFrame(json_list)
            self.log(f"Pass-through (0 ops): returning {len(df)} documents to next stage")
            return df

        join_idx = 0
        while join_idx < len(path_ops) and type(path_ops[join_idx]).__name__ == "JoinOperation":
            join_idx += 1
        join_ops = path_ops[:join_idx]
        rest_ops = path_ops[join_idx:]

        datasets = {}
        steps = []
        prev_step = None

        # Handle joins first: need two datasets
        operations_config = []
        if join_ops and len(table_docs) >= 2:
            left_path = temp_dir / "input_data1.json"
            right_path = temp_dir / "input_data2.json"
            with open(left_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.doc_manager.documents_to_json_list(table_docs[0]), f, indent=2
                )
            with open(right_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.doc_manager.documents_to_json_list(table_docs[1]), f, indent=2
                )
            datasets["input_data1"] = {"type": "file", "path": str(left_path.name)}
            datasets["input_data2"] = {"type": "file", "path": str(right_path.name)}
            join_op = join_ops[0]
            join_cfg = join_op.to_yaml_config(0)
            operations_config.append(join_cfg)
            steps.append({
                "name": join_cfg["name"],
                "operations": [{join_cfg["name"]: {"left": "input_data1", "right": "input_data2"}}],
            })
            prev_step = join_cfg["name"]
            op_idx = 1
            ops_to_run = rest_ops
        else:
            # Single table
            input_path = temp_dir / "input_documents.json"
            docs = table_docs[0] if table_docs else []
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.doc_manager.documents_to_json_list(docs), f, indent=2
                )
            datasets["input_documents"] = {"type": "file", "path": input_path.name}
            prev_step = "input_documents"
            op_idx = 0
            ops_to_run = path_ops

        for op in ops_to_run:
            cfg = op.to_yaml_config(op_idx)
            operations_config.append(cfg)
            step = {
                "name": cfg["name"],
                "input": prev_step,
                "operations": [cfg["name"]],
            }
            steps.append(step)
            prev_step = cfg["name"]
            op_idx += 1

        output_path = temp_dir / "docetl_output.json"
        pipeline_config = {
            "datasets": datasets,
            "default_model": "gpt-4o-mini",
            "operations": operations_config,
            "pipeline": {
                "name": "semantic_pipeline",
                "steps": steps,
                "output": {"type": "file", "path": str(output_path.name)},
            },
        }
        yaml_path = temp_dir / "pipeline.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)

        self.log(f"Running DocETL pipeline ({len(operations_config)} ops)...")
        result, _ = _run_docetl_pipeline(yaml_path, temp_dir, env)
        if result.returncode != 0:
            self.log(f"DocETL failed: {result.stderr[:500]}")
            raise RuntimeError(f"DocETL failed: {result.stderr[:500]}")

        last_op = ops_to_run[-1] if ops_to_run else None
        last_modifies = last_op.modifies_docset() if last_op else True
        return _docetl_output_to_df(output_path, last_modifies)

    def execute(self) -> list:
        """Execute pipelines DFS-style. Returns list of {dfs, operations, trunk_df, percent_count_denom_df} per group."""
        all_group_results = []
        pipeline_groups = getattr(self, "pipeline_groups", None)
        if not pipeline_groups:
            pipeline_groups = [
                {
                    "trunk": p["trunk"],
                    "branches": p["branches"],
                    "operations": self.operations,
                }
                for p in self.pipelines
            ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        for group_idx, group in enumerate(pipeline_groups):
            trunk_ops = [op for op in group["trunk"] if type(op).__name__ != "GroupOperation"]
            branches = [
                [op for op in b if type(op).__name__ != "GroupOperation"]
                for b in group["branches"]
            ]

            table_docs = []
            for path in self.doc_manager.data_paths:
                docs = self.doc_manager.load_documents(str(path))
                table_docs.append(docs)
                self.log(f"Loaded {len(docs)} documents from {path}")

            total_docs = sum(len(d) for d in table_docs)
            self.results = [{"operation": "initial", "doc_count": total_docs, "tables": len(table_docs)}]

            # if trunk starts with joins
            join_idx = 0
            while join_idx < len(trunk_ops) and type(trunk_ops[join_idx]).__name__ == "JoinOperation":
                join_idx += 1
            join_ops = trunk_ops[:join_idx]
            rest_trunk_ops = trunk_ops[join_idx:]

            df_after_join = None
            if join_ops and len(table_docs) >= 2:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    df_join = self._build_yaml_and_run(
                        join_ops, table_docs, temp_path, env
                    )
                    df_after_join = df_join if not df_join.empty else None
                    # Use join output as single table for rest (documents_to_json_list ensures contents)
                    table_docs = [self.doc_manager.documents_to_json_list(df_join.to_dict("records"))]
            percent_count_denom_df = df_after_join if df_after_join is not None else (
                pd.DataFrame(table_docs[0]) if table_docs else None
            )

            table_docs_for_trunk = table_docs

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                df_trunk = self._build_yaml_and_run(
                    rest_trunk_ops, table_docs_for_trunk, temp_path, env
                )

            if not branches:
                group_dfs = [df_trunk]
                self.log(f"Pipeline group {group_idx + 1} complete (no branches) - {len(df_trunk)} documents")
            else:
                group_dfs = []
                for branch_idx, branch_ops in enumerate(branches):
                    if not branch_ops:
                        group_dfs.append(df_trunk)
                        self.log(f"Branch {branch_idx + 1} (empty) - passed through trunk result")
                        continue
                    self.log(f"Branch {branch_idx + 1}/{len(branches)} starting from trunk state ({len(df_trunk)} documents)")
                    trunk_as_docs = self.doc_manager.documents_to_json_list(df_trunk.to_dict("records")) if not df_trunk.empty else []
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        df_branch = self._build_yaml_and_run(
                            branch_ops, [trunk_as_docs], temp_path, env
                        )
                    group_dfs.append(df_branch)
                    self.log(f"Branch {branch_idx + 1} complete - {len(df_branch)} documents in result")

            all_group_results.append({
                "dfs": group_dfs,
                "operations": group["operations"],
                "trunk_df": df_trunk if branches else None,
                "percent_count_denom_df": percent_count_denom_df,
            })

        return all_group_results


from pipeline_sources import extract_tables_from_sql


# ---------------------------------------------------------------------------
# Semantic plan → DocETL YAML compiler + executor
# ---------------------------------------------------------------------------

def _step_to_docetl_op(step: PlannedStep, ctx: SubqueryContext, op_idx: int, post_join: bool = False, post_reduce: bool = False) -> dict | None:
    """Convert a PlannedStep to a DocETL operation config dict.
    
    Args:
        post_join: If True, the step operates on data after an equijoin,
                   which means documents have contents_left and contents_right fields.
        post_reduce: If True, the step operates on data after a reduce operation,
                     which means the document structure has been transformed and
                     original contents fields no longer exist.
    """
    instr = step.instruction
    if ctx:
        instr = ctx.substitute(instr)
    kind = step.kind

    if kind == "sem_filter":
        return FilterOperation(instr).to_yaml_config(op_idx, post_join=post_join, post_reduce=post_reduce)
    if kind == "sem_extract" or kind == "sem_map":
        return ExtractOperation(instr).to_yaml_config(op_idx, post_join=post_join, post_reduce=post_reduce)
    if kind == "sem_agg":
        return AggregateOperation(instr).to_yaml_config(op_idx, post_join=post_join, post_reduce=post_reduce)
    if kind == "sem_topk":
        limit = step.op.get("limit_count", 1)
        return RankOperation(instr, k=limit).to_yaml_config(op_idx, post_join=post_join, post_reduce=post_reduce)
    if kind == "sem_cluster_by":
        return GroupOperation(instr).to_yaml_config(op_idx, post_join=post_join, post_reduce=post_reduce)
    if kind == "sem_join":
        return JoinOperation(instr).to_yaml_config(op_idx)
    return None


def _run_docetl_semantic_subquery(
    sq_plan: dict,
    entry: dict,
    doc_manager: "DocumentManager",
    ctx: SubqueryContext,
    env: dict,
    verbose: bool = False,
) -> object:
    """Build and run a DocETL YAML pipeline for one subquery, return scalar."""
    steps_list: list[PlannedStep] = sq_plan["steps"]
    ops_configs = []
    yaml_steps = []
    prev_step = None
    op_idx = 0

    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td)
        # Write input data
        docs = doc_manager.load_documents(str(doc_manager.data_paths[0]))
        input_path = temp_path / "subq_input.json"
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(doc_manager.documents_to_json_list(docs), f, indent=2)

        datasets = {"subq_input": {"type": "file", "path": input_path.name}}
        prev_step = "subq_input"

        for step in steps_list:
            if step.kind == "subquery_result":
                continue
            cfg = _step_to_docetl_op(step, ctx, op_idx)
            if cfg is None:
                continue
            ops_configs.append(cfg)
            yaml_steps.append({
                "name": cfg["name"],
                "input": prev_step,
                "operations": [cfg["name"]],
            })
            prev_step = cfg["name"]
            op_idx += 1

        if not ops_configs:
            return len(docs)

        output_path = temp_path / "subq_output.json"
        pipeline_config = {
            "datasets": datasets,
            "default_model": "gpt-4o-mini",
            "operations": ops_configs,
            "pipeline": {
                "name": "subquery_pipeline",
                "steps": yaml_steps,
                "output": {"type": "file", "path": output_path.name},
            },
        }
        yaml_path = temp_path / "subq_pipeline.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)

        if verbose:
            print(f"  Running subquery DocETL pipeline ({len(ops_configs)} ops)...")
        result, _ = _run_docetl_pipeline(yaml_path, temp_path, env)
        if result.returncode != 0:
            if verbose:
                print(f"  Subquery DocETL failed: {result.stderr[:300]}")
            return None

        df = _docetl_output_to_df(output_path, True)
        if "_output" in df.columns and len(df) > 0:
            return df["_output"].iloc[0]
        if "extraction" in df.columns and len(df) > 0:
            return df["extraction"].iloc[0]
        if "aggregated_result" in df.columns and len(df) > 0:
            return df["aggregated_result"].iloc[0]
        return len(df)


def execute_semantic_plan_docetl(
    plan: dict,
    entry: dict,
    doc_manager: "DocumentManager",
    verbose: bool = False,
) -> tuple:
    """Execute a semantic plan using DocETL YAML pipelines.

    Runs subqueries first, then compiles the main pipeline with multi-head
    support (equijoin for merges, sequential ops otherwise).
    """
    # Early check for unsupported operations
    unsupported_op = _plan_contains_unsupported_op(plan)
    if unsupported_op:
        raise UnsupportedOperationError(
            f"{unsupported_op} is not supported by DocETL - skipping query"
        )
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    ctx = SubqueryContext()
    enrich_plan_instructions(plan, entry, ctx)

    # 1. Subqueries
    for sq_plan in plan.get("subquery_plans", []):
        var = sq_plan["pipeline"].get("subquery_var", "")
        val = _run_docetl_semantic_subquery(sq_plan, entry, doc_manager, ctx, env, verbose)
        if var:
            ctx.bind(var, val)
            if verbose:
                print(f"  Bound {var} = {val}")

    enrich_plan_instructions(plan, entry, ctx)

    # 2. Main pipeline — compile to YAML
    tables = plan.get("tables", [])
    main_steps: list[PlannedStep] = plan.get("main_steps", [])

    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td)
        datasets = {}
        # Write per-head input data
        for idx, table in enumerate(tables):
            if idx < len(doc_manager.data_paths):
                docs = doc_manager.load_documents(str(doc_manager.data_paths[idx]))
                input_path = temp_path / f"input_head_{idx}.json"
                with open(input_path, "w", encoding="utf-8") as f:
                    json.dump(doc_manager.documents_to_json_list(docs), f, indent=2)
                datasets[f"head_{idx}"] = {"type": "file", "path": input_path.name}

        ops_configs = []
        yaml_steps = []
        head_last_step: dict[int, str] = {i: f"head_{i}" for i in range(len(tables))}
        merged_head: int | None = None
        joined_heads: set[int] = set()  # Track which heads have been through a join
        reduced_heads: set[int] = set()  # Track which heads have been through a reduce (group/rank/agg)
        op_idx = 0

        for step in main_steps:
            kind = step.kind

            if kind == "subquery_result":
                continue

            # Determine if this step operates on post-join or post-reduce data
            target_ids = step.head_ids if step.head_ids else list(head_last_step.keys())
            is_post_join = any(hid in joined_heads for hid in target_ids)
            is_post_reduce = any(hid in reduced_heads for hid in target_ids)

            cfg = _step_to_docetl_op(step, ctx, op_idx, post_join=is_post_join, post_reduce=is_post_reduce)
            if cfg is None:
                continue

            if kind == "sem_join" and step.merge_pair:
                left_id, right_id = step.merge_pair
                ops_configs.append(cfg)
                yaml_steps.append({
                    "name": cfg["name"],
                    "operations": [{
                        cfg["name"]: {
                            "left": head_last_step.get(left_id, f"head_{left_id}"),
                            "right": head_last_step.get(right_id, f"head_{right_id}"),
                        }
                    }],
                })
                head_last_step[left_id] = cfg["name"]
                head_last_step.pop(right_id, None)
                merged_head = left_id
                joined_heads.add(left_id)  # Mark this head as having been through a join
            else:
                target_ids = step.head_ids if step.head_ids else list(head_last_step.keys())
                for hid in target_ids:
                    if hid not in head_last_step:
                        continue
                    step_cfg = dict(cfg)
                    if len(target_ids) > 1:
                        step_cfg["name"] = f"{cfg['name']}_h{hid}"
                    ops_configs.append(step_cfg)
                    yaml_steps.append({
                        "name": step_cfg["name"],
                        "input": head_last_step[hid],
                        "operations": [step_cfg["name"]],
                    })
                    head_last_step[hid] = step_cfg["name"]
                    
                    # Mark head as reduced if this was a reduce operation
                    if kind in ("sem_cluster_by", "sem_topk", "sem_agg"):
                        reduced_heads.add(hid)

            op_idx += 1

        if not ops_configs:
            return ()

        output_path = temp_path / "main_output.json"
        pipeline_config = {
            "datasets": datasets,
            "default_model": "gpt-4o-mini",
            "operations": ops_configs,
            "pipeline": {
                "name": "semantic_main",
                "steps": yaml_steps,
                "output": {"type": "file", "path": output_path.name},
            },
        }
        yaml_path = temp_path / "main_pipeline.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)
        
        # Debug: save a copy to inspect
        debug_yaml_path = Path(__file__).parent / "pipeline_data" / "debug_generated_pipeline.yaml"
        with open(debug_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)

        if verbose:
            print(f"  Running main DocETL pipeline ({len(ops_configs)} ops)...")
        result, _ = _run_docetl_pipeline(yaml_path, temp_path, env)
        if result.returncode != 0:
            if verbose:
                print(f"  DocETL main failed: {result.stderr[:2000]}")
                # Save full error to debug file
                err_path = Path(__file__).parent / "pipeline_data" / "debug_docetl_error.txt"
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(result.stderr)
            return ()

        df = _docetl_output_to_df(output_path, True)

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

    # Extract results
    values: list = []
    
    # If DataFrame is empty, return empty result (equivalent to [None])
    if df.empty:
        return ()
    
    # Check for aggregated_result column (from reduce operations)
    if "aggregated_result" in df.columns and len(df) > 0:
        for v in df["aggregated_result"]:
            if v is not None:
                if _is_divide_by_zero_ratio(v):
                    continue
                values.append(v)
    elif "extracted_value" in df.columns and len(df) > 0:
        for v in df["extracted_value"]:
            if v is not None:
                if _is_divide_by_zero_ratio(v):
                    continue
                values.append(v)
    elif "_output" in df.columns and len(df) > 0:
        for v in df["_output"]:
            if v is not None:
                if _is_divide_by_zero_ratio(v):
                    continue
                values.append(v)
    elif "extraction" in df.columns and len(df) > 0:
        for v in df["extraction"]:
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


def aggregate_metrics_to_csv(rows: list, output_dir: str = "./pipeline_data/results/docetl", num_skipped_unsupported: int = 0) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "docetl_metrics.csv"
    fieldnames = [
        "question_number", "question_id", "db_id", "difficulty", "num_documents",
        "altered_sql", "execution_time_seconds", "llm_total_tokens", "llm_total_cost",
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
    total_tokens = sum(r.get("llm_total_tokens") or 0 for r in rows)
    total_cost = sum(r.get("llm_total_cost") or 0 for r in rows)
    num_correct = sum(1 for r in rows if r.get("correct") is True)
    summary_path = output_path / "docetl_metrics_summary.csv"
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
    parser.add_argument("--num_documents", type=int, default=10)
    parser.add_argument("--semantic", action="store_true", default=True, help="Use semantic plan execution instead of format strings")
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

        print(f"[docetl] Running {len(filtered_entries)} entries | data_root={data_root}")

        metrics_rows = []
        num_skipped_unsupported = 0
        for question_number, entry in enumerate(filtered_entries, start=1):
            tables_list = entry["tables"].split(", ")
            database_name = database_tables.get(tables_list[0])
            if not database_name:
                continue
            ExtractOperation._extract_counter = 0
            doc_manager = DocumentManager(str(data_root), tables_list, max_docs=num_documents)

            sem_plan = prepare_semantic_execution(entry) if use_semantic else None

            start_time = time.perf_counter()
            execute_result = None
            semantic_extracted = None
            try:
                if sem_plan:
                    semantic_extracted = execute_semantic_plan_docetl(
                        sem_plan, entry, doc_manager, verbose=True)
                else:
                    sequence = entry["format"]
                    pipeline = Pipeline(doc_manager, verbose=True)
                    pipeline.parse_format(sequence)
                    execute_result = pipeline.execute()
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
            llm_tokens = 0
            llm_cost = 0.0
            if sem_plan:
                num_docs_processed = sum(len(doc_manager.load_documents(str(p))) for p in doc_manager.data_paths)
            else:
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
                if "_output" in df.columns and len(df) > 0:
                    val = df["_output"].iloc[0]
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return val
                if "extraction" in df.columns and len(df) > 0:
                    val = df["extraction"].iloc[0]
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
                trunk_df = group_result.get("trunk_df")
                percent_count_denom_df = group_result.get("percent_count_denom_df")
                has_zero_doc_df = any(_doc_count(df) == 0 for df in result if df is not None)
                op_failed = False
                prev_was_comparison = False
                group_values = []

                for idx, df in enumerate(result):
                    if df is not None:
                        doc_count = _doc_count(df)
                        print(f"Group {group_idx + 1} dataframe {idx} contains {doc_count} documents")
                        if "_output" in df.columns and len(df) > 0:
                            print(f"  has _output: {df['_output'].iloc[0]}")

                for op_idx, operation in enumerate(operations):
                    print(f"Executing OP (group {group_idx + 1}): {operation}")
                    if operation == "ratio":
                        try:
                            if len(result) == 2:
                                v0, v1 = _get_df_value(result[0]), _get_df_value(result[1])
                                if v1 == 0:
                                    group_values.append("None")
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

                if not group_values:
                    for df_item in result:
                        if df_item is None:
                            continue
                        if "_output" in df_item.columns and len(df_item) > 0:
                            val = df_item["_output"].iloc[0]
                            if val is not None:
                                group_values.append(str(val))
                                break
                        elif "extraction" in df_item.columns and len(df_item) > 0:
                            val = df_item["extraction"].iloc[0]
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
            "docetl", args.null_param, args.noise_param, num_documents
        )
        out_path = aggregate_metrics_to_csv(
            metrics_rows,
            output_dir=str(results_dir),
            num_skipped_unsupported=num_skipped_unsupported,
        )
        print(f"\nMetrics written to {out_path}")
        print(f"Results directory: {results_dir}")
