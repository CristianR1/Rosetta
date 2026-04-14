"""
PostgreSQL EXPLAIN JSON parser and semantic pipeline builder.

Parses PostgreSQL EXPLAIN (FORMAT JSON, VERBOSE) output and constructs:
1. Physical operator DAG from the execution plan tree
2. Semantic operation pipeline with separated subquery and main pipelines

Semantic operations are categorized as:
  - Cumulative operations: Build and transform the working set
    * sem_filter: Row filtering (WHERE, HAVING conditions)
    * sem_join: Combine pipelines
    * sem_cluster_by: Group rows (GROUP BY)
    * sem_dedup: Remove duplicates (DISTINCT via Group node)
    * sem_topk: Order and limit (when not SELECT *)
    
  - Final operations: Extract results from the final set
    * sem_extract: Project specific columns from documents
    * sem_agg: Compute aggregations for output (COUNT, SUM, etc.)
    * sem_topk: Order and limit (when SELECT *)

Subquery pipelines produce scalar values ($0, $1, ...) consumed by the main pipeline.
"""

from __future__ import annotations

import re
from typing import Optional, NamedTuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
#  Physical operator categories
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICAL_OPERATORS = {
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "Tid Scan",
    "Tid Range Scan",
    "Subquery Scan",
    "Function Scan",
    "Table Function Scan",
    "Values Scan",
    "CTE Scan",
    "Named Tuplestore Scan",
    "WorkTable Scan",
    "Foreign Scan",
    "Custom Scan",
    "Nested Loop",
    "Merge Join",
    "Hash Join",
    "Aggregate",
    "Group Aggregate",
    "Hash Aggregate",
    "Mixed Aggregate",
    "Group",
    "WindowAgg",
    "Unique",
    "SetOp",
    "Sort",
    "Incremental Sort",
    "Limit",
    "LockRows",
    "Materialize",
    "Memoize",
    "Hash",
    "Gather",
    "Gather Merge",
    "Append",
    "Merge Append",
    "Recursive Union",
    "BitmapAnd",
    "BitmapOr",
    "Result",
    "ProjectSet",
    "ModifyTable",
    "Sample Scan",
}

SCAN_NODES = {
    "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
    "Bitmap Index Scan", "Tid Scan", "Tid Range Scan", "Sample Scan",
    "CTE Scan", "Function Scan", "Values Scan", "Foreign Scan", "Subquery Scan",
}
NOISE_NODES = {"Memoize", "Hash", "Gather", "Gather Merge", "Materialize"}
JOIN_NODES = {"Merge Join", "Hash Join", "Nested Loop"}
AGGREGATE_NODES = {"Aggregate", "Group Aggregate", "Hash Aggregate", "Mixed Aggregate"}
SORT_NODES = {"Sort", "Incremental Sort"}

_TRANSPARENT_SORT_PARENTS = (
    JOIN_NODES | AGGREGATE_NODES | {"Unique", "Group", "WindowAgg", "SetOp"}
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Semantic operation types
# ═══════════════════════════════════════════════════════════════════════════════

CUMULATIVE_OPS = frozenset({
    "sem_filter",
    "sem_join",
    "sem_cluster_by",
    "sem_dedup",
    "subquery_result",
})

FINAL_OPS = frozenset({
    "sem_extract",
    "sem_agg",
    "sem_topk",
    "sem_map",
})

# sem_topk can be either cumulative or final depending on SELECT clause

_TABLE_COL_RE = re.compile(r"\b(\w+)\.(\w+)")

# Aggregate function detection regex
_AGG_FUNC_RE = re.compile(
    r"\b(count|sum|avg|min|max|stddev|stddev_pop|stddev_samp|variance|var_pop|var_samp|"
    r"bool_and|bool_or|every|array_agg|string_agg|json_agg|jsonb_agg|xmlagg|bit_and|bit_or|"
    r"regr_\w+|covar_\w+|corr|mode|percentile|percentile_cont|percentile_disc|"
    r"rank|dense_rank|row_number|cume_dist|percent_rank|first_value|last_value|nth_value|"
    r"lag|lead)\s*\(",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL SELECT Parser
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SelectClause:
    """Parsed SELECT clause information."""
    raw_items: list[str] = field(default_factory=list)
    is_select_star: bool = False
    column_refs: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    has_distinct: bool = False


@dataclass 
class ParsedSQL:
    """Parsed SQL query components."""
    select_clause: SelectClause = field(default_factory=SelectClause)
    order_by: Optional[str] = None
    order_by_exprs: list[str] = field(default_factory=list)
    limit: Optional[int] = None
    has_group_by: bool = False
    group_by_cols: list[str] = field(default_factory=list)
    has_subquery: bool = False


def _find_matching_paren(sql: str, start: int) -> int:
    """Find the matching closing parenthesis."""
    depth = 1
    i = start + 1
    while i < len(sql) and depth > 0:
        if sql[i] == '(':
            depth += 1
        elif sql[i] == ')':
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def _split_select_items(select_part: str) -> list[str]:
    """Split SELECT items by comma, respecting parentheses and quotes."""
    items = []
    current = []
    depth = 0
    in_string = False
    string_char = None
    
    for char in select_part:
        if char in ("'", '"') and not in_string:
            in_string = True
            string_char = char
            current.append(char)
        elif char == string_char and in_string:
            in_string = False
            string_char = None
            current.append(char)
        elif not in_string:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                items.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
    
    if current:
        items.append(''.join(current).strip())
    
    return [item for item in items if item]


def _is_aggregate_expr(expr: str) -> bool:
    """Check if expression contains an aggregate function."""
    return bool(_AGG_FUNC_RE.search(expr))


def _strip_alias(expr: str) -> str:
    """Remove trailing AS alias from expression, respecting parentheses."""
    expr = expr.strip()
    # Find AS keyword not inside parentheses
    depth = 0
    i = len(expr) - 1
    as_pos = -1
    
    # Scan backwards to find the last top-level AS
    while i >= 0:
        c = expr[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
        elif depth == 0 and i >= 2:
            # Check for AS keyword (case insensitive)
            if expr[i-2:i].upper() == 'AS' and (i == 2 or not expr[i-3].isalnum()):
                # Check character after AS is whitespace
                if i < len(expr) and expr[i-2:i+1].upper().startswith('AS '):
                    as_pos = i - 2
                    break
        i -= 1
    
    if as_pos > 0:
        return expr[:as_pos].strip()
    
    # Also try simple regex for common pattern: " AS identifier" at end
    as_match = re.search(r'\s+AS\s+[\w"]+\s*$', expr, re.IGNORECASE)
    if as_match:
        return expr[:as_match.start()].strip()
    
    return expr


def _extract_select_expression(expr: str) -> str:
    """
    Extract the full expression from a SELECT item, preserving complex expressions.
    
    Handles:
    - Simple columns: T1.name -> T1.name
    - Function calls: SUBSTR(T2.Date, 1, 4) -> SUBSTR(T2.Date, 1, 4)
    - Complex expressions: EXTRACT(YEAR FROM dob) -> EXTRACT(YEAR FROM dob)
    - CASE expressions: CASE WHEN ... END -> CASE WHEN ... END
    - Aliased expressions: expr AS alias -> expr
    """
    expr = _strip_alias(expr)
    return expr


def _extract_column_ref(expr: str) -> Optional[str]:
    """Extract column reference from expression if it's a simple column ref."""
    expr = expr.strip()
    expr = _strip_alias(expr)
    
    # Check if it's a simple column reference (table.column or just column)
    if re.match(r'^[\w\.]+$', expr) and not _is_aggregate_expr(expr):
        return expr
    return None


def _find_top_level_from(sql: str) -> int:
    """
    Find the position of the top-level FROM keyword in SQL.
    
    This handles cases where FROM appears inside expressions like EXTRACT(YEAR FROM dob).
    """
    sql_upper = sql.upper()
    depth = 0
    i = 0
    while i < len(sql):
        c = sql[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and sql_upper[i:i+5] == 'FROM ' or sql_upper[i:i+5] == 'FROM\t' or sql_upper[i:i+5] == 'FROM\n':
            # Check it's a word boundary (not part of another word)
            if i == 0 or not sql[i-1].isalnum():
                return i
        i += 1
    return -1


def parse_sql_select(sql: str) -> ParsedSQL:
    """
    Parse SQL SELECT statement to extract components.
    
    This is a simplified parser that handles common patterns.
    """
    result = ParsedSQL()
    sql_upper = sql.upper()
    
    # Check for subqueries
    if sql_upper.count('SELECT') > 1:
        result.has_subquery = True
    
    # Find SELECT clause using top-level FROM detection
    select_start = sql_upper.find('SELECT')
    if select_start == -1:
        return result
    
    # Check for DISTINCT
    after_select = sql[select_start + 6:].lstrip()
    distinct_offset = 0
    if after_select.upper().startswith('DISTINCT'):
        result.select_clause.has_distinct = True
        distinct_offset = after_select.upper().find('DISTINCT') + 8
        # Skip whitespace after DISTINCT
        while distinct_offset < len(after_select) and after_select[distinct_offset].isspace():
            distinct_offset += 1
    
    # Find the top-level FROM
    from_pos = _find_top_level_from(sql)
    
    if from_pos > select_start:
        # Extract the SELECT items between SELECT (DISTINCT) and FROM
        select_items_start = select_start + 6 + distinct_offset
        if result.select_clause.has_distinct:
            select_items_start = select_start + 6
            # Re-find start after DISTINCT
            after_select = sql[select_start + 6:].lstrip()
            if after_select.upper().startswith('DISTINCT'):
                dist_end = 8
                while dist_end < len(after_select) and after_select[dist_end].isspace():
                    dist_end += 1
                select_items_start = select_start + 6 + (len(sql[select_start + 6:]) - len(after_select)) + dist_end
        
        select_part = sql[select_items_start:from_pos].strip()
        select_match = True
    else:
        # Fallback to simple regex (no FROM found, maybe a simple query)
        select_match = re.search(r'\bSELECT\s+(DISTINCT\s+)?(.+?)$', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            if select_match.group(1):
                result.select_clause.has_distinct = True
            select_part = select_match.group(2).strip()
        else:
            return result
    
    if select_match:
        # Check for SELECT *
        if select_part.strip() == '*' or re.match(r'^\w+\.\*$', select_part.strip()):
            result.select_clause.is_select_star = True
            result.select_clause.raw_items = [select_part.strip()]
        else:
            items = _split_select_items(select_part)
            result.select_clause.raw_items = items
            
            for item in items:
                if _is_aggregate_expr(item):
                    # Store the full expression, stripped of alias
                    result.select_clause.aggregations.append(_extract_select_expression(item))
                else:
                    col_ref = _extract_column_ref(item)
                    if col_ref:
                        result.select_clause.column_refs.append(col_ref)
                    else:
                        # Complex expression that's not an aggregate - preserve full expression
                        result.select_clause.column_refs.append(_extract_select_expression(item))
    
    # Find GROUP BY
    group_match = re.search(r'\bGROUP\s+BY\s+(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)', sql, re.IGNORECASE | re.DOTALL)
    if group_match:
        result.has_group_by = True
        group_part = group_match.group(1).strip()
        result.group_by_cols = [col.strip() for col in _split_select_items(group_part)]
    
    # Find ORDER BY
    order_match = re.search(r'\bORDER\s+BY\s+(.+?)(?:\bLIMIT\b|\bOFFSET\b|$)', sql, re.IGNORECASE | re.DOTALL)
    if order_match:
        result.order_by = order_match.group(1).strip()
        # Split by comma for individual expressions
        result.order_by_exprs = [expr.strip() for expr in _split_select_items(result.order_by)]
    
    # Find LIMIT
    limit_match = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
    if limit_match:
        result.limit = int(limit_match.group(1))
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Physical plan node
# ═══════════════════════════════════════════════════════════════════════════════

class PlanNode:
    """Represents a node in the physical execution plan DAG."""

    def __init__(
        self,
        node_id: int,
        node_type: str,
        relation_name: Optional[str] = None,
        alias: Optional[str] = None,
        parent_relationship: Optional[str] = None,
        index_name: Optional[str] = None,
        startup_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
        plan_rows: Optional[int] = None,
        plan_width: Optional[int] = None,
        filter_cond: Optional[str] = None,
        index_cond: Optional[str] = None,
        join_filter: Optional[str] = None,
        join_type: Optional[str] = None,
        hash_cond: Optional[str] = None,
        merge_cond: Optional[str] = None,
        sort_key: Optional[list] = None,
        group_key: Optional[list] = None,
        strategy: Optional[str] = None,
        partial_mode: Optional[str] = None,
        output: Optional[list] = None,
        command: Optional[str] = None,
        subplan_name: Optional[str] = None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.output = output
        self.relation_name = relation_name
        self.alias = alias
        self.parent_relationship = parent_relationship
        self.index_name = index_name
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.plan_rows = plan_rows
        self.plan_width = plan_width
        self.filter_cond = filter_cond
        self.index_cond = index_cond
        self.join_filter = join_filter
        self.join_type = join_type
        self.hash_cond = hash_cond
        self.merge_cond = merge_cond
        self.sort_key = sort_key
        self.group_key = group_key
        self.strategy = strategy
        self.partial_mode = partial_mode
        self.command = command
        self.subplan_name = subplan_name
        self.children: list[int] = []
        self.parent: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        d: dict = {
            "id": self.node_id,
            "node_type": self.node_type,
        }
        _optional = [
            ("relation_name", self.relation_name),
            ("alias", self.alias),
            ("parent_relationship", self.parent_relationship),
            ("index_name", self.index_name),
            ("startup_cost", self.startup_cost),
            ("total_cost", self.total_cost),
            ("plan_rows", self.plan_rows),
            ("plan_width", self.plan_width),
            ("filter", self.filter_cond),
            ("index_cond", self.index_cond),
            ("join_filter", self.join_filter),
            ("join_type", self.join_type),
            ("hash_cond", self.hash_cond),
            ("merge_cond", self.merge_cond),
            ("sort_key", self.sort_key),
            ("group_key", self.group_key),
            ("strategy", self.strategy),
            ("partial_mode", self.partial_mode),
            ("output", self.output),
            ("command", self.command),
            ("subplan_name", self.subplan_name),
        ]
        for key, val in _optional:
            if val is not None:
                d[key] = val
        if self.children:
            d["children"] = self.children
        if self.parent is not None:
            d["parent"] = self.parent
        return d


# ═══════════════════════════════════════════════════════════════════════════════
#  Semantic operation node
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticOp:
    """A single operation in the semantic pipeline."""
    op_type: str
    is_final: bool = False
    
    # Operation-specific fields
    condition: Optional[str] = None
    join_condition: Optional[str] = None
    join_type: Optional[str] = None
    group_keys: Optional[list[str]] = None
    agg_expression: Optional[str] = None
    extract_column: Optional[str] = None
    sort_keys: Optional[list[str]] = None
    limit_count: Optional[int] = None
    dedup_keys: Optional[list[str]] = None
    subquery_ref: Optional[str] = None  # e.g., "$0"
    expression: Optional[str] = None
    set_op_type: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = {"op_type": self.op_type}
        if self.is_final:
            d["is_final"] = True
        
        attrs = [
            ("condition", self.condition),
            ("join_condition", self.join_condition),
            ("join_type", self.join_type),
            ("group_keys", self.group_keys),
            ("agg_expression", self.agg_expression),
            ("extract_column", self.extract_column),
            ("sort_keys", self.sort_keys),
            ("limit_count", self.limit_count),
            ("dedup_keys", self.dedup_keys),
            ("subquery_ref", self.subquery_ref),
            ("expression", self.expression),
            ("set_op_type", self.set_op_type),
        ]
        for key, val in attrs:
            if val is not None:
                d[key] = val
        return d


@dataclass
class SemanticPipeline:
    """A semantic pipeline with cumulative and final operations."""
    pipeline_id: int
    is_subquery: bool = False
    subquery_var: Optional[str] = None  # e.g., "$0" for subqueries
    cumulative_ops: list[SemanticOp] = field(default_factory=list)
    final_ops: list[SemanticOp] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = {
            "pipeline_id": self.pipeline_id,
            "cumulative_ops": [op.to_dict() for op in self.cumulative_ops],
            "final_ops": [op.to_dict() for op in self.final_ops],
        }
        if self.is_subquery:
            d["is_subquery"] = True
        if self.subquery_var:
            d["subquery_var"] = self.subquery_var
        return d


# ═══════════════════════════════════════════════════════════════════════════════
#  Physical plan parser
# ═══════════════════════════════════════════════════════════════════════════════

class ExplainParser:
    """Parser for PostgreSQL EXPLAIN (FORMAT JSON) output."""

    def __init__(self):
        self.nodes: list[PlanNode] = []
        self.edges: list[tuple[int, int]] = []
        self._next_id = 0

    def reset(self):
        self.nodes = []
        self.edges = []
        self._next_id = 0

    def _get_next_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def _parse_plan_node(
        self,
        plan_dict: dict,
        parent_id: Optional[int] = None,
        parent_relationship: Optional[str] = None,
    ) -> int:
        node_id = self._get_next_id()

        node = PlanNode(
            node_id=node_id,
            node_type=plan_dict.get("Node Type", "Unknown"),
            relation_name=plan_dict.get("Relation Name"),
            alias=plan_dict.get("Alias"),
            parent_relationship=parent_relationship or plan_dict.get("Parent Relationship"),
            index_name=plan_dict.get("Index Name"),
            startup_cost=plan_dict.get("Startup Cost"),
            total_cost=plan_dict.get("Total Cost"),
            plan_rows=plan_dict.get("Plan Rows"),
            plan_width=plan_dict.get("Plan Width"),
            filter_cond=plan_dict.get("Filter"),
            index_cond=plan_dict.get("Index Cond"),
            join_filter=plan_dict.get("Join Filter"),
            join_type=plan_dict.get("Join Type"),
            hash_cond=plan_dict.get("Hash Cond"),
            merge_cond=plan_dict.get("Merge Cond"),
            sort_key=plan_dict.get("Sort Key"),
            group_key=plan_dict.get("Group Key"),
            strategy=plan_dict.get("Strategy"),
            partial_mode=plan_dict.get("Partial Mode"),
            output=plan_dict.get("Output"),
            command=plan_dict.get("Command"),
            subplan_name=plan_dict.get("Subplan Name"),
        )

        if parent_id is not None:
            node.parent = parent_id
            self.edges.append((node_id, parent_id))

        for child_plan in plan_dict.get("Plans", []):
            child_relationship = child_plan.get("Parent Relationship")
            child_id = self._parse_plan_node(
                child_plan,
                parent_id=node_id,
                parent_relationship=child_relationship,
            )
            node.children.append(child_id)

        self.nodes.append(node)
        return node_id

    def parse(self, explain_json: list) -> dict:
        """Parse PostgreSQL EXPLAIN (FORMAT JSON) output."""
        self.reset()

        if not explain_json or not isinstance(explain_json, list):
            return {
                "operator_sequence": [],
                "nodes": [],
                "edges": [],
                "root_id": None,
                "planning_time": None,
                "execution_time": None,
            }

        first_entry = explain_json[0]
        plan = first_entry.get("Plan")

        if not plan:
            return {
                "operator_sequence": [],
                "nodes": [],
                "edges": [],
                "root_id": None,
                "planning_time": first_entry.get("Planning Time"),
                "execution_time": first_entry.get("Execution Time"),
            }

        root_id = self._parse_plan_node(plan)
        operator_sequence = [node.node_type for node in self.nodes]

        return {
            "operator_sequence": operator_sequence,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": self.edges,
            "root_id": root_id,
            "planning_time": first_entry.get("Planning Time"),
            "execution_time": first_entry.get("Execution Time"),
        }

    def get_bottom_up_traversal(self) -> list[PlanNode]:
        return list(self.nodes)

    def get_leaf_nodes(self) -> list[PlanNode]:
        return [node for node in self.nodes if not node.children]

    def get_scan_nodes(self) -> list[PlanNode]:
        return [node for node in self.nodes if node.node_type in SCAN_NODES]

    def get_join_nodes(self) -> list[PlanNode]:
        return [node for node in self.nodes if node.node_type in JOIN_NODES]


# ═══════════════════════════════════════════════════════════════════════════════
#  Semantic Pipeline Builder
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticPipelineBuilder:
    """
    Constructs semantic pipelines from a physical execution plan.
    
    Separates subquery pipelines from the main pipeline and categorizes
    operations as cumulative or final based on their role in the query.
    """

    def __init__(self, physical_nodes: list[PlanNode], sql: str):
        self._pnodes: dict[int, PlanNode] = {n.node_id: n for n in physical_nodes}
        self._ordered: list[PlanNode] = list(physical_nodes)
        self._sql = sql
        self._parsed_sql = parse_sql_select(sql)
        
        # Track subquery pipelines by their variable name ($0, $1, ...)
        self._subquery_pipelines: list[SemanticPipeline] = []
        self._subquery_count = 0
        
        # Main pipeline
        self._main_cumulative: list[SemanticOp] = []
        self._main_final: list[SemanticOp] = []
        
        # Track which physical nodes belong to subqueries
        self._subquery_nodes: set[int] = set()
        self._subquery_roots: dict[int, str] = {}  # node_id -> $N
        
        # Track the current head of each physical node's pipeline
        self._head_ops: dict[int, list[SemanticOp]] = {}
        
        # Track tables from scans (for internal use, not output)
        self._tables: set[str] = set()

    def _identify_subquery_nodes(self):
        """Identify nodes that are part of InitPlan/SubPlan subqueries."""
        for pnode in self._ordered:
            if pnode.parent_relationship in ("InitPlan", "SubPlan"):
                # This node and its descendants are part of a subquery
                self._mark_subquery_tree(pnode.node_id)
                var_name = f"${self._subquery_count}"
                self._subquery_roots[pnode.node_id] = var_name
                self._subquery_count += 1

    def _mark_subquery_tree(self, node_id: int):
        """Mark a node and all its descendants as part of a subquery."""
        self._subquery_nodes.add(node_id)
        node = self._pnodes.get(node_id)
        if node:
            for child_id in node.children:
                self._mark_subquery_tree(child_id)

    def _parent_of(self, pnode: PlanNode) -> Optional[PlanNode]:
        if pnode.parent is not None:
            return self._pnodes.get(pnode.parent)
        return None

    def _regular_children(self, pnode: PlanNode) -> list[int]:
        """Pipeline-input children (excludes InitPlan / SubPlan)."""
        return [
            c for c in pnode.children
            if (self._pnodes[c].parent_relationship or "") not in ("InitPlan", "SubPlan")
        ]

    def _is_join_probe(self, scan_node: PlanNode) -> bool:
        """True when the scan's Index Cond references another table's alias."""
        if not scan_node.index_cond:
            return False
        alias = scan_node.alias or scan_node.relation_name
        if not alias:
            return False
        refs = _TABLE_COL_RE.findall(scan_node.index_cond)
        other_tables = {t for t, _ in refs if t != alias}
        return len(other_tables) > 0

    def _is_sort_transparent(self, sort_node: PlanNode) -> bool:
        """A Sort is transparent when it pre-sorts for a join, aggregate, etc."""
        parent = self._parent_of(sort_node)
        if parent is None:
            return False
        return parent.node_type in _TRANSPARENT_SORT_PARENTS

    def _find_inner_scan(self, node_id: int) -> Optional[PlanNode]:
        """Walk through noise nodes to find the underlying scan."""
        node = self._pnodes.get(node_id)
        while node and node.node_type in NOISE_NODES and node.children:
            node = self._pnodes.get(node.children[0])
        return node

    def _add_op(self, op: SemanticOp, node_id: int):
        """Add operation to the appropriate pipeline."""
        if node_id in self._subquery_nodes:
            # Find which subquery this belongs to
            for root_id, var_name in self._subquery_roots.items():
                if self._is_descendant_of(node_id, root_id) or node_id == root_id:
                    # Add to subquery pipeline
                    pipeline = self._get_or_create_subquery_pipeline(var_name)
                    if op.is_final:
                        pipeline.final_ops.append(op)
                    else:
                        pipeline.cumulative_ops.append(op)
                    return
        
        # Add to main pipeline
        if op.is_final:
            self._main_final.append(op)
        else:
            self._main_cumulative.append(op)

    def _is_descendant_of(self, node_id: int, ancestor_id: int) -> bool:
        """Check if node_id is a descendant of ancestor_id."""
        if node_id == ancestor_id:
            return True
        node = self._pnodes.get(node_id)
        if node and node.parent is not None:
            return self._is_descendant_of(node.parent, ancestor_id)
        return False

    def _get_or_create_subquery_pipeline(self, var_name: str) -> SemanticPipeline:
        """Get or create a subquery pipeline by variable name."""
        for pipeline in self._subquery_pipelines:
            if pipeline.subquery_var == var_name:
                return pipeline
        
        pipeline = SemanticPipeline(
            pipeline_id=len(self._subquery_pipelines),
            is_subquery=True,
            subquery_var=var_name,
        )
        self._subquery_pipelines.append(pipeline)
        return pipeline

    def _has_subquery_ref(self, condition: Optional[str]) -> Optional[str]:
        """Check if condition contains a subquery reference ($0, $1, etc.)."""
        if not condition:
            return None
        match = re.search(r'\$(\d+)', condition)
        if match:
            return f"${match.group(1)}"
        return None

    def build(self) -> dict:
        """Build the semantic pipelines."""
        # First pass: identify subquery nodes
        self._identify_subquery_nodes()
        
        # Second pass: process nodes bottom-up
        for pnode in self._ordered:
            self._dispatch(pnode)
        
        # Third pass: add final operations based on parsed SQL
        self._add_final_operations()
        
        # Build result
        main_pipeline = SemanticPipeline(
            pipeline_id=len(self._subquery_pipelines),
            is_subquery=False,
            cumulative_ops=self._main_cumulative,
            final_ops=self._main_final,
        )
        
        return {
            "subquery_pipelines": [p.to_dict() for p in self._subquery_pipelines],
            "main_pipeline": main_pipeline.to_dict(),
        }

    def _dispatch(self, pnode: PlanNode):
        """Dispatch node to appropriate handler."""
        nt = pnode.node_type

        if nt in NOISE_NODES:
            pass  # Skip noise nodes
        elif nt == "Bitmap Index Scan":
            pass  # Handled by Bitmap Heap Scan
        elif nt == "Bitmap Heap Scan":
            self._do_bitmap_heap_scan(pnode)
        elif nt == "Subquery Scan":
            self._do_subquery_scan(pnode)
        elif nt == "CTE Scan":
            self._do_cte_scan(pnode)
        elif nt in SCAN_NODES:
            self._do_scan(pnode)
        elif nt in JOIN_NODES:
            self._do_join(pnode)
        elif nt in AGGREGATE_NODES:
            self._do_aggregate(pnode)
        elif nt in SORT_NODES:
            self._do_sort(pnode)
        elif nt == "Limit":
            self._do_limit(pnode)
        elif nt == "WindowAgg":
            self._do_window_agg(pnode)
        elif nt == "Unique":
            pass  # DISTINCT handling
        elif nt == "Group":
            self._do_group(pnode)
        elif nt == "SetOp":
            self._do_setop(pnode)
        elif nt in ("Append", "Merge Append"):
            pass  # Handled by SetOp
        elif nt == "Result":
            self._do_result(pnode)

    def _do_scan(self, pnode: PlanNode):
        """Handle scan nodes - track tables internally, emit filters."""
        if pnode.relation_name:
            self._tables.add(pnode.relation_name)
        
        # Emit filter if present
        if pnode.filter_cond:
            subquery_ref = self._has_subquery_ref(pnode.filter_cond)
            op = SemanticOp(
                op_type="sem_filter",
                condition=pnode.filter_cond,
            )
            if subquery_ref:
                # Add subquery_result marker before the filter
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(op, pnode.node_id)

        if pnode.index_cond and not self._is_join_probe(pnode):
            subquery_ref = self._has_subquery_ref(pnode.index_cond)
            op = SemanticOp(
                op_type="sem_filter",
                condition=pnode.index_cond,
            )
            if subquery_ref:
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(op, pnode.node_id)

    def _do_bitmap_heap_scan(self, pnode: PlanNode):
        """Handle bitmap heap scan."""
        if pnode.relation_name:
            self._tables.add(pnode.relation_name)
        
        # Get filter from bitmap index scan child
        for cid in pnode.children:
            child = self._pnodes.get(cid)
            if child and child.node_type == "Bitmap Index Scan" and child.index_cond:
                subquery_ref = self._has_subquery_ref(child.index_cond)
                if subquery_ref:
                    self._add_op(SemanticOp(
                        op_type="subquery_result",
                        subquery_ref=subquery_ref,
                    ), pnode.node_id)
                self._add_op(SemanticOp(
                    op_type="sem_filter",
                    condition=child.index_cond,
                ), pnode.node_id)

        if pnode.filter_cond:
            subquery_ref = self._has_subquery_ref(pnode.filter_cond)
            if subquery_ref:
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=pnode.filter_cond,
            ), pnode.node_id)

    def _do_subquery_scan(self, pnode: PlanNode):
        """Handle subquery scan - emit filter if present."""
        if pnode.filter_cond:
            subquery_ref = self._has_subquery_ref(pnode.filter_cond)
            if subquery_ref:
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=pnode.filter_cond,
            ), pnode.node_id)

    def _do_cte_scan(self, pnode: PlanNode):
        """Handle CTE scan."""
        if pnode.filter_cond:
            subquery_ref = self._has_subquery_ref(pnode.filter_cond)
            if subquery_ref:
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=pnode.filter_cond,
            ), pnode.node_id)

        if pnode.index_cond:
            subquery_ref = self._has_subquery_ref(pnode.index_cond)
            if subquery_ref:
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=pnode.index_cond,
            ), pnode.node_id)

    def _find_join_condition_in_children(self, children: list[int]) -> Optional[str]:
        """
        Search child nodes for index conditions that serve as join predicates.
        
        In Nested Loop joins, PostgreSQL often pushes the join condition down
        to the inner side's Index Cond. This method searches all children
        (including through noise nodes) for such conditions.
        """
        for child_id in children:
            scan = self._find_inner_scan(child_id)
            if scan and scan.index_cond and self._is_join_probe(scan):
                return scan.index_cond
        return None

    def _do_join(self, pnode: PlanNode):
        """Handle join nodes."""
        children = self._regular_children(pnode)
        if len(children) < 2:
            return

        join_type_raw = (pnode.join_type or "Inner").strip()
        is_semi = "semi" in join_type_raw.lower()
        is_anti = "anti" in join_type_raw.lower()

        # Try to find join condition from various sources
        join_cond = pnode.merge_cond or pnode.hash_cond
        residual_filter: Optional[str] = None

        if join_cond and pnode.join_filter:
            # Both explicit join cond and join filter exist
            residual_filter = pnode.join_filter
        elif not join_cond:
            # No explicit merge/hash condition - check join_filter first
            join_cond = pnode.join_filter
            if not join_cond:
                # Still no condition - search children for index conditions
                # that reference multiple tables (indicating a join predicate)
                join_cond = self._find_join_condition_in_children(children)

        if is_semi or is_anti:
            label = "NOT EXISTS" if is_anti else "EXISTS"
            cond_text = f"{label}({join_cond})" if join_cond else label
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=cond_text,
                join_condition=join_cond,
                join_type=join_type_raw,
            ), pnode.node_id)
        else:
            self._add_op(SemanticOp(
                op_type="sem_join",
                join_condition=join_cond,
                join_type=join_type_raw,
            ), pnode.node_id)

        if residual_filter:
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=residual_filter,
            ), pnode.node_id)

    def _do_sort(self, pnode: PlanNode):
        """Handle sort - only emit if semantically meaningful (not for join prep)."""
        if self._is_sort_transparent(pnode):
            return  # Skip sorts that prep for joins/aggs
        
        # This is a meaningful ORDER BY - will be handled as final op
        # Don't emit here, let _add_final_operations handle it

    def _do_limit(self, pnode: PlanNode):
        """Handle limit - combined with sort as topk in final ops."""
        # Don't emit here, handled in _add_final_operations
        pass

    def _do_aggregate(self, pnode: PlanNode):
        """Handle aggregate nodes - emit cluster_by for GROUP BY."""
        group_keys = pnode.group_key
        
        # If there's a GROUP BY, emit sem_cluster_by
        if group_keys:
            self._add_op(SemanticOp(
                op_type="sem_cluster_by",
                group_keys=list(group_keys),
            ), pnode.node_id)

        # HAVING clause (filter condition on aggregate)
        if pnode.filter_cond:
            subquery_ref = self._has_subquery_ref(pnode.filter_cond)
            if subquery_ref:
                self._add_op(SemanticOp(
                    op_type="subquery_result",
                    subquery_ref=subquery_ref,
                ), pnode.node_id)
            self._add_op(SemanticOp(
                op_type="sem_filter",
                condition=pnode.filter_cond,
            ), pnode.node_id)
        
        # For subqueries, the aggregate IS the final operation
        if pnode.node_id in self._subquery_nodes:
            output = pnode.output or []
            agg_exprs = [o for o in output if o not in (group_keys or [])]
            if agg_exprs:
                for expr in agg_exprs:
                    self._add_op(SemanticOp(
                        op_type="sem_agg",
                        agg_expression=expr,
                        is_final=True,
                    ), pnode.node_id)

    def _do_window_agg(self, pnode: PlanNode):
        """Handle window aggregation."""
        # Window functions are computed but don't collapse rows
        # The output handling is similar to regular aggregates
        pass

    def _do_group(self, pnode: PlanNode):
        """Handle GROUP node (GROUP BY without aggregation) -> dedup."""
        if pnode.group_key:
            self._add_op(SemanticOp(
                op_type="sem_dedup",
                dedup_keys=list(pnode.group_key),
            ), pnode.node_id)

    def _do_setop(self, pnode: PlanNode):
        """Handle set operations (UNION, EXCEPT, INTERSECT)."""
        set_type = pnode.command or pnode.strategy or "Except"
        self._add_op(SemanticOp(
            op_type="sem_filter",
            set_op_type=set_type,
        ), pnode.node_id)

    def _do_result(self, pnode: PlanNode):
        """Handle Result node - scalar composition."""
        if pnode.output:
            self._add_op(SemanticOp(
                op_type="sem_map",
                expression="; ".join(pnode.output),
                is_final=True,
            ), pnode.node_id)

    def _add_final_operations(self):
        """Add final operations based on parsed SQL SELECT clause."""
        parsed = self._parsed_sql
        select = parsed.select_clause
        
        # Determine if topk is present and whether it's final
        has_topk = parsed.order_by is not None or parsed.limit is not None
        topk_is_final = select.is_select_star
        
        # If topk is cumulative (not SELECT *), add it to cumulative ops
        if has_topk and not topk_is_final:
            self._main_cumulative.append(SemanticOp(
                op_type="sem_topk",
                sort_keys=parsed.order_by_exprs if parsed.order_by_exprs else None,
                limit_count=parsed.limit,
                is_final=False,
            ))
        
        # Add final operations based on SELECT clause items
        if select.is_select_star:
            # SELECT * with ORDER BY LIMIT -> topk is final
            if has_topk:
                self._main_final.append(SemanticOp(
                    op_type="sem_topk",
                    sort_keys=parsed.order_by_exprs if parsed.order_by_exprs else None,
                    limit_count=parsed.limit,
                    is_final=True,
                ))
        else:
            # Process each SELECT item
            for col_ref in select.column_refs:
                self._main_final.append(SemanticOp(
                    op_type="sem_extract",
                    extract_column=col_ref,
                    is_final=True,
                ))
            
            for agg_expr in select.aggregations:
                self._main_final.append(SemanticOp(
                    op_type="sem_agg",
                    agg_expression=agg_expr,
                    is_final=True,
                ))


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

def parse_explain_json(explain_json: list, sql: str = "") -> dict:
    """
    Parse PostgreSQL EXPLAIN (FORMAT JSON) output and build both
    the physical operator DAG and the semantic pipeline.

    Args:
        explain_json: PostgreSQL EXPLAIN JSON output
        sql: The original SQL query (required for semantic pipeline)

    Returns dict with:
        - operator_sequence, nodes, edges, root_id (physical DAG)
        - planning_time, execution_time
        - semantic_pipeline: Pipeline format with cumulative/final ops
    """
    parser = ExplainParser()
    result = parser.parse(explain_json)

    if parser.nodes and sql:
        pipeline_builder = SemanticPipelineBuilder(parser.nodes, sql)
        result["semantic_pipeline"] = pipeline_builder.build()
    else:
        result["semantic_pipeline"] = None

    return result


def build_semantic_pipeline(physical_nodes: list[PlanNode], sql: str) -> dict:
    """Build semantic pipelines from physical nodes and SQL."""
    builder = SemanticPipelineBuilder(physical_nodes, sql)
    return builder.build()


# ═══════════════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def format_plan_tree(dag: dict, indent: int = 2) -> str:
    """Format a physical plan DAG as an indented text tree."""
    if not dag["nodes"]:
        return "(empty plan)"

    nodes_by_id = {node["id"]: node for node in dag["nodes"]}

    def get_depth(node_id: int, depths: dict) -> int:
        if node_id in depths:
            return depths[node_id]
        node = nodes_by_id[node_id]
        if "parent" not in node or node["parent"] is None:
            depths[node_id] = 0
        else:
            depths[node_id] = get_depth(node["parent"], depths) + 1
        return depths[node_id]

    depths: dict[int, int] = {}
    for node in dag["nodes"]:
        get_depth(node["id"], depths)

    sorted_nodes = sorted(
        dag["nodes"],
        key=lambda n: (depths.get(n["id"], 0), n["id"]),
    )

    lines = []
    for node in sorted_nodes:
        depth = depths.get(node["id"], 0)
        prefix = " " * (indent * depth) + "--  "

        node_str = node["node_type"]
        if node.get("relation_name"):
            node_str += f" on {node['relation_name']}"
        if node.get("alias") and node.get("alias") != node.get("relation_name"):
            node_str += f" ({node['alias']})"
        if node.get("index_name"):
            node_str += f" using {node['index_name']}"
        if node.get("join_type"):
            node_str = f"{node['join_type']} {node_str}"

        filter_parts = []
        if node.get("filter"):
            filter_parts.append(f"Filter: {node['filter']}")
        if node.get("index_cond"):
            filter_parts.append(f"Index Cond: {node['index_cond']}")
        if node.get("join_filter"):
            filter_parts.append(f"Join Filter: {node['join_filter']}")
        if node.get("hash_cond"):
            filter_parts.append(f"Hash Cond: {node['hash_cond']}")
        if node.get("merge_cond"):
            filter_parts.append(f"Merge Cond: {node['merge_cond']}")
        if filter_parts:
            node_str += "  " + "  ".join(filter_parts)

        lines.append(prefix + node_str)

    return "\n".join(lines)


def format_semantic_pipeline(pipeline_data: dict) -> str:
    """Format semantic pipeline as a readable text representation."""
    if not pipeline_data:
        return "(no pipeline data)"
    
    lines = []
    
    # Format subquery pipelines
    subquery_pipelines = pipeline_data.get("subquery_pipelines", [])
    for sp in subquery_pipelines:
        lines.append(f"=== Subquery Pipeline ({sp.get('subquery_var', '?')}) ===")
        
        cumulative = sp.get("cumulative_ops", [])
        if cumulative:
            lines.append("  Cumulative Operations:")
            for op in cumulative:
                lines.append(f"    - {_format_op(op)}")
        
        final = sp.get("final_ops", [])
        if final:
            lines.append("  Final Operations:")
            for op in final:
                lines.append(f"    - {_format_op(op)}")
        lines.append("")
    
    # Format main pipeline
    main = pipeline_data.get("main_pipeline", {})
    lines.append("=== Main Pipeline ===")
    
    cumulative = main.get("cumulative_ops", [])
    if cumulative:
        lines.append("  Cumulative Operations:")
        for op in cumulative:
            lines.append(f"    - {_format_op(op)}")
    
    final = main.get("final_ops", [])
    if final:
        lines.append("  Final Operations:")
        for op in final:
            lines.append(f"    - {_format_op(op)}")
    
    return "\n".join(lines)


def _format_op(op: dict) -> str:
    """Format a single operation for display."""
    parts = [op.get("op_type", "?")]
    
    if op.get("condition"):
        cond = op["condition"]
        if len(cond) > 50:
            cond = cond[:47] + "..."
        parts.append(f"cond={cond}")
    if op.get("join_condition"):
        parts.append(f"on={op['join_condition']}")
    if op.get("group_keys"):
        parts.append(f"group_by={op['group_keys']}")
    if op.get("agg_expression"):
        expr = op["agg_expression"]
        if len(expr) > 40:
            expr = expr[:37] + "..."
        parts.append(f"agg={expr}")
    if op.get("extract_column"):
        parts.append(f"column={op['extract_column']}")
    if op.get("sort_keys"):
        parts.append(f"order_by={op['sort_keys']}")
    if op.get("limit_count") is not None:
        parts.append(f"limit={op['limit_count']}")
    if op.get("subquery_ref"):
        parts.append(f"ref={op['subquery_ref']}")
    if op.get("is_final"):
        parts.append("[FINAL]")
    
    return "  ".join(parts)


def get_operator_summary(dag: dict) -> dict:
    """Get a summary of operators in the physical plan."""
    if not dag["nodes"]:
        return {
            "total_operators": 0,
            "operator_counts": {},
            "scans": [],
            "joins": [],
            "aggregates": [],
            "sorts": [],
            "other": [],
        }

    operator_counts: dict[str, int] = {}
    scans: list[dict] = []
    joins: list[dict] = []
    aggregates: list[dict] = []
    sorts: list[dict] = []
    other: list[dict] = []

    agg_types = AGGREGATE_NODES | {"WindowAgg"}

    for node in dag["nodes"]:
        node_type = node["node_type"]
        operator_counts[node_type] = operator_counts.get(node_type, 0) + 1

        if node_type in SCAN_NODES:
            scans.append(node)
        elif node_type in JOIN_NODES:
            joins.append(node)
        elif node_type in agg_types:
            aggregates.append(node)
        elif node_type in SORT_NODES:
            sorts.append(node)
        else:
            other.append(node)

    return {
        "total_operators": len(dag["nodes"]),
        "operator_counts": operator_counts,
        "scans": scans,
        "joins": joins,
        "aggregates": aggregates,
        "sorts": sorts,
        "other": other,
    }


def format_aggregate_info(dag: dict) -> str:
    """Format verbose aggregate info: GROUP BY and computed expressions."""
    summary = get_operator_summary(dag)
    aggregates = summary.get("aggregates", [])
    if not aggregates:
        return ""
    lines = []
    for i, node in enumerate(aggregates):
        group_key = node.get("group_key") or []
        output = node.get("output") or []
        agg_exprs = [o for o in output if o not in group_key]
        parts = []
        if group_key:
            parts.append(f"GROUP BY {', '.join(group_key)}")
        if agg_exprs:
            parts.append(f"computes {', '.join(agg_exprs)}")
        if parts:
            lines.append(f"  Aggregate {i + 1}: {'; '.join(parts)}")
    return "\n".join(lines) if lines else ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Demo / smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    # Sample query
    sample_sql = """SELECT T1.CustomerID FROM customers T1 INNER JOIN yearmonth T2
       ON T1.CustomerID = T2.CustomerID
       WHERE T1.Segment='LAM' AND SUBSTR(T2.Date,1,4)='2012'
       GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1"""

    sample_explain = [
        {
            "Plan": {
                "Node Type": "Limit",
                "Plan Rows": 1,
                "Plan Width": 12,
                "Output": ["t1.customerid", "(sum(t2.consumption))"],
                "Plans": [
                    {
                        "Node Type": "Sort",
                        "Parent Relationship": "Outer",
                        "Plan Rows": 215,
                        "Plan Width": 12,
                        "Output": ["t1.customerid", "(sum(t2.consumption))"],
                        "Sort Key": ["(sum(t2.consumption)) NULLS FIRST"],
                        "Plans": [
                            {
                                "Node Type": "Aggregate",
                                "Strategy": "Sorted",
                                "Partial Mode": "Simple",
                                "Parent Relationship": "Outer",
                                "Plan Rows": 215,
                                "Plan Width": 12,
                                "Output": ["t1.customerid", "sum(t2.consumption)"],
                                "Group Key": ["t1.customerid"],
                                "Plans": [
                                    {
                                        "Node Type": "Merge Join",
                                        "Parent Relationship": "Outer",
                                        "Join Type": "Inner",
                                        "Plan Rows": 215,
                                        "Plan Width": 12,
                                        "Merge Cond": "(t1.customerid = t2.customerid)",
                                        "Output": ["t1.customerid", "t2.consumption"],
                                        "Plans": [
                                            {
                                                "Node Type": "Sort",
                                                "Parent Relationship": "Outer",
                                                "Plan Rows": 3649,
                                                "Sort Key": ["t1.customerid"],
                                                "Output": ["t1.customerid"],
                                                "Plans": [
                                                    {
                                                        "Node Type": "Seq Scan",
                                                        "Parent Relationship": "Outer",
                                                        "Relation Name": "customers",
                                                        "Alias": "t1",
                                                        "Plan Rows": 3649,
                                                        "Output": ["t1.customerid"],
                                                        "Filter": "(t1.segment = 'LAM'::text)",
                                                    }
                                                ],
                                            },
                                            {
                                                "Node Type": "Sort",
                                                "Parent Relationship": "Inner",
                                                "Plan Rows": 1916,
                                                "Sort Key": ["t2.customerid"],
                                                "Output": ["t2.consumption", "t2.customerid"],
                                                "Plans": [
                                                    {
                                                        "Node Type": "Seq Scan",
                                                        "Parent Relationship": "Outer",
                                                        "Relation Name": "yearmonth",
                                                        "Alias": "t2",
                                                        "Plan Rows": 1916,
                                                        "Output": ["t2.consumption", "t2.customerid"],
                                                        "Filter": "(substr(t2.date, 1, 4) = '2012'::text)",
                                                    }
                                                ],
                                            },
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            "Planning Time": 0.5,
        }
    ]

    dag = parse_explain_json(sample_explain, sample_sql)

    print("Physical Plan Tree:")
    print(format_plan_tree(dag))
    print()

    print(f"Operator Sequence: {dag['operator_sequence']}")
    print()

    print("Semantic Pipeline:")
    print(format_semantic_pipeline(dag["semantic_pipeline"]))
    print()

    print("Pipeline JSON:")
    print(json.dumps(dag["semantic_pipeline"], indent=2))
