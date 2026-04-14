"""
Shared dataclasses used across the pipeline.

Provides typed containers for sentence templates, column analysis results,
and narrative analysis results.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SentenceTemplate:
    """Represents a sentence template with variations."""
    original: str
    template_pattern: str
    primary_data_fields: List[str]
    foreign_data_fields: List[str]
    variations: List[str]
    counter_variations: List[str]
    lexical_sets: Dict[str, List[str]]
    field_data_types: Dict[str, str] = None
    is_static: bool = False
    null_variations: List[str] = None


@dataclass
class ColumnAnalysis:
    """Analysis results for a single column in a narrative."""
    column_name: str
    column_value: str
    detected: bool
    detection_method: str
    matched_text: str
    confidence: str
    field_type: str = ""
    detected_sentence: str = ""
    replacement_attempted: bool = False
    replacement_succeeded: bool = False
    replaced_sentence: str = ""


@dataclass
class NarrativeAnalysis:
    """Analysis results for a complete narrative."""
    database: str
    table: str
    total_columns: int
    detected_columns: int
    undetected_columns: int
    detection_rate: float
    column_analyses: List[ColumnAnalysis]
    narrative_text: str
