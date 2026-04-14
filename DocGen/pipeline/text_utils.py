"""
Shared text processing functions for the pipeline.

Provides field cleaning, LLM response sanitisation, value classification
helpers, placeholder counting, hash stripping, field metadata identification,
and detection pattern constants for null/binary modes.
"""

import re
from typing import Dict, List, Tuple

from .config import MISC_CHARACTERS


EXPLICIT_NULL_PATTERNS = [
    'null', 'none', 'n/a', 'na', 'nil', 'empty'
]

IMPLICIT_NULL_PATTERNS = [
    'not specified', 'is unspecified', 'remains unspecified', 'is not detailed',
    'unspecified', 'no specified', 'an unspecified', 'not detailed',
    'lacks a specified', 'has not been specified', 'does not specify',
    'remains not specified', 'is not specified', 'is absent',
    'not available', 'unavailable', 'unknown', 'not provided', 'not given',
    'missing', 'not recorded', 'not listed', 'not mentioned', 'not defined',
    'information unavailable', 'data not provided', 'not disclosed',
    'undisclosed', 'left blank', 'not entered', 'no information available',
    'details not provided'
]

EXPLICIT_BINARY_TRUE_PATTERNS = ['1', 'true', 'yes', 'y', 't']
EXPLICIT_BINARY_FALSE_PATTERNS = ['0', 'false', 'no', 'n', 'f']

IMPLICIT_BINARY_TRUE_PATTERNS = [
    'is a', 'operates as a', 'functions as a',
    'is designated as a', 'serves as a', 'operates virtually', 'provides',
    'is currently active', 'is in active status', 'is operational', 'remains active',
    'is active', 'is enabled', 'is confirmed', 'is approved', 'is verified',
    'is valid', 'is applicable', 'designated as such', 'classified as such',
    'functions as', 'serves as', 'has been confirmed'
]

IMPLICIT_BINARY_FALSE_PATTERNS = [
    'does not operate as a', 'is not a', 'is not designated as a',
    'does not function as a', 'does not operate virtually', 'does not provide',
    'is currently inactive', 'is not active', 'is not operational',
    'has inactive status', 'is inactive', 'is disabled', 'not confirmed',
    'not approved', 'not verified', 'not valid', 'not applicable',
    'not designated as such', 'not classified as such',
    'does not operate as', 'does not function as', 'does not serve as',
    'has not been confirmed'
]


def get_detection_patterns(
    null_mode: str, binary_mode: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (null_patterns, binary_true_patterns, binary_false_patterns) for the
    given mode combination.
    """
    if null_mode == "explicit":
        null_patterns = EXPLICIT_NULL_PATTERNS
    else:
        null_patterns = IMPLICIT_NULL_PATTERNS + EXPLICIT_NULL_PATTERNS

    if binary_mode == "explicit":
        binary_true_patterns = EXPLICIT_BINARY_TRUE_PATTERNS
        binary_false_patterns = EXPLICIT_BINARY_FALSE_PATTERNS
    else:
        binary_true_patterns = IMPLICIT_BINARY_TRUE_PATTERNS + EXPLICIT_BINARY_TRUE_PATTERNS
        binary_false_patterns = IMPLICIT_BINARY_FALSE_PATTERNS + EXPLICIT_BINARY_FALSE_PATTERNS

    return null_patterns, binary_true_patterns, binary_false_patterns


def clean_field_string(text: str) -> str:
    """
    Clean field names and values by removing problematic characters that could interfere with parsing.
    Handles HTML tags, quotes, backslashes, and other special characters.
    """
    if not text:
        return text

    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\\', ' ')
    text = text.replace('/', ' ')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace('`', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing miscellaneous characters and formatting.
    Returns cleaned text that can be used as replacement value.
    """
    if not response:
        return ""

    result = response.strip()
    result = result.strip('"\'`')
    result = result.replace('**', '')
    result = result.replace('*', '')
    result = re.sub(r'```[^`]*```', '', result)
    result = re.sub(r'`([^`]*)`', r'\1', result)
    result = result.strip()
    result = re.sub(r'\s+', ' ', result)

    return result


def is_misc_value(value: str) -> bool:
    """
    Check if a value consists entirely of miscellaneous characters.
    Returns True if all non-whitespace characters are in MISC_CHARACTERS set.
    """
    if not value:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    for char in stripped:
        if char not in MISC_CHARACTERS and not char.isspace():
            return False

    return True


def is_date_value(value: str) -> bool:
    """Return True if the value matches a common date format."""
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}',
        r'\d{4}/\d{2}/\d{2}',
        r'\d{1,2}-\w{3}-\d{2,4}',
        r'\w{3}\s+\d{1,2},?\s+\d{4}',
        r'\d{8}',
    ]

    for pattern in date_patterns:
        if re.match(pattern, value.strip()):
            return True

    return False


def count_placeholders(text: str) -> int:
    """Count bracket-delimited placeholders like [FIELD_NAME] in text."""
    return len(re.findall(r'\[[^\]]+\]', text))


def strip_hashes_from_text(text: str) -> str:
    """Remove all (Hash: ...) tags and pipe delimiters from text before final output."""
    text = re.sub(r'\s*\(Hash:\s*[a-f0-9]+\)', '', text)
    text = text.replace('|', '')
    return text


def identify_field_metadata(data_fields: Dict[str, str]) -> Dict[str, str]:
    """Identify field types (BINARY, NULL, MISC, STANDARD) based on values and names."""
    field_metadata = {}

    for field_name, field_value in data_fields.items():
        field_value_str = str(field_value)
        stripped_value = field_value_str.strip()

        if field_value_str.upper() in ["NULL", "NONE"] or stripped_value == "":
            field_metadata[field_name] = "NULL"
        elif is_misc_value(stripped_value):
            field_metadata[field_name] = "MISC"
        elif (('(Y/N)' in field_name or '(T/F)' in field_name) or
              (field_value_str in ['0', '1'] and not any(keyword in field_name.lower()
               for keyword in ['id', 'code', 'number', 'count', 'year', 'grade', 'reputation', 'view', 'vote', 'rank', 'score', 'size', 'amount', 'total']))):
            field_metadata[field_name] = "BINARY"
        else:
            field_metadata[field_name] = "STANDARD"

    return field_metadata
