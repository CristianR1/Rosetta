"""
Data loading functions for the pipeline.

Handles reading column descriptors, sampling table entries from text files
and SQLite databases, detecting legacy template layouts, and migrating
templates to the current split layout.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    get_mode_folder_name,
    get_ground_truth_data_path,
    get_ground_truth_table_sample_description_dir,
    get_ground_truth_column_descriptors_enhanced_path,
    get_narrative_templates_dir,
    get_output_root,
    get_sentence_templates_dir,
)

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_docgen_base_dir() -> str:
    return os.path.dirname(_PIPELINE_DIR)


def load_column_descriptors(base_dir) -> Dict[str, Any]:
    """Load the enhanced column descriptors JSON from ``GroundTruth/``."""
    column_descriptors_path = get_ground_truth_column_descriptors_enhanced_path(base_dir)
    try:
        with open(column_descriptors_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading column descriptors: {e}")
        return {}


def get_sample_entries(
    base_dir,
    db_name: str,
    table_name: str,
    num_samples: int = 1,
    dataset_folder_name: str = "MINIDEV",
) -> List[Dict[str, str]]:
    """Read sample entry text files for a given database table."""
    entries = []
    table_sample_data_dir = get_ground_truth_table_sample_description_dir(base_dir, dataset_folder_name)
    table_dir = os.path.join(table_sample_data_dir, db_name, table_name)

    if not os.path.exists(table_dir):
        print(f"Warning: Table directory not found: {table_dir}")
        return entries

    first_entry_file = f"{table_name}1.txt"
    file_path = os.path.join(table_dir, first_entry_file)

    if not os.path.exists(file_path):
        first_entry_file = f"{table_name}0.txt"
        file_path = os.path.join(table_dir, first_entry_file)

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                entry = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        entry[key.strip()] = value.strip()

                if entry:
                    entries.append(entry)
        except Exception as e:
            print(f"Error reading first entry file {file_path}: {e}")
    else:
        text_files = [f for f in os.listdir(table_dir) if f.endswith('.txt')]
        if text_files:
            fallback_file = sorted(text_files)[0]
            file_path = os.path.join(table_dir, fallback_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    entry = {}
                    for line in content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            entry[key.strip()] = value.strip()

                    if entry:
                        entries.append(entry)
            except Exception as e:
                print(f"Error reading fallback file {file_path}: {e}")

    return entries


def get_table_row_count(
    database_name: str,
    table_name: str,
    minidev_db_path: str = None,
    base_dir: str = None,
    dataset_folder_name: str = "MINIDEV",
) -> int:
    """
    Fetch the total row count from a specified table in the MINIDEV database.

    Args:
        database_name: The name of the database (for logging/validation purposes)
        table_name: The name of the table to count rows from
        minidev_db_path: Path to the SQLite file (auto-detected under GroundTruth if None)
        base_dir: DocGen directory for resolving GroundTruth (defaults to this package's DocGen parent)
        dataset_folder_name: Subfolder under ``GroundTruth/`` (default MINIDEV)

    Returns:
        Total number of rows in the table

    Raises:
        sqlite3.Error: If there's an error connecting to the database or executing the query
        ValueError: If the table doesn't exist
    """
    try:
        if minidev_db_path is None:
            bd = base_dir if base_dir is not None else _default_docgen_base_dir()
            minidev_db_path = os.path.join(
                get_ground_truth_data_path(bd, dataset_folder_name),
                "dev_databases",
                database_name,
                f"{database_name}.sqlite",
            )

        print(f"Connecting to database: {minidev_db_path}")

        conn = sqlite3.connect(minidev_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))

        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in database '{database_name}'")

        query = f"SELECT COUNT(*) FROM [{table_name}]"
        cursor.execute(query)

        row_count = cursor.fetchone()[0]

        print(f"Database: {database_name}")
        print(f"Table: {table_name}")
        print(f"Total row count: {row_count:,}")

        return row_count

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Error fetching row count: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def get_sample_data_from_database(
    db_name: str,
    table_name: str,
    limit: int = 10,
    base_dir: str = None,
    dataset_folder_name: str = "MINIDEV",
) -> List[Dict[str, str]]:
    """Get sample data rows from the database."""
    try:
        bd = base_dir if base_dir is not None else _default_docgen_base_dir()
        minidev_db_path = os.path.join(
            get_ground_truth_data_path(bd, dataset_folder_name),
            "dev_databases",
            db_name,
            f"{db_name}.sqlite",
        )

        if not os.path.exists(minidev_db_path):
            raise FileNotFoundError(f"Database file not found: {minidev_db_path}")

        conn = sqlite3.connect(minidev_db_path)
        cursor = conn.cursor()

        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [row[1] for row in cursor.fetchall()]

        cursor.execute(f"SELECT * FROM [{table_name}] LIMIT {limit}")
        rows = cursor.fetchall()

        sample_data = []
        for row in rows:
            row_dict = {}
            for i, value in enumerate(row):
                column_name = columns[i]
                row_dict[column_name] = str(value) if value is not None else "NULL"
            sample_data.append(row_dict)

        conn.close()
        return sample_data

    except Exception as e:
        print(f"Error getting sample data: {e}")
        return []


def is_complex_embedded_value(value: str) -> bool:
    """Return True if the value looks like an embedded dict/list/JSON structure."""
    if not value or not isinstance(value, str):
        return False
    value_stripped = value.strip()
    if len(value_stripped) < 10:
        return False
    dict_indicators = [
        "{'", '{"', "': ", '": ', "': {", '": {', "': [", '": [',
        "':", '":', "}, ", '}, ', "], ", '], '
    ]
    list_indicators = [
        "[{", "{'", '["', "['"
    ]
    nested_patterns = [
        "{'", '{"', "[{", "[[", "{{",
        "':", '":', "}: ", '}: ', "]: ", ']: '
    ]
    dict_count = sum(1 for ind in dict_indicators if ind in value_stripped)
    if dict_count >= 3:
        return True
    list_count = sum(1 for ind in list_indicators if ind in value_stripped)
    if list_count >= 2:
        return True
    nested_count = sum(1 for pat in nested_patterns if pat in value_stripped)
    if nested_count >= 4:
        return True
    if value_stripped.startswith('{') and ':' in value_stripped:
        brace_depth = 0
        for char in value_stripped:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
            if brace_depth > 1:
                return True
    if value_stripped.startswith('[') and '{' in value_stripped:
        return True
    return False


def detect_complex_columns(entry: Dict[str, str]) -> List[str]:
    """Return column names whose values appear to be complex embedded structures."""
    complex_columns = []
    for column_name, column_value in entry.items():
        if is_complex_embedded_value(str(column_value)):
            complex_columns.append(column_name)
    return complex_columns


def detect_legacy_template_layout(base_dir: str, null_mode: str, binary_mode: str) -> bool:
    """
    Detect if the old (pre-split) template layout exists.

    Old layout: DocSets/templates/{mode_folder}/{db}/{table}_template.json (with inline narrative)
    New layout: DocSets/templates/{mode_folder}/sentence_templates/{db}/{table}_template.json

    Returns True if legacy layout is detected (templates exist directly under mode folder,
    not under sentence_templates).
    """
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    mode_dir = os.path.join(get_output_root(base_dir), "templates", mode_folder)
    new_sentence_dir = os.path.join(mode_dir, "sentence_templates")

    if not os.path.isdir(mode_dir):
        return False

    if os.path.isdir(new_sentence_dir):
        return False

    for item in os.listdir(mode_dir):
        item_path = os.path.join(mode_dir, item)
        if os.path.isdir(item_path) and item not in ['sentence_templates', 'narrative_templates']:
            for file in os.listdir(item_path):
                if file.endswith('_template.json'):
                    return True
    return False


def migrate_legacy_templates(base_dir: str, null_mode: str, binary_mode: str,
                             data_noise_x: int = 1, data_noise_y: int = 0) -> Dict[str, Any]:
    """
    Migrate templates from old layout to new split layout.

    Reads templates from DocSets/templates/{mode}/{db}/{table}_template.json
    Writes sentence data to DocSets/templates/{mode}/sentence_templates/{db}/{table}_template.json
    Writes narrative data to DocSets/templates/{mode}/narrative_templates/{X}_data_{Y}_noise/{db}/{table}_template.json
    """
    mode_folder = get_mode_folder_name(null_mode, binary_mode)
    old_dir = os.path.join(get_output_root(base_dir), "templates", mode_folder)
    new_sentence_dir = get_sentence_templates_dir(base_dir, null_mode, binary_mode)
    new_narrative_dir = get_narrative_templates_dir(base_dir, null_mode, binary_mode, data_noise_x, data_noise_y)

    results = {'migrated': 0, 'failed': 0, 'skipped': 0, 'details': []}

    if not os.path.isdir(old_dir):
        print(f"Legacy templates directory not found: {old_dir}")
        return results

    Path(new_sentence_dir).mkdir(parents=True, exist_ok=True)
    if data_noise_y > 0:
        Path(new_narrative_dir).mkdir(parents=True, exist_ok=True)

    for db_name in os.listdir(old_dir):
        db_path = os.path.join(old_dir, db_name)
        if not os.path.isdir(db_path) or db_name in ['sentence_templates', 'narrative_templates']:
            continue

        for file in os.listdir(db_path):
            if not file.endswith('_template.json'):
                continue

            old_file = os.path.join(db_path, file)
            table_name = file.replace('_template.json', '')

            try:
                with open(old_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)

                sentence_data = {
                    'database': old_data.get('database', db_name),
                    'table': old_data.get('table', table_name),
                    'entry_index': old_data.get('entry_index', 0),
                    'original_data': old_data.get('original_data', {}),
                    'generated_sentences': old_data.get('generated_sentences', []),
                    'null_mode': old_data.get('null_mode', null_mode),
                    'binary_mode': old_data.get('binary_mode', binary_mode),
                    'data_noise_ratio': f"{data_noise_x}:{data_noise_y}",
                    'tbd_columns': old_data.get('tbd_columns', []),
                    'hash_to_column': old_data.get('hash_to_column', {}),
                    'timestamp': old_data.get('timestamp', '')
                }

                sentence_file = os.path.join(new_sentence_dir, db_name, f"{table_name}_template.json")
                Path(os.path.dirname(sentence_file)).mkdir(parents=True, exist_ok=True)
                with open(sentence_file, 'w', encoding='utf-8') as f:
                    json.dump(sentence_data, f, indent=2, ensure_ascii=False)

                if data_noise_y > 0:
                    narrative_data = {
                        'database': old_data.get('database', db_name),
                        'table': old_data.get('table', table_name),
                        'narrative': old_data.get('narrative', []),
                        'data_noise_ratio': f"{data_noise_x}:{data_noise_y}",
                        'hash_to_replacement': old_data.get('hash_to_replacement', {}),
                        'timestamp': old_data.get('timestamp', '')
                    }
                    narrative_file = os.path.join(new_narrative_dir, db_name, f"{table_name}_template.json")
                    Path(os.path.dirname(narrative_file)).mkdir(parents=True, exist_ok=True)
                    with open(narrative_file, 'w', encoding='utf-8') as f:
                        json.dump(narrative_data, f, indent=2, ensure_ascii=False)

                results['migrated'] += 1
                results['details'].append(f"{db_name}.{table_name}: migrated")

            except Exception as e:
                results['failed'] += 1
                results['details'].append(f"{db_name}.{table_name}: failed - {e}")

    print(f"Migration complete: {results['migrated']} migrated, {results['failed']} failed")
    return results
