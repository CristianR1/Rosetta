"""TemplateGenerator - sentence and narrative template generation."""

import argparse
import json
import math
import openai
import random
import re
import time
import os
import sys
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cost_tracker import get_cost_tracker, track_openai_response

from .config import (
    get_mode_folder_name, get_noise_ratio_folder_name,
    get_sentence_template_path, get_narrative_template_path,
    get_sentence_templates_dir, get_narrative_templates_dir,
    variation_templates_exist, parse_data_noise_ratio_str,
    parse_ratio_folder_name, find_narrative_template_path_scan,
    resolve_backfill_noise_xy_and_narrative_path,
    compute_expected_transition_count,
    get_ground_truth_data_path,
    get_ground_truth_table_sample_description_dir,
    get_ground_truth_column_descriptors_enhanced_path,
    load_repo_dotenv,
)
from .data_loader import (
    load_column_descriptors as _load_column_descriptors,
    detect_complex_columns, detect_legacy_template_layout,
    migrate_legacy_templates,
)
from .narrative_generation import NarrativeGenerationMixin

cost_tracker = get_cost_tracker()

load_repo_dotenv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


class TemplateGenerator(NarrativeGenerationMixin):
    def __init__(self, base_dir: str = None, null_mode: str = "implicit", binary_mode: str = "implicit",
                 data_noise_x: int = 1, data_noise_y: int = 0, dataset_folder_name: str = "MINIDEV"):
        if base_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)

        self.base_dir = base_dir
        self.dataset_folder_name = dataset_folder_name
        self.table_sample_data_dir = get_ground_truth_table_sample_description_dir(
            base_dir, dataset_folder_name
        )
        self.column_descriptors_path = get_ground_truth_column_descriptors_enhanced_path(base_dir)
        self.minidev_path = os.path.join(
            get_ground_truth_data_path(base_dir, dataset_folder_name), "dev_databases"
        )

        self.null_mode = null_mode
        self.binary_mode = binary_mode
        self.data_noise_x = data_noise_x
        self.data_noise_y = data_noise_y

        self.sentence_templates_dir = get_sentence_templates_dir(base_dir, null_mode, binary_mode)
        self.narrative_templates_dir = get_narrative_templates_dir(base_dir, null_mode, binary_mode, data_noise_x, data_noise_y)

        self.output_dir = self.sentence_templates_dir

        Path(self.sentence_templates_dir).mkdir(parents=True, exist_ok=True)
        if data_noise_y > 0:
            Path(self.narrative_templates_dir).mkdir(parents=True, exist_ok=True)

        self.column_descriptors = self.load_column_descriptors()

    @staticmethod
    def generate_sentence_hash() -> str:
        return uuid.uuid4().hex[:16]

    @staticmethod
    def append_hash(sentence: str, hash_value: str = None) -> str:
        if hash_value is None:
            hash_value = TemplateGenerator.generate_sentence_hash()
        if not sentence.rstrip().endswith(')') or '(Hash:' not in sentence:
            return f"{sentence.rstrip()} (Hash: {hash_value})"
        return sentence

    @staticmethod
    def extract_hash(sentence: str) -> Optional[str]:
        m = re.search(r'\(Hash:\s*([a-f0-9]+)\)', sentence)
        return m.group(1) if m else None

    @staticmethod
    def extract_all_hashes(sentences: List[str]) -> List[str]:
        hashes = []
        for s in sentences:
            h = TemplateGenerator.extract_hash(s)
            if h:
                hashes.append(h)
        return hashes

    @staticmethod
    def verify_hashes_in_narrative(narrative_text: str, expected_hashes: List[str]) -> Tuple[bool, List[str]]:
        missing = [h for h in expected_hashes if f'(Hash: {h})' not in narrative_text]
        return (len(missing) == 0, missing)

    @staticmethod
    def verify_delimiters_in_narrative(narrative_text: str) -> bool:
        """Return True if narrative has well-formed pipe delimiters: even count, non-zero."""
        pipe_count = narrative_text.count('|')
        return pipe_count > 0 and pipe_count % 2 == 0

    @staticmethod
    def needs_backfill(
        template_data: Dict[str, Any],
        base_dir: str = None,
        null_mode: str = None,
        binary_mode: str = None,
        data_noise_x: int = None,
        data_noise_y: int = None,
    ) -> bool:
        """
        Return True if sentences or narrative are missing hashes.

        For ratio-aware checking: if data_noise_y > 0, the narrative must exist at
        the correct ratio path (e.g. 1_data_1_noise). If the narrative file does not
        exist at that path, returns False (nothing to backfill yet for that ratio).

        Also returns False when sentence variations already exist for this
        table (under any ratio folder), since backfilling would overwrite
        hashes that the variation templates depend on.
        """
        if base_dir and null_mode is not None and binary_mode is not None:
            db = template_data.get('database', '')
            table = template_data.get('table', '')
            if db and table and variation_templates_exist(base_dir, null_mode, binary_mode, db, table):
                return False

        generated_sentences = template_data.get('generated_sentences', [])
        filtered = [s for s in generated_sentences if s != "TBD"]
        if not filtered:
            return False

        if any(TemplateGenerator.extract_hash(s) is None for s in filtered):
            return True

        expected_hashes = TemplateGenerator.extract_all_hashes(filtered)
        if not expected_hashes:
            return False

        db = template_data.get('database', '')
        table = template_data.get('table', '')

        narrative_text = ''
        narrative_path: Optional[str] = None

        if base_dir and null_mode is not None and binary_mode is not None:
            if data_noise_x is not None and data_noise_y is not None and data_noise_y > 0:
                narrative_path = get_narrative_template_path(
                    base_dir, null_mode, binary_mode, data_noise_x, data_noise_y, db, table
                )
                if not os.path.isfile(narrative_path):
                    return False
            else:
                narrative_path = find_narrative_template_path_scan(
                    base_dir, null_mode, binary_mode, db, table
                )
                if not narrative_path or not os.path.isfile(narrative_path):
                    sx, sy = parse_data_noise_ratio_str(
                        template_data.get('data_noise_ratio'), 1, 0
                    )
                    if sy == 0:
                        return False

            if narrative_path and os.path.isfile(narrative_path):
                try:
                    with open(narrative_path, 'r', encoding='utf-8') as nf:
                        nd = json.load(nf)
                    narr = nd.get('narrative', [])
                    narrative_text = ' '.join(narr) if isinstance(narr, list) else str(narr or '')
                except Exception:
                    pass

        if not narrative_text.strip():
            narrative = template_data.get('narrative', [])
            narrative_text = ' '.join(narrative) if isinstance(narrative, list) else str(narrative or '')

        if not narrative_text.strip():
            return False

        all_present, _ = TemplateGenerator.verify_hashes_in_narrative(narrative_text, expected_hashes)
        if not all_present:
            return True
        if not TemplateGenerator.verify_delimiters_in_narrative(narrative_text):
            return True
        return False

    @staticmethod
    def strip_hash(sentence: str) -> str:
        return re.sub(r'\s*\(Hash:\s*[a-f0-9]+\)', '', sentence).rstrip()

    @staticmethod
    def count_sentences_in_narrative(narrative_text: str) -> Tuple[int, int]:
        """
        Count total sentences and hashed sentences in a pipe-delimited narrative.
        Returns (total_sentences, hashed_sentences).
        """
        segments = re.findall(r'\|\s*([^|]+?)\s*\|', narrative_text)
        total = len(segments)
        hashed = sum(1 for seg in segments if '(Hash:' in seg)
        return (total, hashed)

    @staticmethod
    def verify_transition_count(narrative_text: str, expected_hashes: List[str], expected_transitions: int) -> Tuple[bool, str]:
        """
        Verify that the narrative has the expected number of transition sentences.
        Returns (is_valid, error_message).
        """
        total, hashed = TemplateGenerator.count_sentences_in_narrative(narrative_text)
        actual_transitions = total - hashed

        if hashed != len(expected_hashes):
            return (False, f"hash count mismatch: expected {len(expected_hashes)}, got {hashed}")
        if expected_transitions > 0 and actual_transitions != expected_transitions:
            return (False, f"transition count mismatch: expected {expected_transitions}, got {actual_transitions}")
        return (True, "")

    def load_column_descriptors(self) -> Dict[str, Any]:
        try:
            with open(self.column_descriptors_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading column descriptors: {e}")
            return {}

    def query_gpt4o(self, prompt: str, system_message: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )
                track_openai_response(response, "gpt-4o")
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"Error querying GPT-4o after {max_retries} attempts: {str(e)}"

    def get_all_databases_and_tables(self) -> Dict[str, List[str]]:
        databases_tables = {}

        if not os.path.exists(self.table_sample_data_dir):
            print(f"Error: table_sample_data directory not found at {self.table_sample_data_dir}")
            return databases_tables

        for db_name in os.listdir(self.table_sample_data_dir):
            db_path = os.path.join(self.table_sample_data_dir, db_name)
            if os.path.isdir(db_path):
                tables = []
                for table_name in os.listdir(db_path):
                    table_path = os.path.join(db_path, table_name)
                    if os.path.isdir(table_path):
                        tables.append(table_name)

                if tables:
                    databases_tables[db_name] = sorted(tables)

        return databases_tables

    def get_sample_entries(self, db_name: str, table_name: str, num_samples: int = 1) -> List[Dict[str, str]]:
        entries = []
        table_dir = os.path.join(self.table_sample_data_dir, db_name, table_name)

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

    def format_column_value(self, column_name: str, column_value: str) -> str:
        column_value_str = str(column_value)
        column_name_lower = column_name.lower()

        if column_value_str.upper() in ["NULL", "NONE", "N/A", ""]:
            if self.null_mode == "explicit":
                return "NULL"
            else:
                return "not specified"

        if 'percent' in column_name_lower or '%' in column_name_lower:
            try:
                numeric_value = float(column_value_str)
                if 0 <= numeric_value <= 1:
                    percentage = numeric_value * 100
                    return f"{percentage:.2f}%"
                elif numeric_value > 1:
                    return f"{numeric_value:.2f}%"
            except ValueError:
                pass

        try:
            numeric_value = float(column_value_str)

            if column_value_str.startswith('0') and '.' not in column_value_str:
                return column_value_str

            if numeric_value == int(numeric_value):
                return str(int(numeric_value))
            else:
                return f"{numeric_value:.2f}"
        except ValueError:
            pass

        return column_value_str

    def is_binary_value(self, value: str) -> bool:
        return str(value).strip() in ['0', '1', 'true', 'false', 'True', 'False', 'TRUE', 'FALSE']

    def is_null_value(self, value: str) -> bool:
        return str(value).strip().upper() in ['NULL', 'NONE', 'N/A', '']

    def generate_sentence_for_column_null_explicit(self, column_name: str, column_value: str, descriptor: str) -> str:
        prompt = f"""Generate a single well-formed, natural sentence that conveys the following as missing. Context: {descriptor}. Use the literal word NULL for the missing value.

        RULES:
        1. Use the literal word NULL in the sentence to indicate the missing value.
        2. Keep the sentence simple, direct, and natural.
        3. Do not use phrases like "not specified" or "not available" — use NULL exactly.
        4. Do NOT include any natural language that suggests the data comes from a column, field, database, or table. The sentence must read as ordinary prose — not detectable as coming from structured data.
        5. NEVER assume any additional or external information beyond the context above — do not invent or imply places (cities, counties, states, countries), times, institutions, people, quantities, or causal stories unless they appear explicitly in the context; stay strictly within what the context states.
        6. Return only the sentence with no markdown.
        """
        system_message = "You generate natural sentences with explicit NULL for missing data. Output must be well-formed prose that never mentions or implies columns, fields, or databases."
        return self.query_gpt4o(prompt, system_message)

    def generate_sentence_for_column_null_implicit(self, column_name: str, column_value: str, descriptor: str) -> str:
        prompt = f"""Generate a single well-formed, natural sentence that conveys the following as not available, using the context below. Context: {descriptor}

        RULES:
        1. Use natural language to indicate the value is missing (e.g. "not specified", "not available", "not provided", "unknown").
        2. Do NOT use the word NULL or None in the sentence.
        3. Keep the sentence natural and human-readable.
        4. Do NOT include any natural language that suggests the data comes from a column, field, database, or table. The sentence must read as ordinary prose — not detectable as coming from structured data.
        5. NEVER assume any additional or external information beyond the context above — do not invent or imply places (cities, counties, states, countries), times, institutions, people, quantities, or causal stories unless they appear explicitly in the context; stay strictly within what the context states.
        6. Return only the sentence with no markdown.
        """
        system_message = "You generate natural sentences for missing data. Output must be well-formed prose that never mentions or implies columns, fields, or databases."
        return self.query_gpt4o(prompt, system_message)

    def generate_sentence_for_column_binary_explicit(self, column_name: str, column_value: str, descriptor: str) -> str:
        display_value = str(column_value).strip()
        prompt = f"""Generate a single well-formed, natural sentence that incorporates the value {display_value} in a way that fits this context. Context: {descriptor}

        RULES:
        1. Use the literal value {display_value} in the sentence.
        2. Do NOT convert 0/1 to yes/no or true/false in the text.
        3. Keep the sentence direct and natural.
        4. Do NOT include any natural language that suggests the data comes from a column, field, database, or table. The sentence must read as ordinary prose — not detectable as coming from structured data.
        5. NEVER assume any additional or external information beyond the context and the value above — do not invent or imply places (cities, counties, states, countries), times, institutions, people, quantities, or causal stories unless they appear explicitly in the context or in the value {display_value}; stay strictly within what those sources state.
        6. Return only the sentence with no markdown.
        """
        system_message = "You generate natural sentences that include the given value. Output must be well-formed prose that never mentions or implies columns, fields, or databases."
        return self.query_gpt4o(prompt, system_message)

    def generate_sentence_for_column_binary_implicit(self, column_name: str, column_value: str, descriptor: str) -> str:
        is_true = str(column_value).strip() in ['1', 'true', 'True', 'TRUE']
        clean_name = column_name.replace('(Y/N)', '').replace('(T/F)', '').strip()
        prompt = f"""Generate a single well-formed, natural sentence that expresses this binary state in prose. Context: {descriptor}. The state is: {"positive/yes/true" if is_true else "negative/no/false"}.

        RULES:
        1. Use natural language to express this binary state (e.g. "is a", "operates as", "has" for positive; "is not", "does not have" for negative).
        2. Do NOT include the raw 0 or 1 value in the sentence.
        3. Do NOT include any natural language that suggests the data comes from a column, field, database, or table. The sentence must read as ordinary prose — not detectable as coming from structured data.
        4. NEVER assume any additional or external information beyond the context above and the binary state described — do not invent or imply places (cities, counties, states, countries), times, institutions, people, quantities, or causal stories unless they appear explicitly in the context; stay strictly within what the context states.
        5. Return only the sentence with no markdown.
        """
        system_message = "You generate natural sentences for binary states. Output must be well-formed prose that never mentions or implies columns, fields, or databases."
        return self.query_gpt4o(prompt, system_message)

    def generate_sentence_for_column_standard(self, column_name: str, column_value: str, descriptor: str) -> str:
        display_value = self.format_column_value(column_name, column_value)
        prompt = f"""Generate a single well-formed, natural sentence that weaves in the following value as ordinary prose in direct relation to the name associated with this value: {column_name}. Context: {descriptor}. Value to include: {display_value}.

        RULES:
            1. THE VALUE "{display_value}" MUST APPEAR LITERALLY IN YOUR SENTENCE — this is mandatory, not optional. The exact characters "{display_value}" must be present somewhere in the output.
            2. Mention the {column_name} concept naturally in the sentence.
            3. Write one sentence of about 15-25 words that reads as natural, simple, well-formed sentence that is not overly narrative or unrealistic in relation to the context of the value.
            4. Do NOT include any natural language that suggests the data comes from a column, field, database, table, or row. The sentence must not be detectable as coming from structured data — no "the X field", "according to the Y column", "the value is", or similar.
            5. Do NOT add extra explanation of the value or its source. No nicknames or replacement values — use only the provided value.
            6. NEVER assume any additional or external information beyond the context and the value above — do not invent or imply places (cities, counties, states, countries), times, institutions, people, quantities, or causal stories unless they appear explicitly in the context or in the value {display_value}; stay strictly within what those sources state.
            7. NEVER use example values from the context; use ONLY the actual value: {display_value}.
            8. Return only the sentence with no markdown or other text.

        EXAMPLE for numeric ID values:
        - If column_name is "card_id" and value is "1", a good sentence would be: "The credit card with card_id 1 is registered in the system for transaction tracking purposes."
        - If column_name is "account_id" and value is "42", a good sentence would be: "The bank account identified by account_id 42 has been active since its creation date."
        """
        system_message = "You generate natural, well-formed sentences that integrate a given value into prose. THE VALUE MUST APPEAR LITERALLY IN THE SENTENCE. Output must never mention or imply columns, fields, databases, or tables; it should read as ordinary writing."
        return self.query_gpt4o(prompt, system_message)

    def get_column_descriptor(self, column_name: str, db_name: str, table_name: str) -> str:

        table_descriptors = {}
        if db_name in self.column_descriptors and table_name in self.column_descriptors[db_name]:
            table_descriptors = self.column_descriptors[db_name][table_name]

        descriptor = ""
        if column_name in table_descriptors:
            descriptor = table_descriptors[column_name].get('descriptor', f'Column {column_name}')

        return descriptor

    def generate_sentence_for_column(self, column_name: str, column_value: str, descriptor: str, append_hash: bool = True) -> str:
        if self.is_null_value(column_value):
            if self.null_mode == "explicit":
                sentence = self.generate_sentence_for_column_null_explicit(column_name, column_value, descriptor)
            else:
                sentence = self.generate_sentence_for_column_null_implicit(column_name, column_value, descriptor)
        elif self.is_binary_value(column_value):
            if self.binary_mode == "explicit":
                sentence = self.generate_sentence_for_column_binary_explicit(column_name, column_value, descriptor)
            else:
                sentence = self.generate_sentence_for_column_binary_implicit(column_name, column_value, descriptor)
        else:
            sentence = self.generate_sentence_for_column_standard(column_name, column_value, descriptor)

        if append_hash:
            sentence = self.append_hash(sentence)
        return sentence

    def generate_sentences_for_entry(self, entry: Dict[str, str], db_name: str, table_name: str, complex_columns: List[str] = None) -> Tuple[List[str], List[str], Dict[str, str]]:
        sentences = []
        tbd_columns = []
        hash_to_column = {}

        if complex_columns is None:
            complex_columns = []

        table_descriptors = {}
        if db_name in self.column_descriptors and table_name in self.column_descriptors[db_name]:
            table_descriptors = self.column_descriptors[db_name][table_name]

        for column_name, column_value in entry.items():
            if column_name in complex_columns:
                sentences.append("TBD")
                tbd_columns.append(column_name)
                continue

            if column_name in table_descriptors:
                descriptor = table_descriptors[column_name].get('descriptor', f'Column {column_name}')
                sentence = self.generate_sentence_for_column(column_name, column_value, descriptor, append_hash=True)
            else:
                display_value = self.format_column_value(column_name, column_value)
                sentence = f"The {column_name} is {display_value}."
                sentence = self.append_hash(sentence)

            h = self.extract_hash(sentence)
            if h:
                hash_to_column[h] = column_name
            sentences.append(sentence)

        return sentences, tbd_columns, hash_to_column

    def generate_template_for_table(self, db_name: str, table_name: str) -> bool:
        print(f"\nProcessing: {db_name} -> {table_name}")

        sample_entries = self.get_sample_entries(db_name, table_name, num_samples=1)

        if not sample_entries:
            print(f"No sample entries found for {db_name}.{table_name}")
            return False

        entry = sample_entries[0]
        print(f"  Generating template...")

        complex_columns = detect_complex_columns(entry)
        if complex_columns:
            print(f"  Detected complex embedded columns: {complex_columns}")

        sentences, tbd_columns, hash_to_column = self.generate_sentences_for_entry(entry, db_name, table_name, complex_columns)

        if not sentences:
            print(f"  No sentences generated for the entry")
            return False

        filtered_sentences = [s for s in sentences if s != "TBD"]
        num_data_sentences = len(filtered_sentences)
        expected_transitions = compute_expected_transition_count(num_data_sentences, self.data_noise_x, self.data_noise_y)
        ratio_str = f"{self.data_noise_x}:{self.data_noise_y}"

        sentence_template_data = {
            'database': db_name,
            'table': table_name,
            'entry_index': 0,
            'original_data': entry,
            'generated_sentences': sentences,
            'null_mode': self.null_mode,
            'binary_mode': self.binary_mode,
            'data_noise_ratio': ratio_str,
            'tbd_columns': tbd_columns,
            'hash_to_column': hash_to_column,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        sentence_file = get_sentence_template_path(self.base_dir, self.null_mode, self.binary_mode, db_name, table_name)
        Path(os.path.dirname(sentence_file)).mkdir(parents=True, exist_ok=True)

        try:
            with open(sentence_file, 'w', encoding='utf-8') as f:
                json.dump(sentence_template_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved sentence template to: {sentence_file}")
        except Exception as e:
            print(f"  Error saving sentence template: {e}")
            return False

        if self.data_noise_y == 0:
            txt_file = sentence_file.replace('_template.json', '_sentences.txt')
            try:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    for s in filtered_sentences:
                        f.write(f"{s}\n")
                print(f"  Saved raw sentences to: {txt_file}")
            except Exception as e:
                print(f"  Warning: Could not save text dump: {e}")

            if tbd_columns:
                print(f"  TBD columns (complex values): {tbd_columns}")
            return True

        narrative = self.generate_narrative_from_sentences(sentences, db_name, table_name, expected_transitions)
        formatted_narrative = self.format_narrative_for_json(narrative)

        narrative_template_data = {
            'database': db_name,
            'table': table_name,
            'narrative': formatted_narrative,
            'data_noise_ratio': ratio_str,
            'data_sentence_count': num_data_sentences,
            'expected_transition_count': expected_transitions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        narrative_file = get_narrative_template_path(self.base_dir, self.null_mode, self.binary_mode,
                                                      self.data_noise_x, self.data_noise_y, db_name, table_name)
        Path(os.path.dirname(narrative_file)).mkdir(parents=True, exist_ok=True)

        try:
            with open(narrative_file, 'w', encoding='utf-8') as f:
                json.dump(narrative_template_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved narrative template to: {narrative_file}")
        except Exception as e:
            print(f"  Error saving narrative template: {e}")
            return False

        if tbd_columns:
            print(f"  TBD columns (complex values): {tbd_columns}")
        return True

    def generate_templates_for_tables(self, selected_tables: List[Tuple[str, str]], null_mode: str = None, binary_mode: str = None,
                                       data_noise_x: int = None, data_noise_y: int = None, skip_existing: bool = True) -> Dict[str, Any]:
        if null_mode is not None:
            self.null_mode = null_mode
        if binary_mode is not None:
            self.binary_mode = binary_mode
        if data_noise_x is not None:
            self.data_noise_x = data_noise_x
        if data_noise_y is not None:
            self.data_noise_y = data_noise_y

        self.sentence_templates_dir = get_sentence_templates_dir(self.base_dir, self.null_mode, self.binary_mode)
        self.narrative_templates_dir = get_narrative_templates_dir(self.base_dir, self.null_mode, self.binary_mode, self.data_noise_x, self.data_noise_y)
        self.output_dir = self.sentence_templates_dir

        Path(self.sentence_templates_dir).mkdir(parents=True, exist_ok=True)
        if self.data_noise_y > 0:
            Path(self.narrative_templates_dir).mkdir(parents=True, exist_ok=True)

        ratio_str = f"{self.data_noise_x}:{self.data_noise_y}"
        print(f"Generating templates with null_mode={self.null_mode}, binary_mode={self.binary_mode}, data:noise={ratio_str}")
        print(f"Processing {len(selected_tables)} tables")
        if skip_existing:
            print(f"Skipping tables with existing templates")

        results = {
            'success': True,
            'total_tables': len(selected_tables),
            'processed_tables': 0,
            'successful_tables': 0,
            'failed_tables': 0,
            'skipped_tables': 0,
            'details': {}
        }

        for db_name, table_name in selected_tables:
            results['processed_tables'] += 1

            sentence_file = get_sentence_template_path(self.base_dir, self.null_mode, self.binary_mode, db_name, table_name)
            narrative_file = get_narrative_template_path(self.base_dir, self.null_mode, self.binary_mode, self.data_noise_x, self.data_noise_y, db_name, table_name)

            sentence_exists = os.path.exists(sentence_file)
            narrative_exists = os.path.exists(narrative_file)

            if skip_existing and sentence_exists and not narrative_exists:
                print(f"\n{db_name}.{table_name}: Sentence exists but narrative missing for ratio {self.data_noise_x}:{self.data_noise_y}")
                try:
                    success = self.generate_narrative_only(sentence_file, self.data_noise_x, self.data_noise_y)
                    if success:
                        print(f"  Narrative generated for {db_name}.{table_name}")
                        results['details'][f"{db_name}.{table_name}"] = 'narrative_generated'
                        results['successful_tables'] += 1
                    else:
                        print(f"  Narrative generation failed for {db_name}.{table_name}")
                        results['details'][f"{db_name}.{table_name}"] = 'narrative_generation_failed'
                        results['failed_tables'] += 1
                except Exception as e:
                    print(f"  Error generating narrative for {db_name}.{table_name}: {e}")
                    results['details'][f"{db_name}.{table_name}"] = f'error: {str(e)}'
                    results['failed_tables'] += 1
                continue

            if skip_existing and sentence_exists and narrative_exists:
                print(f"\nSkipping {db_name}.{table_name} (templates already exist)")
                results['skipped_tables'] += 1
                try:
                    with open(sentence_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)

                    if TemplateGenerator.needs_backfill(
                        template_data, self.base_dir, self.null_mode, self.binary_mode,
                        self.data_noise_x, self.data_noise_y
                    ):
                        print(f"  Checking {db_name}.{table_name} for backfill...")
                        print(f"  Backfill needed: sentences or narrative missing hashes")
                        success = self.backfill_template_hashes(sentence_file, narrative_file)
                        if success:
                            print(f"  Backfill complete for {db_name}.{table_name}")
                            results['details'][f"{db_name}.{table_name}"] = 'skipped_backfilled'
                        else:
                            print(f"  Backfill failed for {db_name}.{table_name}")
                            results['details'][f"{db_name}.{table_name}"] = 'skipped_backfill_failed'
                    else:
                        print(f"  {db_name}.{table_name}: already has hashes, skipping backfill")
                        results['details'][f"{db_name}.{table_name}"] = 'skipped'
                except Exception as e:
                    print(f"  Error checking/backfilling {db_name}.{table_name}: {e}")
                    results['details'][f"{db_name}.{table_name}"] = 'skipped'
                continue

            try:
                success = self.generate_template_for_table(db_name, table_name)

                if success:
                    results['successful_tables'] += 1
                    results['details'][f"{db_name}.{table_name}"] = 'success'
                else:
                    results['failed_tables'] += 1
                    results['details'][f"{db_name}.{table_name}"] = 'failed'

            except Exception as e:
                print(f"Error processing {db_name}.{table_name}: {e}")
                results['failed_tables'] += 1
                results['details'][f"{db_name}.{table_name}"] = f'error: {str(e)}'

            time.sleep(1)

        cost_tracker.track_skipped_templates(results['skipped_tables'])

        print(f"\nTemplate generation complete:")
        print(f"  Generated: {results['successful_tables']}")
        print(f"  Skipped (cached): {results['skipped_tables']}")
        print(f"  Failed: {results['failed_tables']}")

        summary_file = os.path.join(self.output_dir, "generation_summary.json")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save summary report: {e}")

        return results

    def generate_all_templates(self, skip_existing: bool = True) -> Dict[str, Any]:
        """Generate templates for all tables in all databases."""
        print("Starting comprehensive template generation...")
        print(f"Output directory: {self.output_dir}")

        databases_tables = self.get_all_databases_and_tables()

        if not databases_tables:
            print("No databases found in table_sample_data directory!")
            return {'success': False, 'error': 'No databases found'}

        print(f"\nFound {len(databases_tables)} databases:")
        total_tables = 0
        for db_name, tables in databases_tables.items():
            print(f"  {db_name}: {len(tables)} tables")
            total_tables += len(tables)

        print(f"\nTotal tables to process: {total_tables}")

        all_tables = []
        for db_name, tables in databases_tables.items():
            for table_name in tables:
                all_tables.append((db_name, table_name))

        return self.generate_templates_for_tables(all_tables, skip_existing=skip_existing)

    def print_final_summary(self, results: Dict[str, Any]):
        print(f"\n{'='*80}")
        print("TEMPLATE GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total databases processed: {results.get('total_databases', 'N/A')}")
        print(f"Total tables processed: {results['processed_tables']}")
        print(f"Successful: {results['successful_tables']}")
        print(f"Failed: {results['failed_tables']}")
        print(f"Skipped: {results.get('skipped_tables', 0)}")
        print(f"\nTemplates saved to: {self.output_dir}")

        if results['failed_tables'] > 0:
            print(f"\n{results['failed_tables']} tables failed to generate templates.")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Template Generator")
    parser.add_argument("--backfill", action="store_true", help="Backfill hashes on existing templates instead of generating new ones")
    parser.add_argument("--migrate", action="store_true", help="Migrate legacy templates to new split layout")
    parser.add_argument("--data-noise-ratio", type=str, default="1:0", help="Data:Noise ratio (e.g., 5:1)")
    parser.add_argument("--null-mode", type=str, default="implicit", choices=["explicit", "implicit"])
    parser.add_argument("--binary-mode", type=str, default="implicit", choices=["explicit", "implicit"])
    args = parser.parse_args()

    print("Starting Comprehensive Template Generator")
    print("=" * 60)

    ratio_parts = args.data_noise_ratio.split(':')
    data_noise_x = int(ratio_parts[0]) if len(ratio_parts) == 2 else 1
    data_noise_y = int(ratio_parts[1]) if len(ratio_parts) == 2 else 0

    generator = TemplateGenerator(
        null_mode=args.null_mode,
        binary_mode=args.binary_mode,
        data_noise_x=data_noise_x,
        data_noise_y=data_noise_y
    )

    if args.migrate:
        print(f"Migrating legacy templates to new split layout...")
        print(f"Data:Noise ratio: {data_noise_x}:{data_noise_y}")
        if detect_legacy_template_layout(generator.base_dir, args.null_mode, args.binary_mode):
            results = migrate_legacy_templates(
                generator.base_dir, args.null_mode, args.binary_mode,
                data_noise_x, data_noise_y
            )
            print(f"\nMigration summary: Migrated={results['migrated']}, Failed={results['failed']}")
        else:
            print("No legacy templates detected (or already migrated).")
        return

    if args.backfill:
        results = generator.backfill_all_templates()
        print(f"\nBackfill summary: Backfilled={results['backfilled_count']}, Skipped={results['skipped_count']}, Failed={results['failed_count']}")
        return

    if not os.path.exists(generator.column_descriptors_path):
        print(f"Error: Column descriptors file not found at {generator.column_descriptors_path}")
        print("Please run generate_column_descriptors.py first to create the enhanced column descriptors!")
        return
    if not os.path.exists(generator.table_sample_data_dir):
        print(f"Error: Table sample data directory not found at {generator.table_sample_data_dir}")
        print("Please run generate_table_sample_text_files.py first!")
        return
    results = generator.generate_all_templates(skip_existing=True)
    generator.print_final_summary(results)

if __name__ == "__main__":
    main()
