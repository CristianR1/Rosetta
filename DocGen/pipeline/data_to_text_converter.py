"""Data-to-text conversion - generates document variations from templates."""

import json
import csv
import random
import re
import sqlite3
import os
import openai
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .config import (
    get_mode_folder_name,
    get_noise_ratio_folder_name,
    get_output_root,
    get_ground_truth_data_path,
    load_repo_dotenv,
    create_local_llm_openai_client,
    get_local_llm_model,
)

load_repo_dotenv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .text_utils import strip_hashes_from_text


class DataToTextConverter:
    
    def __init__(self, base_dir: str = None, null_mode: str = "implicit", binary_mode: str = "implicit",
                 data_noise_x: int = 1, data_noise_y: int = 0):
        if base_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
        
        self.base_dir = base_dir
        self.null_mode = null_mode
        self.binary_mode = binary_mode
        self.data_noise_x = data_noise_x
        self.data_noise_y = data_noise_y
        self.minidev_path = os.path.join(get_ground_truth_data_path(base_dir, "MINIDEV"), "dev_databases")
        
        mode_folder = get_mode_folder_name(null_mode, binary_mode)
        ratio_folder = get_noise_ratio_folder_name(data_noise_x, data_noise_y)
        out_root = get_output_root(base_dir)

        self.templates_dir = os.path.join(out_root, "variations", mode_folder, ratio_folder)
        self.sentence_templates_dir = os.path.join(out_root, "templates", mode_folder, "sentence_templates")
        self.narrative_templates_dir = os.path.join(out_root, "templates", mode_folder, "narrative_templates", ratio_folder)
        self.raw_templates_dir = self.sentence_templates_dir
        self.results_dir = os.path.join(out_root, "documents", mode_folder, ratio_folder)
        
        self.multi_column_dir = os.path.join(self.results_dir, "Multi Column")
        self.single_column_dir = os.path.join(self.results_dir, "Single Column")
        self.text_dir = os.path.join(self.results_dir, "Text")
        
        Path(self.multi_column_dir).mkdir(parents=True, exist_ok=True)
        Path(self.single_column_dir).mkdir(parents=True, exist_ok=True)
        Path(self.text_dir).mkdir(parents=True, exist_ok=True)
        
        self.field_variations = {}
        self.all_field_names = set()
        self.field_names_ordered = []
        self.database_info = {}
        self.field_types = {}
        self.static_sentences = []
        self.template_order = []
        self.tbd_columns = []
        
        self.used_sentences_global = set()
        self.used_lexical_combinations = set()
        
        self.tbd_cache_path = os.path.join(out_root, "tbd_sentence_cache.json")
        self.tbd_sentence_cache = self._load_tbd_cache()
        
        self.local_llm_client = create_local_llm_openai_client()
    
    def _load_tbd_cache(self) -> dict:
        """Load TBD sentence cache from disk if it exists."""
        if os.path.exists(self.tbd_cache_path):
            try:
                with open(self.tbd_cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {tuple(k.split("|||")): v for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Could not load TBD cache: {e}")
        return {}
    
    def _save_tbd_cache(self):
        """Save TBD sentence cache to disk."""
        try:
            data = {"|||".join(k): v for k, v in self.tbd_sentence_cache.items()}
            with open(self.tbd_cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save TBD cache: {e}")
    
    def load_sentence_templates(self, db_name: str, table_name: str) -> bool:
        template_file = os.path.join(self.templates_dir, db_name, f"{table_name}_sentence_templates.json")
        
        if not os.path.exists(template_file):
            print(f"Template file not found: {template_file}")
            return False
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading template file: {e}")
            return False
        
        templates = data.get("templates", [])
        
        self.field_variations = {}
        self.all_field_names = set()
        self.field_names_ordered = []
        self.field_types = {}
        self.static_sentences = []
        self.template_order = []
        self.tbd_columns = data.get("tbd_columns", [])
        
        if self.tbd_columns:
            print(f"  TBD columns detected (complex values): {self.tbd_columns}")
        
        for template_idx, template in enumerate(templates):
            is_static = template.get("is_static", False)
            
            variations = template.get("variations", [])
            counter_variations = template.get("counter_variations", [])
            null_variations = template.get("null_variations", [])
            lexical_sets = template.get("lexical_sets", {})

            if is_static:
                static_sentence = template.get("original", "")
                if static_sentence:
                    self.template_order.append({
                        "type": "static",
                        "index": len(self.static_sentences),
                        "variations": variations,
                        "lexical_sets": lexical_sets,
                        "sentence": static_sentence
                    })
                    self.static_sentences.append(static_sentence)
                continue
            
            primary_fields = template.get("primary_data_fields", [])
            foreign_fields = template.get("foreign_data_fields", [])
            all_template_fields = primary_fields + foreign_fields
            
            field_type = template.get("field_type", "STANDARD")

            has_variations = bool(variations)
            has_counter = bool(counter_variations)
            has_null = bool(null_variations)

            if has_variations and has_counter and has_null:
                variation_class = "NULLABLE_BINARY"
            elif has_variations and has_counter and not has_null:
                counter_text = " ".join(counter_variations).lower()
                if "null" in counter_text:
                    variation_class = "NULL"
                else:
                    variation_class = "BINARY"
            else:
                variation_class = "STANDARD"
            
            primary_field = primary_fields[0] if primary_fields else (foreign_fields[0] if foreign_fields else None)
            
            if primary_field:
                self.template_order.append({
                    "type": "dynamic",
                    "field": primary_field,
                    "all_fields": all_template_fields,
                    "variations": variations,
                    "counter_variations": counter_variations,
                    "null_variations": null_variations,
                    "variation_class": variation_class,
                    "lexical_sets": lexical_sets
                })
            
            for field_name in all_template_fields:
                if field_name not in self.field_variations:
                    self.field_variations[field_name] = {
                        "variations": [],
                        "counter_variations": [],
                        "null_variations": [],
                        "variation_class": variation_class,
                        "lexical_sets": {}
                    }
                    self.all_field_names.add(field_name)
                    self.field_names_ordered.append(field_name)
                
                if field_name not in self.field_types:
                    self.field_types[field_name] = self._detect_field_type(field_name, field_type)
                
                self.field_variations[field_name]["variations"].extend(variations)
                
                if counter_variations:
                    self.field_variations[field_name]["counter_variations"].extend(counter_variations)
                
                if null_variations:
                    self.field_variations[field_name]["null_variations"].extend(null_variations)
                
                for word, synonyms in lexical_sets.items():
                    if word not in self.field_variations[field_name]["lexical_sets"]:
                        self.field_variations[field_name]["lexical_sets"][word] = synonyms
                    else:
                        existing_synonyms = set(self.field_variations[field_name]["lexical_sets"][word])
                        new_synonyms = set(synonyms)
                        self.field_variations[field_name]["lexical_sets"][word] = list(existing_synonyms.union(new_synonyms))
        
        for field_name in self.field_variations:
            self.field_variations[field_name]["variations"] = list(set(self.field_variations[field_name]["variations"]))
            self.field_variations[field_name]["counter_variations"] = list(set(self.field_variations[field_name]["counter_variations"]))
            self.field_variations[field_name]["null_variations"] = list(set(self.field_variations[field_name]["null_variations"]))
        
        for tbd_column in self.tbd_columns:
            if tbd_column not in self.field_names_ordered:
                self.field_names_ordered.append(tbd_column)
                self.all_field_names.add(tbd_column)
        
        print(f"Loaded templates for {len(self.all_field_names)} fields and {len(self.static_sentences)} static sentences")
        print(f"Total template order: {len(self.template_order)} elements")
        if self.tbd_columns:
            print(f"TBD columns (complex values will be generated at runtime): {self.tbd_columns}")
        return True
    
    def _detect_field_type(self, field_name: str, template_type: str = "STANDARD") -> str:
        """Detect field type based on field name patterns and template metadata."""
        field_name_lower = field_name.lower()
        
        binary_patterns = [
            '(y/n)', '(t/f)', '(yes/no)', '(true/false)',
            '_flag', 'is_', 'has_', 'can_', 'should_',
            'active', 'enabled', 'disabled', 'valid',
            'charter', 'magnet', 'virtual', 'status'
        ]
        
        for pattern in binary_patterns:
            if pattern in field_name_lower:
                return "BINARY"
        
        if template_type in ["BINARY", "NULL", "MISC"]:
            return template_type
        
        return "STANDARD"
    
    def _is_binary_value(self, value: str) -> Tuple[bool, bool]:
        """
        Check if a value is binary and determine its boolean state.
        Returns (is_binary, is_true_value).
        """
        value_str = str(value).strip().upper()
        
        true_values = {'1', 'TRUE', 'YES', 'Y', 'T', 'ACTIVE', 'ENABLED', 'ON'}
        false_values = {'0', 'FALSE', 'NO', 'N', 'F', 'INACTIVE', 'DISABLED', 'OFF'}
        
        if value_str in true_values:
            return True, True
        elif value_str in false_values:
            return True, False
        
        return False, False
    
    def _is_null_value(self, value: str) -> bool:
        value_str = str(value).strip().upper()
        null_values = {'NULL', 'NONE', 'NA', 'N/A', '', 'NAN', 'NIL', 'MISSING'}
        return value_str in null_values

    def _resolve_variation_pool(
        self,
        variation_class: str,
        variations: List[str],
        counter_variations: List[str],
        null_variations: List[str],
        is_null: bool,
        is_binary: bool,
        is_true: bool,
        field_name: str,
    ) -> List[str]:
        """Pick the right list of sentence variations for a given value.

        NULLABLE_BINARY (all 3 lists present):
            null value        → null_variations
            binary false/off  → counter_variations
            binary true/on    → variations
            non-binary value  → variations

        NULL (variations + counter_variations where counter mentions "null"):
            null value  → counter_variations
            other       → variations

        BINARY (variations + counter_variations, no null connotation):
            binary false/off → counter_variations
            binary true/on   → variations
            other            → variations

        STANDARD / static:
            always → variations
        """
        if variation_class == "NULLABLE_BINARY":
            if is_null and null_variations:
                return null_variations
            if is_binary and not is_true and counter_variations:
                return counter_variations
            return variations if variations else []

        if variation_class == "NULL":
            if is_null and counter_variations:
                return counter_variations
            return variations if variations else []

        if variation_class == "BINARY":
            if is_binary and not is_true and counter_variations:
                return counter_variations
            return variations if variations else []

        if variations:
            return variations
        fv = self.field_variations.get(field_name, {})
        return fv.get("variations", [])

    def get_sample_data_from_database(self, db_name: str, table_name: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        try:
            sqlite_file = os.path.join(self.minidev_path, db_name, f"{db_name}.sqlite")
            
            if not os.path.exists(sqlite_file):
                raise FileNotFoundError(f"Database file not found: {sqlite_file}")
            
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()
            
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            columns = [row[1] for row in cursor.fetchall()]
            
            query = f"SELECT * FROM [{table_name}] ORDER BY ROWID"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
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
    
    def format_data_value(self, field_name: str, field_value: str) -> str:
        field_value_str = str(field_value)
        field_name_lower = field_name.lower()
        
        if 'percent' in field_name_lower or '%' in field_name_lower:
            try:
                numeric_value = float(field_value_str)
                if 0 <= numeric_value <= 1:
                    percentage = numeric_value * 100
                    return f"{percentage:.1f}%"
                elif numeric_value > 1:
                    return f"{numeric_value:.1f}%"
            except ValueError:
                pass
        
        try:
            numeric_value = float(field_value_str)
            if numeric_value == int(numeric_value):
                return str(int(numeric_value))
            else:
                return f"{numeric_value:.2f}"
        except ValueError:
            pass
        
        return field_value_str
    
    def replace_placeholders(self, sentence: str, row_data: Dict[str, str]) -> str:
        placeholder_pattern = r'\[([^\]]+)\]'
        placeholders_in_sentence = re.findall(placeholder_pattern, sentence)
        
        for placeholder_text in placeholders_in_sentence:
            actual_field_name = None
            for data_field in row_data.keys():
                if data_field.upper() == placeholder_text.upper():
                    actual_field_name = data_field
                    break
            
            if actual_field_name and actual_field_name in row_data:
                formatted_value = self.format_data_value(actual_field_name, row_data[actual_field_name])
                pattern = r'\[' + re.escape(placeholder_text) + r'\]'
                sentence = re.sub(pattern, formatted_value, sentence)
        
        return sentence
    
    def generate_unique_lexical_combination(self, lexical_sets: Dict[str, List[str]]) -> Dict[str, str]:
        max_attempts = 100
        
        for attempt in range(max_attempts):
            lexical_combination = {}
            for word, synonyms in lexical_sets.items():
                if len(synonyms) > 1:
                    lexical_combination[word] = random.choice(synonyms)
                else:
                    lexical_combination[word] = word
            
            combination_signature = tuple(sorted(lexical_combination.items()))
            
            if combination_signature not in self.used_lexical_combinations:
                self.used_lexical_combinations.add(combination_signature)
                return lexical_combination
        
        lexical_combination = {}
        for word, synonyms in lexical_sets.items():
            lexical_combination[word] = random.choice(synonyms) if len(synonyms) > 1 else word
        return lexical_combination
    
    def apply_unique_lexical_variations(self, sentence: str, lexical_sets: Dict[str, List[str]], row_data: Dict[str, str]) -> str:
        all_data_values = set()
        for field_value in row_data.values():
            all_data_values.add(str(field_value).lower())

        sentence_lower = sentence.lower()

        applicable_sets = {}
        for word, synonyms in lexical_sets.items():
            if re.search(r'\b' + re.escape(word) + r'\b', sentence_lower):
                is_data_value = any(word.lower() in dv for dv in all_data_values)
                if not is_data_value:
                    applicable_sets[word] = synonyms

        if not applicable_sets:
            return sentence

        lexical_combination = self.generate_unique_lexical_combination(applicable_sets)
        
        for word, replacement in lexical_combination.items():
            if replacement != word:
                pattern = r'\b' + re.escape(word) + r'\b'
                
                def make_replacer(repl):
                    def replacement_func(match):
                        original_word = match.group()
                        if original_word.isupper():
                            return repl.upper()
                        elif original_word.istitle():
                            return repl.capitalize()
                        elif original_word.islower():
                            return repl.lower()
                        return repl
                    return replacement_func
                
                sentence = re.sub(pattern, make_replacer(replacement), sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def clean_sentence(self, sentence: str) -> str:
        sentence = strip_hashes_from_text(sentence)
        sentence = re.sub(r'\brow value for\b', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\bindicates that\b', 'shows', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\bthe value\b', 'the figure', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\brecorded as\b', 'showing', sentence, flags=re.IGNORECASE)
        
        sentence = re.sub(r'\b(\w+)\s+Name\s+of\b', '', sentence, flags=re.IGNORECASE)
        
        sentence = re.sub(r"\bPercent \(\%\) \w+ [^']*'s?\b", 'percentage', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b\w+ Count \([^)]+\)\b", 'count', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\bEnrollment \([^)]+\)\b", 'total enrollment', sentence, flags=re.IGNORECASE)
        
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()
        
        if sentence and not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        elif sentence.endswith('..'):
            sentence = sentence[:-1]
        
        return sentence
    
    def generate_complex_value_sentence(self, column_name: str, column_value: str, context: str = "") -> str:
        closing_phrases = [
            "Lastly,", "On a final note,", "To conclude,", "Finally,",
            "In closing,", "As a concluding remark,", "To wrap up,"
        ]
        
        selected_phrase = random.choice(closing_phrases)
        
        prompt = f"""Generate a closing sentence that describes the components of a complex data structure.

Column name: {column_name}
Complex value: {column_value}
Context: {context if context else "This is part of a data narrative document."}

CRITICAL REQUIREMENTS:
1. Start the sentence with "{selected_phrase}"
2. Include as many key-value pairs from the complex structure as are visible
3. The sentence must read naturally as a closing statement
4. If the data appears truncated or incomplete, still generate a sentence describing the AVAILABLE data
5. YOUR OUTPUT MUST BE ONLY THE SENTENCE — no explanations, no questions, no commentary
6. NEVER ask for more data or say you cannot generate — always output a sentence with what is provided
7. Return only the sentence with no markdown, no quotes, no explanations

FEW-SHOT EXAMPLES:

Example 1 (complete data):
Column name: purchaseUrls
Complex value: {{'cardKingdom': 'https://mtgjson.com/links/9fb51af0ad6f0736', 'cardmarket': 'https://mtgjson.com/links/ace8861194ee0b6a', 'tcgplayer': 'https://mtgjson.com/links/4843cea124a0d515'}}
Good Response: On a final note, the purchase URLs for this card are available through cardKingdom at https://mtgjson.com/links/9fb51af0ad6f0736, cardmarket at https://mtgjson.com/links/ace8861194ee0b6a, and tcgplayer at https://mtgjson.com/links/4843cea124a0d515.

Example 2 (truncated data):
Column name: booster
Complex value: {{'default': {{'boosters': [{{'contents': {{'basic': 1, 'common': 10, 'rare': 1, 'uncommon': 3}}, 'weight': 1913922}}, {{'contents': {{'basic': 1, 'common': 9, 'foilCommon': 1
Good Response: Lastly, the booster configuration includes a default setup with booster packs, the first containing 1 basic, 10 common, 1 rare, and 3 uncommon cards with a weight of 1913922, and the second featuring 1 basic, 9 common, and 1 foilCommon among other cards.

Example 3 (complete data):
Column name: legalities
Complex value: {{'commander': 'legal', 'duel': 'legal', 'legacy': 'banned', 'modern': 'legal', 'vintage': 'restricted'}}
Good Response: Finally, regarding format legalities, this card is legal in commander, duel, and modern formats, banned in legacy, and restricted in vintage.

Your response (ONLY the closing sentence, nothing else):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a sentence generator that creates closing sentences describing complex data structures. You MUST output ONLY a sentence — never ask questions, never say you cannot complete the task, never request more data. If data is truncated or incomplete, generate a sentence describing whatever data IS available. Your entire response must be a single natural language sentence starting with the provided closing phrase."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            
            if response and response.choices:
                message = response.choices[0].message
                content = message.content
                
                if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                    content = message.reasoning_content
                
                if content:
                    result = content.strip()
                    result = result.strip('"\'`')
                    result = result.replace('**', '')
                    result = result.replace('*', '')
                    
                    if not result.endswith(('.', '!', '?')):
                        result += '.'
                    
                    return result
            
            return f"{selected_phrase} the {column_name} contains: {column_value}."
            
        except Exception as e:
            print(f"      Error generating complex value sentence: {e}")
            return f"{selected_phrase} the {column_name} contains: {column_value}."
    
    def select_unique_variations(self, data_rows: List[Dict[str, str]], db_name: str = "", table_name: str = "") -> Tuple[List[Dict[str, str]], List[str]]:
        multi_column_results = []
        text_documents = []
        
        for row_idx, row_data in enumerate(data_rows):
            print(f"  Processing row {row_idx + 1}/{len(data_rows)}...")
            
            selected_variations = {}
            used_sentences_this_row = set()
            document_sentences = []
            complex_value_sentences = []
            
            for tbd_column in self.tbd_columns:
                if tbd_column in row_data:
                    complex_value = row_data[tbd_column]
                    complex_value_str = str(complex_value)
                    
                    self.tbd_sentence_cache = self._load_tbd_cache()
                    
                    cache_key = (tbd_column, complex_value_str)
                    if cache_key in self.tbd_sentence_cache:
                        complex_sentence = self.tbd_sentence_cache[cache_key]
                        print(f"    Using cached sentence for complex column: {tbd_column}")
                    else:
                        context = f"This is data from the {table_name} table in the {db_name} database."
                        print(f"    Generating sentence for complex column: {tbd_column}")
                        complex_sentence = self.generate_complex_value_sentence(tbd_column, complex_value, context)
                        complex_sentence = self.clean_sentence(complex_sentence)
                        self.tbd_sentence_cache[cache_key] = complex_sentence
                        self._save_tbd_cache()
                    
                    complex_value_sentences.append(complex_sentence)
                    selected_variations[tbd_column] = complex_sentence
            
            for template_item in self.template_order:
                if template_item["type"] == "static":
                    field_name = "static"
                else:
                    field_name = template_item["field"]

                variations_list = template_item.get("variations", [])
                counter_variations_list = template_item.get("counter_variations", [])
                null_variations_list = template_item.get("null_variations", [])
                variation_class = template_item.get("variation_class", "STANDARD")
                lexical_sets = template_item.get("lexical_sets", {})
                
                if field_name not in row_data and field_name != "static":
                    continue
                
                if field_name == "static":
                    field_value = "static"
                else:
                    field_value = row_data.get(field_name, "NULL")
                
                if field_name != "static":
                    is_null = self._is_null_value(field_value)
                    is_binary, is_true = self._is_binary_value(field_value)
                else:
                    is_null = False
                    is_binary, is_true = False, False

                available_variations = self._resolve_variation_pool(
                    variation_class, variations_list, counter_variations_list,
                    null_variations_list, is_null, is_binary, is_true, field_name,
                )
                
                if not available_variations:
                    formatted_value = self.format_data_value(field_name, field_value)
                    base_sentence = f"The {field_name} value is {formatted_value}."
                else:
                    globally_unused = [v for v in available_variations if v not in self.used_sentences_global]
                    row_unused = [v for v in available_variations if v not in used_sentences_this_row]
                    
                    if globally_unused:
                        base_sentence = random.choice(globally_unused)
                    elif row_unused:
                        base_sentence = random.choice(row_unused)
                    else:
                        base_sentence = random.choice(available_variations)
                
                selected_sentence = self.replace_placeholders(base_sentence, row_data)
                
                if not lexical_sets and field_name in self.field_variations:
                    lexical_sets = self.field_variations[field_name].get("lexical_sets", {})
                
                if lexical_sets:
                    selected_sentence = self.apply_unique_lexical_variations(
                        selected_sentence, 
                        lexical_sets,
                        row_data
                    )
                
                selected_sentence = self.clean_sentence(selected_sentence)
                
                selected_variations[field_name] = selected_sentence
                used_sentences_this_row.add(selected_sentence)
                self.used_sentences_global.add(selected_sentence)
                document_sentences.append(selected_sentence)
            
            multi_column_results.append(selected_variations)
            
            document_text = self.create_document_text(document_sentences)
            
            if complex_value_sentences:
                complex_paragraph = ' '.join(complex_value_sentences)
                document_text = document_text.rstrip() + '\n\n' + complex_paragraph
            
            text_documents.append(document_text)
        
        return multi_column_results, text_documents
    
    def create_document_text(self, sentences: List[str]) -> str:
        paragraphs = []
        current_paragraph = []
        
        transitions = [
            "Furthermore,", "Additionally,", "Moreover,", "In addition,",
            "Meanwhile,", "Subsequently,", "Consequently,", "As a result,",
            "Building on this,", "In this context,", "Similarly,", "Notably,"
        ]
        
        for i, sentence in enumerate(sentences):
            if len(current_paragraph) > 0 and random.random() < 0.3:
                transition = random.choice(transitions)
                if not sentence.lower().startswith(('the ', 'this ', 'these ', 'that ')):
                    sentence = f"{transition} {sentence.lower()}"
            
            current_paragraph.append(sentence)
            
            paragraph_length = random.randint(2, 4)
            if len(current_paragraph) >= paragraph_length:
                paragraph_text = ' '.join(current_paragraph)
                paragraphs.append(paragraph_text)
                current_paragraph = []
        
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            paragraphs.append(paragraph_text)
        
        return '\n\n'.join(paragraphs)
    
    def write_multi_column_csv(self, results: List[Dict[str, str]], db_name: str, table_name: str):
        if not results:
            return
        
        output_file = os.path.join(self.multi_column_dir, f"{db_name}_{table_name}_multi_column.csv")
        
        fieldnames = self.field_names_ordered
        
        print(f"  Writing multi-column CSV: {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in results:
                complete_row = {}
                for field_name in fieldnames:
                    complete_row[field_name] = row.get(field_name, "")
                writer.writerow(complete_row)
        
    def write_single_column_csv(self, results: List[Dict[str, str]], db_name: str, table_name: str):
        if not results:
            return
        
        output_file = os.path.join(self.single_column_dir, f"{db_name}_{table_name}_single_column.csv")
        
        print(f"  Writing single-column CSV: {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['row_id', 'field_name', 'sentence'])
            
            for row_idx, row in enumerate(results, 1):
                for field_name in self.field_names_ordered:
                    if field_name in row:
                        writer.writerow([row_idx, field_name, row[field_name]])
    
    def write_text_documents(self, documents: List[str], db_name: str, table_name: str):
        if not documents:
            return
        
        db_text_dir = os.path.join(self.text_dir, db_name)
        table_text_dir = os.path.join(db_text_dir, table_name)
        Path(table_text_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"  Writing text documents to: {table_text_dir}")
        
        for row_idx, document in enumerate(documents, 1):
            filename = f"{table_name}{row_idx}.txt"
            filepath = os.path.join(table_text_dir, filename)
            
            cleaned_document = strip_hashes_from_text(document)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_document)
    
    def get_table_row_count(self, db_name: str, table_name: str) -> int:
        try:
            sqlite_file = os.path.join(self.minidev_path, db_name, f"{db_name}.sqlite")
            
            if not os.path.exists(sqlite_file):
                return 0
            
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()
            
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            row_count = cursor.fetchone()[0]
            
            conn.close()
            return row_count
            
        except Exception as e:
            print(f"Error fetching row count: {e}")
            return 0
    
    def process_table(self, db_name: str, table_name: str, document_limit: Optional[int] = None):
        print(f"\nProcessing table: {db_name}.{table_name}")
        
        if not self.load_sentence_templates(db_name, table_name):
            print(f"  Skipping - no templates found")
            return
        
        total_rows = self.get_table_row_count(db_name, table_name)
        print(f"  Total rows in database: {total_rows}")
        
        if document_limit:
            rows_to_process = min(document_limit, total_rows)
        else:
            rows_to_process = total_rows
        
        print(f"  Processing {rows_to_process} rows...")
        
        data_rows = self.get_sample_data_from_database(db_name, table_name, limit=rows_to_process)
        
        if not data_rows:
            print(f"  No data retrieved from database")
            return
        
        self.used_sentences_global = set()
        self.used_lexical_combinations = set()
        
        multi_column_results, text_documents = self.select_unique_variations(data_rows, db_name, table_name)
        
        self.write_multi_column_csv(multi_column_results, db_name, table_name)
        self.write_single_column_csv(multi_column_results, db_name, table_name)
        self.write_text_documents(text_documents, db_name, table_name)
        
        print(f"  Completed: {len(multi_column_results)} documents generated")
    
    def process_tables(self, selected_tables: List[Tuple[str, str]], document_limit: Optional[int] = None):
        print("Starting document generation...")
        print(f"Processing {len(selected_tables)} tables")
        
        if document_limit:
            print(f"Document limit per table: {document_limit}")
        else:
            print("Generating all documents (full mode)")
        
        for db_name, table_name in selected_tables:
            self.process_table(db_name, table_name, document_limit)
        
        print("\nDocument generation complete")
        print(f"Output directories:")
        print(f"  Multi-column CSV: {self.multi_column_dir}")
        print(f"  Single-column CSV: {self.single_column_dir}")
        print(f"  Text documents: {self.text_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert database data to natural language documents')
    parser.add_argument('--limit', type=int, default=10,
                        help='Maximum documents per table (default: 10, use 0 for all)')
    parser.add_argument('--null-mode', type=str, default='implicit',
                        choices=['implicit', 'explicit'],
                        help='Null value handling mode (default: implicit)')
    parser.add_argument('--binary-mode', type=str, default='implicit',
                        choices=['implicit', 'explicit'],
                        help='Binary value handling mode (default: implicit)')
    
    args = parser.parse_args()
    
    converter = DataToTextConverter(
        null_mode=args.null_mode,
        binary_mode=args.binary_mode
    )
    
    random.seed(42)
    
    templates_dir = converter.templates_dir
    if not os.path.exists(templates_dir):
        print(f"Templates directory not found: {templates_dir}")
        print("Please run the template generation pipeline first!")
        return
    
    selected_tables = []
    for db_name in os.listdir(templates_dir):
        db_path = os.path.join(templates_dir, db_name)
        if os.path.isdir(db_path):
            for template_file in os.listdir(db_path):
                if template_file.endswith('_sentence_templates.json'):
                    table_name = template_file.replace('_sentence_templates.json', '')
                    selected_tables.append((db_name, table_name))
    
    if not selected_tables:
        print("No sentence template files found!")
        return
    
    document_limit = args.limit if args.limit > 0 else None
    converter.process_tables(selected_tables, document_limit=document_limit)


if __name__ == "__main__":
    main()
