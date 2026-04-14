"""Document assembly, template I/O, and standalone entry points."""

import json
import os
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

from .config import get_mode_folder_name as _cfg_get_mode_folder_name
from .config import get_noise_ratio_folder_name as _cfg_get_noise_ratio_folder_name
from .config import get_output_root
from .config import MISC_CHARACTERS
from .models import SentenceTemplate
from .text_utils import clean_field_string
from utils.cost_tracker import get_cost_tracker, track_openai_response


class DocumentAssemblyMixin:
    """Mixin providing document variation generation and template I/O.

    Expects the host class to provide:
        self.variation_generator (VariationBankGenerator)
        self.templates (list)
        self.null_mode, self.binary_mode
        self.document_context (str)
        self.hash_to_column, self.hash_to_replacement (dicts)
    """

    def save_templates(self, filename: str = "sentence_templates.json", tbd_columns: List[str] = None):
        data = {
            "document_context": self.document_context,
            "tbd_columns": tbd_columns if tbd_columns else [],
            "hash_to_replacement": getattr(self, 'hash_to_replacement', {}),
            "hash_to_column": getattr(self, 'hash_to_column', {}),
            "templates": [
                {
                    "original": t.original,
                    "template_pattern": t.template_pattern,
                    "primary_data_fields": t.primary_data_fields,
                    "foreign_data_fields": t.foreign_data_fields,
                    "variations": t.variations,
                    "counter_variations": t.counter_variations,
                    "null_variations": t.null_variations or [],
                    "lexical_sets": t.lexical_sets,
                    "field_data_types": t.field_data_types or {},
                    "is_static": t.is_static
                }
                for t in self.templates
            ]
        }

        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        json_str = json_str.replace('\\"[', '[').replace(']\\"', ']')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_str)

        print(f"Templates saved to {filename}")
        if tbd_columns:
            print(f"  TBD columns (complex values): {tbd_columns}")

    def load_templates(self, filename: str = "sentence_templates.json"):
        """Load templates from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.document_context = data.get("document_context", "professional data report")

            self.templates = []
            for template_data in data["templates"]:
                if "primary_data_fields" in template_data:
                    template = SentenceTemplate(
                        original=template_data["original"],
                        template_pattern=template_data["template_pattern"],
                        primary_data_fields=template_data["primary_data_fields"],
                        foreign_data_fields=template_data.get("foreign_data_fields", []),
                        variations=template_data["variations"],
                        counter_variations=template_data["counter_variations"],
                        lexical_sets=template_data["lexical_sets"],
                        field_data_types=template_data.get("field_data_types", {}),
                        is_static=template_data.get("is_static", False),
                        null_variations=template_data.get("null_variations", None)
                    )
                else:
                    template = SentenceTemplate(
                        original=template_data["original"],
                        template_pattern=template_data["template_pattern"],
                        primary_data_fields=template_data.get("primary_data_fields", []) if "primary_data_fields" in template_data else (template_data["data_fields"][0] if template_data.get("data_fields") else []),
                        foreign_data_fields=template_data.get("foreign_data_fields", []),
                        variations=template_data["variations"],
                        counter_variations=template_data.get("counter_variations", []),
                        lexical_sets=template_data["lexical_sets"],
                        field_data_types=template_data.get("field_data_types", {}),
                        is_static=False,
                        null_variations=template_data.get("null_variations", None)
                    )

                self.templates.append(template)

            print(f"Loaded {len(self.templates)} templates from {filename}")
            print(f"Document context: {self.document_context}")
            return True
        except FileNotFoundError:
            print(f"Template file {filename} not found")
            return False

    def format_data_value(self, field_name: str, field_value: str) -> str:
        """Format data values properly, especially percentage fields"""
        field_value_str = str(field_value)
        field_name_lower = field_name.lower()

        if 'percent' in field_name_lower or '%' in field_name_lower:
            try:
                numeric_value = float(field_value_str)
                if 0 <= numeric_value <= 1:
                    percentage = numeric_value * 100
                    return f"{percentage}%"
                elif numeric_value > 1:
                    return f"{numeric_value}%"
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

    def generate_document_variation(self, data_fields: Dict[str, str], consistency_seed: int = None) -> str:
        """Generate a document variation using templates"""
        if consistency_seed:
            random.seed(consistency_seed)

        consistency_mapping = {}
        generated_sentences = []

        used_template_indices = set()

        for field_name, field_value in data_fields.items():
            for template_idx, template in enumerate(self.templates):
                if field_name in template.primary_data_fields and template_idx not in used_template_indices:
                    field_value_str = str(field_value)
                    is_null = (field_value_str.upper() in ('NULL', 'NONE', '') or field_value_str.strip() == '')
                    is_zero = (field_value_str == '0')
                    is_one = (field_value_str == '1')

                    if template.null_variations:
                        if is_null and template.null_variations:
                            sentence = random.choice(template.null_variations)
                        elif is_zero and template.counter_variations:
                            sentence = random.choice(template.counter_variations)
                        elif template.variations:
                            sentence = random.choice(template.variations)
                        else:
                            sentence = template.original
                    elif (is_null or is_zero) and template.counter_variations:
                        sentence = random.choice(template.counter_variations)
                    elif not is_null and not is_zero and template.variations:
                        sentence = random.choice(template.variations)
                    elif template.is_static:
                        sentence = template.original
                    else:
                        if template.template_pattern:
                            sentence = template.template_pattern
                        elif template.variations:
                            sentence = random.choice(template.variations)
                        else:
                            sentence = template.original

                    generated_sentences.append(sentence)
                    used_template_indices.add(template_idx)
                    break

        all_placeholders = set()
        for template in self.templates:
            for field in template.primary_data_fields + template.foreign_data_fields:
                all_placeholders.add(f"[{field.upper()}]")

        for i, sentence in enumerate(generated_sentences):

            placeholder_pattern = r'\[([^\]]+)\]'
            placeholders_in_sentence = re.findall(placeholder_pattern, sentence)

            for placeholder_text in placeholders_in_sentence:
                actual_field_name = None
                for data_field in data_fields.keys():
                    if data_field.upper() == placeholder_text.upper():
                        actual_field_name = data_field
                        break

                if actual_field_name and actual_field_name in data_fields:
                    formatted_value = self.format_data_value(actual_field_name, data_fields[actual_field_name])

                    pattern = r'\[' + re.escape(placeholder_text) + r'\]'
                    sentence = re.sub(pattern, formatted_value, sentence)

                else:
                    pass

            generated_sentences[i] = sentence

        for i, sentence in enumerate(generated_sentences):
            current_template = None
            for template in self.templates:
                if (template.original == sentence or
                    (template.variations and sentence in template.variations) or
                    (template.counter_variations and sentence in template.counter_variations)):
                    current_template = template
                    break

            if current_template and current_template.lexical_sets:
                sentence = self._apply_lexical_variations(sentence, current_template, data_fields, consistency_mapping)
                generated_sentences[i] = sentence

        cleaned_sentences = []
        for sentence in generated_sentences:
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

            if len(sentence) > 15 and not sentence.endswith('..'):
                cleaned_sentences.append(sentence)

        document = ""

        paragraphs = []
        current_paragraph = []

        transitions = [
            "Furthermore,", "Additionally,", "Moreover,", "In addition,",
            "Meanwhile,", "Subsequently,", "Consequently,", "As a result,",
            "Building on this,", "In this context,", "Similarly,", "Notably,"
        ]

        for i, sentence in enumerate(cleaned_sentences):
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

        document += '\n\n'.join(paragraphs)

        return document

    def _apply_lexical_variations(self, sentence: str, template, data_fields: Dict[str, str], consistency_mapping: Dict[str, str]) -> str:
        """Apply lexical variations to a sentence based on template's lexical sets"""
        all_data_values = set()
        proper_nouns = set()

        for field_name, field_value in data_fields.items():
            field_value_str = str(field_value)
            all_data_values.add(field_value_str)

            words_in_value = field_value_str.split()
            for word_in_value in words_in_value:
                word_clean = re.sub(r'[^\w]', '', word_in_value)
                if word_clean and len(word_clean) > 1 and word_clean[0].isupper():
                    proper_nouns.add(word_clean)

        for word, synonyms in template.lexical_sets.items():
            if word in consistency_mapping:
                replacement = consistency_mapping[word]
            else:
                replacement = random.choice(synonyms)
                consistency_mapping[word] = replacement

            if replacement != word:
                pattern = r'\b' + re.escape(word) + r'\b'

                is_data_value = False
                for data_value in all_data_values:
                    if word.lower() in data_value.lower():
                        is_data_value = True
                        break

                is_proper_noun = False
                for proper_noun in proper_nouns:
                    if word.lower() == proper_noun.lower():
                        is_proper_noun = True
                        break

                field_name_patterns = [
                    r'\b' + re.escape(word) + r'\s+(Code|Name|Number|Type|Status|Count|Grade)\b',
                    r'\b(Academic|Charter|District|County|School|Free|FRPM|Enrollment|Percent)\s+' + re.escape(word) + r'\b'
                ]

                should_replace = True
                if is_data_value or is_proper_noun:
                    should_replace = False
                else:
                    for field_pattern in field_name_patterns:
                        if re.search(field_pattern, sentence, flags=re.IGNORECASE):
                            should_replace = False
                            break

                if should_replace:
                    def replacement_func(match, _replacement=replacement):
                        original_word = match.group()
                        if original_word.isupper():
                            return _replacement.upper()
                        elif original_word.istitle():
                            return _replacement.capitalize()
                        elif original_word.islower():
                            return _replacement.lower()
                        else:
                            return _replacement

                    sentence = re.sub(pattern, replacement_func, sentence)

        return sentence


def get_mode_folder_name(null_mode: str, binary_mode: str) -> str:
    """Re-exported for backward compatibility; canonical version lives in config.py."""
    return _cfg_get_mode_folder_name(null_mode, binary_mode)


def get_noise_ratio_folder_name(data_noise_x: int, data_noise_y: int) -> str:
    """Re-exported for backward compatibility; canonical version lives in config.py."""
    return _cfg_get_noise_ratio_folder_name(data_noise_x, data_noise_y)


def process_templates_standalone(base_dir: str, selected_tables: List[Tuple[str, str]],
                                   num_variations: int = 15, null_mode: str = "implicit",
                                   binary_mode: str = "implicit", skip_existing: bool = True,
                                   data_noise_x: int = 1, data_noise_y: int = 0,
                                   dataset_folder_name: str = "MINIDEV"):
    from .template_system import DocumentTemplateSystem

    cost_tracker = get_cost_tracker()

    mode_folder = _cfg_get_mode_folder_name(null_mode, binary_mode)
    ratio_folder = _cfg_get_noise_ratio_folder_name(data_noise_x, data_noise_y)

    out_root = get_output_root(base_dir)
    sentence_templates_dir = os.path.join(out_root, "templates", mode_folder, "sentence_templates")
    narrative_templates_dir = os.path.join(out_root, "templates", mode_folder, "narrative_templates", ratio_folder)

    output_dir = os.path.join(out_root, "variations", mode_folder, ratio_folder)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ratio_str = f"{data_noise_x}:{data_noise_y}"
    print(f"Processing templates with {num_variations} variations per sentence")
    print(f"Null mode: {null_mode}, Binary mode: {binary_mode}, Data:Noise: {ratio_str}")
    if skip_existing:
        print(f"Skipping tables with existing sentence templates")

    skipped_count = 0
    processed_count = 0

    for db_name, table_name in selected_tables:
        sentence_file = os.path.join(sentence_templates_dir, db_name, f"{table_name}_template.json")
        narrative_file = os.path.join(narrative_templates_dir, db_name, f"{table_name}_template.json")
        output_file = os.path.join(output_dir, db_name, f"{table_name}_sentence_templates.json")

        Path(os.path.join(output_dir, db_name)).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(sentence_file):
            print(f"Sentence template not found for {db_name}.{table_name}, skipping...")
            continue

        if skip_existing and os.path.exists(output_file):
            print(f"\nSkipping {db_name}.{table_name} (sentence templates already exist)")
            skipped_count += 1
            continue

        print(f"\nProcessing {db_name}.{table_name}...")

        try:
            with open(sentence_file, 'r', encoding='utf-8') as f:
                sentence_data = json.load(f)
        except Exception as e:
            print(f"Error loading sentence template: {e}")
            continue

        original_data = sentence_data.get('original_data', {})
        tbd_columns = sentence_data.get('tbd_columns', [])
        generated_sentences = sentence_data.get('generated_sentences', [])
        hash_to_column = sentence_data.get('hash_to_column', {})

        narrative = []
        hash_to_replacement = {}
        narrative_data_dict = None

        if data_noise_y > 0 and os.path.exists(narrative_file):
            try:
                with open(narrative_file, 'r', encoding='utf-8') as f:
                    narrative_data_dict = json.load(f)
                narrative = narrative_data_dict.get('narrative', [])
                hash_to_replacement = narrative_data_dict.get('hash_to_replacement', {})
            except Exception as e:
                print(f"  Warning: Could not load narrative template: {e}")

        if isinstance(narrative, list):
            narrative_text = '\n\n'.join(narrative)
        else:
            narrative_text = str(narrative) if narrative else ''

        template_system = DocumentTemplateSystem(
            null_mode=null_mode,
            binary_mode=binary_mode,
            dataset_folder_name=dataset_folder_name,
            docgen_base_dir=base_dir,
            num_variations=num_variations,
        )
        template_system.hash_to_replacement = hash_to_replacement
        template_system.hash_to_column = hash_to_column

        if data_noise_y == 0 or not narrative_text:
            sentences = generated_sentences if generated_sentences else []
            print(f"  Using generated sentences directly (no narrative)")
        elif tbd_columns:
            print(f"  TBD columns detected (complex values): {tbd_columns}")
            sentences = generated_sentences if generated_sentences else template_system.analyze_narrative_structure(narrative_text, original_data)
        else:
            sentences = template_system.analyze_narrative_structure(narrative_text, original_data)
        print(f"  Found {len(sentences)} sentences")

        template_system.create_sentence_templates(
            sentences, original_data, db_name, table_name,
            num_fields_to_check=1000,
            base_dir=base_dir,
            narrative_json_path=narrative_file if data_noise_y > 0 and os.path.exists(narrative_file) else None,
            sentence_template_data=sentence_data,
            narrative_template_data=narrative_data_dict,
            sentence_json_path=sentence_file,
        )

        print(f"  Created {len(template_system.templates)} sentence templates")

        template_system.save_templates(output_file, tbd_columns=tbd_columns)
        processed_count += 1

        print(f"  Saved to: {output_file}")

    cost_tracker.track_skipped_sentence_variations(skipped_count)

    print(f"\nSentence variation processing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (cached): {skipped_count}")


def main():
    """Main execution function"""
    from .template_system import DocumentTemplateSystem
    from .data_loader import get_table_row_count, get_sample_data_from_database

    print("Advanced Template-Based Document Generation System")
    print("=" * 60)

    template_system = DocumentTemplateSystem()

    db_name = ""
    table_name = ""
    data_fields = {}
    total_rows = 0
    try:
        with open('avoid_replacement.txt', 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                if ',' in line:
                    if line_num != 1:
                        key, value = line.strip().split(',', 1)
                        data_fields[key.strip()] = value.strip()
                    elif line_num == 1:
                        db_name, table_name = line.strip().split(',', 1)
    except FileNotFoundError:
        print("Error: avoid_replacement.txt not found.")
        return

    if db_name and table_name:
        try:
            total_rows = get_table_row_count(db_name, table_name)
            print(f"Processing data from {db_name}.{table_name} with {total_rows:,} total records")
            print()
        except Exception as e:
            print(f"Warning: Could not fetch row count from database: {e}")
            print("Continuing with template generation...")
            total_rows = 1
            print()
    else:
        total_rows = 1
        print("No database information found, generating 1 variation")

    if not template_system.load_templates():
        print("No existing templates found. Creating new templates...")

        try:
            with open('generated_narrative.txt', 'r', encoding='utf-8') as f:
                narrative = f.read()
        except FileNotFoundError:
            print("Error: generated_narrative.txt not found. Please run the original script first.")
            return

        sentences = template_system.analyze_narrative_structure(narrative, data_fields)
        print(f"Found {len(sentences)} sentences to process")

        template_system.create_sentence_templates(sentences, data_fields, db_name, table_name, 1000)

        template_system.save_templates()

    print("\nGenerating document variations from database...")

    if db_name and table_name:
        try:
            output_folder = f"{db_name}_{table_name}"
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created output folder: {output_folder}")

            sample_data = get_sample_data_from_database(db_name, table_name, limit=total_rows)

            for i, row_data in enumerate(sample_data):
                print(f"\nGenerating variation {i+1} for row data...")
                start_time = time.time()

                variation = template_system.generate_document_variation(row_data, consistency_seed=i)

                generation_time = time.time() - start_time
                print(f"Generation time: {generation_time:.3f} seconds")

                filename = f"{table_name}_row_{i+1}.txt"
                filepath = os.path.join(output_folder, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(variation)

                print(f"Saved to {filepath}")

        except Exception as e:
            print(f"Error processing database: {e}")
            print("Falling back to sample data...")

            output_folder = "fallback_output"
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created fallback output folder: {output_folder}")

            for i in range(1):
                print(f"\nGenerating variation {i+1}...")
                start_time = time.time()

                variation = template_system.generate_document_variation(data_fields, consistency_seed=i)

                generation_time = time.time() - start_time
                print(f"Generation time: {generation_time:.3f} seconds")

                filename = f"fallback_row_{i+1}.txt"
                filepath = os.path.join(output_folder, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(variation)

                print(f"Saved to {filepath}")
    else:
        print("No database information, using sample data...")

        output_folder = "sample_data_output"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created sample data output folder: {output_folder}")

        for i in range(1):
            print(f"\nGenerating variation {i+1}...")
            start_time = time.time()

            variation = template_system.generate_document_variation(data_fields, consistency_seed=i)

            generation_time = time.time() - start_time
            print(f"Generation time: {generation_time:.3f} seconds")

            filename = f"sample_row_{i+1}.txt"
            filepath = os.path.join(output_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(variation)

            print(f"Saved to {filepath}")
