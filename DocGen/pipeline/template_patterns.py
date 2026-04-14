"""Template pattern building and conflict resolution methods."""

import json
import os
import re
import string
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

from .text_utils import clean_field_string, is_misc_value, count_placeholders
from .config import MISC_CHARACTERS, table_sample_entry_path


class TemplatePatternMixin:
    """Mixin providing template pattern building and conflict resolution.

    Expects the host class to provide:
        self.variation_generator
        self.local_llm_client
        self.null_mode, self.binary_mode
        self.templates (list)
        self.check_field_in_sentence()
        self.check_name_in_sentence()
        self.query_local_llm_for_field_verification()
        self.query_local_llm_for_field_detection()
    """

    def validate_variation_placeholders(self, variation: str, expected_count: int) -> bool:
        actual_count = count_placeholders(variation)
        return actual_count == expected_count

    def validate_counter_variation_placeholders(self, variation: str, expected_count: int, primary_field_name: str) -> bool:
        actual_count = count_placeholders(variation)
        if actual_count != expected_count:
            return False
        primary_placeholder = f"[{primary_field_name.upper()}]"
        if primary_placeholder in variation:
            return False
        return True

    def regenerate_variation_with_validation(self, sentence: str, field_name: str, context: str, expected_placeholders: int, is_counter: bool = False, primary_field_name: str = None) -> str:
        max_attempts = 2
        null_replacement_phrase = "NULL" if self.null_mode == "explicit" else "not specified"
        for attempt in range(max_attempts):
            if is_counter:
                variations = self.variation_generator.generate_null_variations_null(sentence, field_name, null_replacement_phrase, context)
            else:
                variations = self.variation_generator.generate_structural_variations_standard(sentence, field_name, context)

            if variations:
                variation = variations[0]
                if is_counter:
                    if self.validate_counter_variation_placeholders(variation, expected_placeholders, primary_field_name or field_name):
                        return variation
                else:
                    if self.validate_variation_placeholders(variation, expected_placeholders):
                        return variation
        return sentence

    def clean_malformed_variations(self, variations: List[str]) -> List[str]:
        """Clean malformed variations that contain incorrect placeholder patterns.

        Fixes patterns like:
        - "The Percent Eligible Free K-12, which is [PLACEHOLDER]% for this school"
        - "not specified%"
        - Other malformed percentage and placeholder patterns

        Args:
            variations (List[str]): List of variation strings to clean

        Returns:
            List[str]: Cleaned variation strings
        """
        cleaned_variations = []

        for variation in variations:
            cleaned = variation

            cleaned = re.sub(
                r'([A-Za-z\s]+),\s+which\s+is\s+\[PLACEHOLDER\]%',
                r'\1 is [PLACEHOLDER]%',
                cleaned
            )

            cleaned = re.sub(
                r'\bnot\s+specified%',
                'not specified',
                cleaned,
                flags=re.IGNORECASE
            )

            cleaned = re.sub(
                r'([A-Za-z\s]+),\s+\[PLACEHOLDER\]%',
                r'\1 is [PLACEHOLDER]%',
                cleaned
            )

            cleaned = re.sub(
                r'([A-Za-z\s]+)\s+which\s+is\s+\[PLACEHOLDER\]%',
                r'\1 is [PLACEHOLDER]%',
                cleaned
            )

            cleaned = re.sub(
                r'([A-Za-z\s]+),\s+(\[PLACEHOLDER\]%)',
                r'\1 is \2',
                cleaned
            )

            cleaned = re.sub(
                r'\bwhich\s+is\s+(\[PLACEHOLDER\]%)',
                r'is \1',
                cleaned
            )

            cleaned = re.sub(r'\s+', ' ', cleaned)

            cleaned = re.sub(r'\s*,\s*$', '', cleaned)
            cleaned = re.sub(r'\s*,\s*\.', '.', cleaned)

            cleaned_variations.append(cleaned.strip())

        return cleaned_variations

    def fetch_dummy_value(self, field_name: str, database: str, table: str, num_fields_to_check: int = 1000) -> str:
        """Fetch the first non-null value for a field from sample data files."""
        print(f"    Fetching dummy value for field: {field_name}")

        bd = getattr(self, "_docgen_base_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ds = getattr(self, "dataset_folder_name", "MINIDEV")
        for i in range(num_fields_to_check):
            try:
                path = table_sample_entry_path(bd, ds, database, table, i)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    for line in content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            if key == field_name:
                                value = value.strip()
                                if value.lower() not in ["null", "none", ""]:
                                    print(f"    Found dummy value: {value}")
                                    return value
            except Exception as e:
                print(f"    Ran out of sample data files")
                break

        print(f"    No non-null value found for {field_name}, using default")
        return f"Sample_{field_name.replace(' ', '_')}"

    @staticmethod
    def _value_present_in_sentence(sentence: str, replacement_value: str) -> bool:
        """Return True only if *replacement_value* actually appears in *sentence*."""
        if not replacement_value:
            return False
        return replacement_value.lower() in sentence.lower()

    @staticmethod
    def _value_present_in_sentence_strict(sentence: str, value: str) -> bool:
        """Strict standalone-word check for static sentence guardrails.

        The value must appear as a standalone token separated by whitespace or
        sentence-level punctuation (period, comma, semicolon, colon, etc.) on
        both sides.  Single/two-character values like 'D' or 'No' are matched
        only when they stand completely alone as a word -- never as a substring
        inside another word.
        """
        if not value:
            return False
        value_stripped = value.strip()
        if not value_stripped:
            return False
        boundary = r'(?:(?<=^)|(?<=[\s,.:;!?()"\'\-]))'
        boundary_after = r'(?=[\s,.:;!?()"\'\-]|$)'
        pattern = boundary + re.escape(value_stripped) + boundary_after
        return bool(re.search(pattern, sentence, re.IGNORECASE))

    def resolve_hashed_field(self, sentence: str, sentence_hash: str, data_fields: Dict[str, str], field_metadata: Dict[str, str]) -> Dict[str, Dict]:
        """Resolve the single expected field for a hash-tagged sentence.

        Uses hash_to_replacement (exact surface string from narrative) when
        available; otherwise falls back to a single-column check_field_in_sentence
        for the column identified by hash_to_column.

        A detection is only accepted when the replacement_value can actually be
        located inside the sentence.  If check_field_in_sentence finds the field
        *name* but not the *value*, the match is rejected and the sentence is
        flagged for regeneration.
        """
        hash_to_replacement = getattr(self, 'hash_to_replacement', {}) or {}
        hash_to_column = getattr(self, 'hash_to_column', {}) or {}

        column = hash_to_column.get(sentence_hash)
        if not column or column not in data_fields:
            print(f"    [HASH-RESOLVE] WARNING: hash {sentence_hash} has no column mapping or column not in data_fields — skipping")
            return {}

        if sentence_hash in hash_to_replacement:
            replacement_value = hash_to_replacement[sentence_hash]
            print(f"    [HASH-RESOLVE] column='{column}' resolved via replacement map -> '{replacement_value[:60]}'")
            return {column: {
                'replacement_value': replacement_value,
                'matched_name': column,
                'field_type': field_metadata.get(column, 'STANDARD'),
                'field_value': data_fields[column],
                'match_type': 'hash_validated'
            }}

        print(f"    [HASH-RESOLVE] column='{column}' not in replacement map — running single-column detection")
        found, replacement_value, matched_name, match_type = self.check_field_in_sentence(
            sentence, column, data_fields[column], field_metadata,
            type_override=None, use_llm_fallback=False, data_fields=data_fields
        )
        if found:
            if self._value_present_in_sentence(sentence, replacement_value):
                print(f"    [HASH-RESOLVE] single-column detection succeeded -> '{replacement_value[:60]}'")
                return {column: {
                    'replacement_value': replacement_value,
                    'matched_name': matched_name,
                    'field_type': field_metadata.get(column, 'STANDARD'),
                    'field_value': data_fields[column],
                    'match_type': match_type
                }}
            else:
                print(f"    [HASH-RESOLVE] field name matched but value '{replacement_value[:40]}' not present in sentence — rejecting")

        print(f"    [HASH-RESOLVE] fast detection failed — retrying with LLM fallback for '{column}'")
        found, replacement_value, matched_name, match_type = self.check_field_in_sentence(
            sentence, column, data_fields[column], field_metadata,
            type_override=None, use_llm_fallback=True, data_fields=data_fields
        )
        if found:
            if self._value_present_in_sentence(sentence, replacement_value):
                print(f"    [HASH-RESOLVE] LLM-assisted detection succeeded -> '{replacement_value[:60]}'")
                return {column: {
                    'replacement_value': replacement_value,
                    'matched_name': matched_name,
                    'field_type': field_metadata.get(column, 'STANDARD'),
                    'field_value': data_fields[column],
                    'match_type': match_type
                }}
            else:
                print(f"    [HASH-RESOLVE] LLM found field name but value '{replacement_value[:40]}' not present in sentence — rejecting")

        actual_value = str(data_fields[column])
        print(f"    [HASH-RESOLVE] all detection failed for column='{column}' — flagging for sentence regeneration")
        return {column: {
            'replacement_value': actual_value,
            'matched_name': column,
            'field_type': field_metadata.get(column, 'STANDARD'),
            'field_value': actual_value,
            'match_type': 'needs_regeneration'
        }}

    def regenerate_sentence_with_validation(
        self,
        column: str,
        data_fields: Dict[str, str],
        field_metadata: Dict[str, str],
        database: str,
        table: str,
        sentence_index: int,
        sentences: List[str],
        sentence_template_data: Dict[str, Any],
        narrative_template_data: Dict[str, Any] = None,
        narrative_json_path: str = None,
        sentence_json_path: str = None,
        base_dir: str = None,
        max_regenerations: int = 5,
    ) -> Tuple[str, str, Dict[str, Dict]]:
        """Regenerate a sentence when hash validation detects the value is missing.

        Uses the TemplateGenerator from generate_all_templates to produce a new
        sentence, assigns a fresh hash, and re-runs hash validation (Stages 1-3).
        If the value still cannot be parsed from the new sentence the cycle repeats
        up to *max_regenerations* times (default 5).

        Returns (new_sentence_with_hash, new_hash, detected_fields).
        On exhaustion returns the last generated sentence with an empty detected_fields dict.
        """
        from .template_generator import TemplateGenerator as TG

        if base_dir is None:
            base_dir = getattr(self, "_docgen_base_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        tg = TG(
            base_dir=base_dir,
            null_mode=self.null_mode,
            binary_mode=self.binary_mode,
            dataset_folder_name=getattr(self, "dataset_folder_name", "MINIDEV"),
        )
        descriptor = tg.get_column_descriptor(column, database, table)
        if not descriptor:
            descriptor = f"Column {column}"

        column_value = str(data_fields[column])
        hash_to_column = getattr(self, 'hash_to_column', {}) or {}
        old_hash = None
        old_sentence_with_hash = (sentences[sentence_index] or "").strip() if sentence_index < len(sentences) else ""
        if old_sentence_with_hash:
            old_hash = TG.extract_hash(old_sentence_with_hash)

        last_sentence_with_hash = old_sentence_with_hash
        last_hash = old_hash

        for attempt in range(1, max_regenerations + 1):
            print(f"    [REGEN] attempt {attempt}/{max_regenerations} for column='{column}' value='{column_value}'")
            new_sentence = tg.generate_sentence_for_column(column, column_value, descriptor, append_hash=True)
            new_hash = TG.extract_hash(new_sentence)
            new_sentence_bare = TG.strip_hash(new_sentence)
            print(f"    [REGEN] generated: {new_sentence_bare[:100]}...")

            if not new_hash:
                print(f"    [REGEN] generated sentence has no hash — retrying")
                continue

            found, replacement_value, matched_name, match_type = self.check_field_in_sentence(
                new_sentence_bare, column, column_value, field_metadata,
                type_override=None, use_llm_fallback=False, data_fields=data_fields
            )
            if found and not self._value_present_in_sentence(new_sentence_bare, replacement_value):
                print(f"    [REGEN] field name matched but value not in sentence — treating as not found")
                found = False
            if not found:
                found, replacement_value, matched_name, match_type = self.check_field_in_sentence(
                    new_sentence_bare, column, column_value, field_metadata,
                    type_override=None, use_llm_fallback=True, data_fields=data_fields
                )
                if found and not self._value_present_in_sentence(new_sentence_bare, replacement_value):
                    print(f"    [REGEN] LLM found field name but value not in sentence — treating as not found")
                    found = False

            if not found:
                print(f"    [REGEN] value not parseable from regenerated sentence — will retry")
                last_sentence_with_hash = new_sentence
                last_hash = new_hash
                continue

            print(f"    [REGEN] validation passed (match_type={match_type}) — updating templates")

            if old_hash and old_hash in hash_to_column:
                del hash_to_column[old_hash]
            hash_to_column[new_hash] = column
            self.hash_to_column = hash_to_column

            if sentence_index < len(sentences):
                sentences[sentence_index] = new_sentence

            gen_sentences = sentence_template_data.get('generated_sentences', [])
            if sentence_index < len(gen_sentences):
                gen_sentences[sentence_index] = new_sentence
            sentence_template_data['generated_sentences'] = gen_sentences
            sentence_template_data['hash_to_column'] = hash_to_column

            if sentence_json_path:
                try:
                    with open(sentence_json_path, 'w', encoding='utf-8') as f:
                        json.dump(sentence_template_data, f, indent=2, ensure_ascii=False)
                    print(f"    [REGEN] sentence template saved to {sentence_json_path}")
                except Exception as e:
                    print(f"    [REGEN] WARNING: could not save sentence template: {e}")

            if narrative_template_data is not None and narrative_json_path:
                narrative_list = narrative_template_data.get('narrative', [])
                if isinstance(narrative_list, list) and old_hash:
                    old_hash_tag = f"(Hash: {old_hash})"
                    new_hash_tag = f"(Hash: {new_hash})"
                    updated_narrative = []
                    for para in narrative_list:
                        if old_hash_tag in para:
                            old_sentence_bare = TG.strip_hash(old_sentence_with_hash) if old_sentence_with_hash else ""
                            if old_sentence_bare and old_sentence_bare in para:
                                para = para.replace(old_sentence_bare + f" {old_hash_tag}", new_sentence_bare + f" {new_hash_tag}")
                            else:
                                para = para.replace(old_hash_tag, new_hash_tag)
                                segments = re.findall(r'\|\s*([^|]*?' + re.escape(new_hash_tag) + r')\s*\|', para)
                                if segments:
                                    old_segment = segments[0]
                                    new_segment = f"{new_sentence_bare} {new_hash_tag}"
                                    para = para.replace(f"| {old_segment} |", f"| {new_segment} |")
                        updated_narrative.append(para)
                    narrative_template_data['narrative'] = updated_narrative
                    try:
                        with open(narrative_json_path, 'w', encoding='utf-8') as f:
                            json.dump(narrative_template_data, f, indent=2, ensure_ascii=False)
                        print(f"    [REGEN] narrative template saved to {narrative_json_path}")
                    except Exception as e:
                        print(f"    [REGEN] WARNING: could not save narrative template: {e}")

            detected_fields = {column: {
                'replacement_value': replacement_value,
                'matched_name': matched_name,
                'field_type': field_metadata.get(column, 'STANDARD'),
                'field_value': column_value,
                'match_type': match_type
            }}
            return new_sentence_bare, new_hash, detected_fields

        print(f"    [REGEN] exhausted {max_regenerations} attempts for column='{column}' — no valid sentence produced")
        return TG.strip_hash(last_sentence_with_hash), last_hash, {}

    def parse_all_fields_in_sentence(self, sentence: str, data_fields: Dict[str, str], field_metadata: Dict[str, str], use_llm_fallback: bool = False, hash_to_replacement: Dict[str, str] = None, hash_to_column: Dict[str, str] = None) -> Dict[str, Dict]:
        """Legacy full-scan parser used only by main() / generated_narrative paths that lack hashes."""
        detected_fields = {}

        for field_name, field_value in data_fields.items():
            if field_name in detected_fields:
                continue
            found, replacement_value, matched_name, match_type = self.check_field_in_sentence(
                sentence, field_name, field_value, field_metadata, type_override=None, use_llm_fallback=use_llm_fallback, data_fields=data_fields
            )

            if found:
                detected_fields[field_name] = {
                    'replacement_value': replacement_value,
                    'matched_name': matched_name,
                    'field_type': field_metadata.get(field_name, 'STANDARD'),
                    'field_value': field_value,
                    'match_type': match_type
                }

        return detected_fields

    def assign_field_roles(self, detected_fields: Dict[str, Dict], primary_keys_used: set) -> Tuple[List[str], List[str]]:
        """Assign primary and foreign roles to detected fields based on match strength."""

        match_strength_order = {
            'hash_validated': 6,
            'field_name_direct': 4,
            'field_name_word_by_word': 3,
            'field_name_llm': 2,
            'field_value': 1,
            '': 0
        }

        def get_match_strength(field_name):
            match_type = detected_fields[field_name].get('match_type', '')
            return match_strength_order.get(match_type, 0)

        unused_fields = [f for f in detected_fields.keys() if f not in primary_keys_used]
        used_fields = [f for f in detected_fields.keys() if f in primary_keys_used]

        primary_fields = []

        if unused_fields:
            unused_fields_sorted = sorted(unused_fields, key=lambda f: (get_match_strength(f), f), reverse=True)
            best_field = unused_fields_sorted[0]
            primary_fields.append(best_field)
            primary_keys_used.add(best_field)
            match_type = detected_fields[best_field].get('match_type', '')
            print(f"    Using UNUSED primary key: {best_field} (match_type: {match_type}, now marked as used)")
        elif used_fields:
            used_fields_sorted = sorted(used_fields, key=lambda f: (get_match_strength(f), f), reverse=True)
            best_field = used_fields_sorted[0]
            primary_fields.append(best_field)
            match_type = detected_fields[best_field].get('match_type', '')
            print(f"    Reusing PREVIOUSLY USED primary key: {best_field} (match_type: {match_type})")

        foreign_fields = [f for f in detected_fields.keys() if f not in primary_fields]

        if len(detected_fields) > 1:
            print(f"    Match strength ranking for all fields:")
            all_fields_sorted = sorted(detected_fields.keys(), key=lambda f: (get_match_strength(f), f), reverse=True)
            for field_name in all_fields_sorted:
                match_type = detected_fields[field_name].get('match_type', '')
                strength = get_match_strength(field_name)
                role = "PRIMARY" if field_name in primary_fields else "FOREIGN"
                print(f"      {field_name}: {match_type} (strength: {strength}) - {role}")

        return primary_fields, foreign_fields

    def safe_replace_value(self, text: str, field_name: str, value: str, placeholder: str) -> str:
        """Replace value with placeholder, skipping occurrences that are part of the field name phrase.

        E.g. field_name='CALPADS 1', value='1', sentence='The CALPADS 1 value is 1.'
        Only the standalone '1' (after 'is') is replaced, not the '1' inside the label.
        """
        if not value:
            return text
        natural_name = field_name.replace('_', ' ')
        name_tokens = natural_name.lower().split()
        val_lower = value.lower()

        punctuation_no_dash = string.punctuation.replace("-", "")
        boundary_before = r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))'
        boundary_after  = r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'
        value_pattern = re.compile(boundary_before + re.escape(value) + boundary_after, re.IGNORECASE)

        if len(name_tokens) > 1:
            name_phrase_pattern = re.compile(
                r'[\s' + re.escape(punctuation_no_dash) + r']*'.join(re.escape(t) for t in name_tokens),
                re.IGNORECASE
            )
            name_spans = [(m.start(), m.end()) for m in name_phrase_pattern.finditer(text)]
        else:
            name_spans = []

        suffix_prefix_tokens = []
        if val_lower in name_tokens:
            for k in range(len(name_tokens)):
                suffix = name_tokens[k:]
                if ' '.join(suffix) == val_lower or (len(suffix) == 1 and suffix[0] == val_lower):
                    suffix_prefix_tokens = name_tokens[:k]
                    break

        def _is_inside_name_span(match_start, match_end):
            for ns, ne in name_spans:
                if match_start >= ns and match_end <= ne:
                    return True
            return False

        def _preceded_by_field_prefix(match_start, src):
            """Check if the tokens immediately before match_start are suffix_prefix_tokens."""
            if not suffix_prefix_tokens:
                return False
            before_text = src[:match_start].rstrip()
            for tok in reversed(suffix_prefix_tokens):
                pattern_tok = re.compile(re.escape(tok) + r'[\s' + re.escape(punctuation_no_dash) + r']*$', re.IGNORECASE)
                m = pattern_tok.search(before_text)
                if not m:
                    return False
                before_text = before_text[:m.start()].rstrip()
            return True

        matches = list(value_pattern.finditer(text))
        for m in reversed(matches):
            if _is_inside_name_span(m.start(), m.end()):
                continue
            if _preceded_by_field_prefix(m.start(), text):
                continue
            text = text[:m.start()] + placeholder + text[m.end():]

        return text

    def try_safe_replace_values(self, text: str, field_name: str, field_value_str: str, replacement_value: str, placeholder: str) -> Tuple[str, bool, int]:
        """Try replacing value variants with placeholder using field-name-aware safe replacement.

        Tries replacement_value, then field_value_str (if different), then numeric
        alternate forms (int, zero-padded). Returns (result_text, success, count).
        """
        values_to_try = [replacement_value]
        if field_value_str != replacement_value:
            values_to_try.append(field_value_str)

        if field_value_str.replace('.', '', 1).isdigit():
            try:
                numeric_value = float(field_value_str)
                if numeric_value == int(numeric_value):
                    int_str = str(int(numeric_value))
                    if int_str not in values_to_try:
                        values_to_try.append(int_str)
                    if len(field_value_str) > len(int_str) and field_value_str.startswith('0'):
                        if field_value_str not in values_to_try:
                            values_to_try.append(field_value_str)
            except ValueError:
                pass

        for val in values_to_try:
            result = self.safe_replace_value(text, field_name, val, placeholder)
            if result != text:
                count = result.count(placeholder) - text.count(placeholder)
                return result, True, max(count, 1)

        return text, False, 0

    def build_template_pattern(self, sentence: str, detected_fields: Dict[str, Dict], primary_fields: List[str] = None, foreign_fields: List[str] = None) -> str:
        """Build template pattern with placeholders - primary keys replaced first."""
        template_pattern = sentence

        ordered_fields = []
        if primary_fields:
            for f in primary_fields:
                if f in detected_fields:
                    ordered_fields.append(f)
        if foreign_fields:
            for f in foreign_fields:
                if f in detected_fields and f not in ordered_fields:
                    ordered_fields.append(f)
        for f in detected_fields.keys():
            if f not in ordered_fields:
                ordered_fields.append(f)

        for field_name in ordered_fields:
            field_info = detected_fields[field_name]
            replacement_value = field_info['replacement_value']
            placeholder = f"[{field_name.upper()}]"

            template_pattern = self.safe_replace_value(template_pattern, field_name, replacement_value, placeholder)
            print(f"    Replaced field value '{replacement_value}' -> '[{field_name.upper()}]'")

        return template_pattern

    def remove_conflicting_fields(self, detected_fields: Dict[str, Dict], data_fields: Dict[str, str], primary_fields: List[str], foreign_fields: List[str], sentence: str = None, field_metadata: Dict[str, str] = None) -> Tuple[List[str], List[str]]:
        """Remove fields whose values are contained in other field names or values.
        Uses LLM verification to remove false positives before conflict resolution.
        """
        all_trusted = all(
            info.get('match_type') == 'hash_validated'
            for info in detected_fields.values()
        )
        if len(detected_fields) <= 1 or all_trusted:
            print(f"      [CONFLICT-CHECK] Short-circuit — {'single field' if len(detected_fields) <= 1 else 'all hash-trusted'}, no conflict resolution needed")
            return primary_fields, foreign_fields

        fields_to_remove_from_primary = []
        fields_to_remove_from_foreign = []

        verification_status = {}

        if sentence and field_metadata:
            print(f"      Step 1: LLM verification to remove false positive field detections...")
            verified_fields = {}
            for field_name, field_info in detected_fields.items():
                field_value = str(data_fields.get(field_name, ""))
                match_type = field_info.get('match_type', '')
                field_type = field_metadata.get(field_name, "STANDARD")

                is_binary_or_null = field_type in ["BINARY", "NULL", "NULLABLE_BINARY"]

                should_verify = (
                    match_type == 'field_value' or
                    match_type == 'field_name_word_by_word'
                )

                if should_verify:
                    print(f"        Verifying '{field_name}' (match_type: {match_type}, field_type: {field_type}) with LLM...")

                    other_field_names = [fn for fn in data_fields.keys() if fn != field_name]

                    if is_binary_or_null:
                        field_verified = self.query_local_llm_for_field_verification(
                            sentence, field_name, field_value, field_type, self.null_mode, self.binary_mode
                        )
                        verification_msg = "both name and value"
                    else:
                        field_verified = self.query_local_llm_for_field_detection(sentence, field_name, field_value, other_field_names)
                        verification_msg = "eligible for replacement"

                    if field_verified:
                        verified_fields[field_name] = field_info
                        verification_status[field_name] = 'llm_verified'
                        print(f"        ✓ LLM confirmed '{field_name}' ({verification_msg})")
                    else:
                        if field_name in primary_fields:
                            fields_to_remove_from_primary.append(field_name)
                        elif field_name in foreign_fields:
                            fields_to_remove_from_foreign.append(field_name)
                        print(f"        ✗ LLM confirmed '{field_name}' is NOT eligible for replacement - removing")
                else:
                    verified_fields[field_name] = field_info
                    verification_status[field_name] = 'pre_verified'
                    print(f"        ✓ '{field_name}' (match_type: {match_type}) - trusted without LLM verification")

            detected_fields = verified_fields

        print(f"      Step 2: Resolving conflicts between verified fields...")
        removed_fields = set(fields_to_remove_from_primary + fields_to_remove_from_foreign)

        for field_name, field_info in detected_fields.items():
            if field_name in removed_fields:
                continue

            field_value = str(data_fields[field_name])

            for other_field_name, other_field_info in detected_fields.items():
                if (field_name == other_field_name or
                    other_field_name in removed_fields or
                    field_name in removed_fields):
                    continue

                other_field_value = str(data_fields[other_field_name])

                if field_value in other_field_name.split():
                    if field_name in primary_fields:
                        if other_field_name in primary_fields:
                            fields_to_remove_from_primary.append(other_field_name)
                        elif other_field_name in foreign_fields:
                            fields_to_remove_from_foreign.append(other_field_name)
                        removed_fields.add(other_field_name)
                        print(f"      Conflict: Field VALUE '{field_value}' in field name '{other_field_name}'. Primary field '{field_name}' wins, removing '{other_field_name}'.")
                    else:
                        if field_name in foreign_fields:
                            fields_to_remove_from_foreign.append(field_name)
                        removed_fields.add(field_name)
                        print(f"      Conflict: Field VALUE '{field_value}' in field name '{other_field_name}'. Removing non-primary field '{field_name}'.")
                    continue

                if field_value == other_field_value or field_value in other_field_value:
                    if field_name in primary_fields:
                        if other_field_name in foreign_fields:
                            fields_to_remove_from_foreign.append(other_field_name)
                        elif other_field_name in primary_fields:
                            fields_to_remove_from_primary.append(other_field_name)
                        removed_fields.add(other_field_name)
                        print(f"      Conflict: Same value '{field_value}'. Primary field '{field_name}' wins, removing '{other_field_name}'.")
                        continue

                    if field_name in foreign_fields and other_field_name in foreign_fields:
                        field_verification = verification_status.get(field_name, 'llm_verified')
                        other_verification = verification_status.get(other_field_name, 'llm_verified')

                        if field_verification == 'pre_verified' and other_verification == 'llm_verified':
                            fields_to_remove_from_foreign.append(other_field_name)
                            removed_fields.add(other_field_name)
                            print(f"      Conflict: Same value '{field_value}'. '{field_name}' was pre-verified (trusted), '{other_field_name}' was LLM-verified. Keeping trusted field, removing '{other_field_name}'.")
                            continue
                        elif other_verification == 'pre_verified' and field_verification == 'llm_verified':
                            fields_to_remove_from_foreign.append(field_name)
                            removed_fields.add(field_name)
                            print(f"      Conflict: Same value '{field_value}'. '{other_field_name}' was pre-verified (trusted), '{field_name}' was LLM-verified. Keeping trusted field, removing '{field_name}'.")
                            continue

                        if sentence:
                            print(f"      Conflict: Both '{field_name}' and '{other_field_name}' have value '{field_value}'. Using LLM to check which field name is in sentence...")

                            all_other_fields = [fn for fn in data_fields.keys() if fn != field_name]
                            field_name_present = self.query_local_llm_for_field_detection(sentence, field_name, field_value, all_other_fields)
                            all_other_fields_for_other = [fn for fn in data_fields.keys() if fn != other_field_name]
                            other_field_name_present = self.query_local_llm_for_field_detection(sentence, other_field_name, other_field_value, all_other_fields_for_other)

                            if field_name_present and not other_field_name_present:
                                fields_to_remove_from_foreign.append(other_field_name)
                                removed_fields.add(other_field_name)
                                print(f"      LLM: '{field_name}' field name IS in sentence, '{other_field_name}' is NOT. Keeping '{field_name}', removing '{other_field_name}'.")
                            elif other_field_name_present and not field_name_present:
                                fields_to_remove_from_foreign.append(field_name)
                                removed_fields.add(field_name)
                                print(f"      LLM: '{other_field_name}' field name IS in sentence, '{field_name}' is NOT. Keeping '{other_field_name}', removing '{field_name}'.")
                            elif field_name_present and other_field_name_present:
                                match_strength_order = {
                                    'field_name_direct': 4,
                                    'field_name_word_by_word': 3,
                                    'field_name_llm': 2,
                                    'field_value': 1,
                                    '': 0
                                }
                                field_strength = match_strength_order.get(field_info.get('match_type', ''), 0)
                                other_strength = match_strength_order.get(other_field_info.get('match_type', ''), 0)

                                if field_strength > other_strength:
                                    fields_to_remove_from_foreign.append(other_field_name)
                                    removed_fields.add(other_field_name)
                                    print(f"      LLM: Both present. Keeping '{field_name}' (stronger match), removing '{other_field_name}'.")
                                elif other_strength > field_strength:
                                    fields_to_remove_from_foreign.append(field_name)
                                    removed_fields.add(field_name)
                                    print(f"      LLM: Both present. Keeping '{other_field_name}' (stronger match), removing '{field_name}'.")
                                else:
                                    if len(field_name) >= len(other_field_name):
                                        fields_to_remove_from_foreign.append(other_field_name)
                                        removed_fields.add(other_field_name)
                                        print(f"      LLM: Both present, same strength. Keeping '{field_name}' (longer), removing '{other_field_name}'.")
                                    else:
                                        fields_to_remove_from_foreign.append(field_name)
                                        removed_fields.add(field_name)
                                        print(f"      LLM: Both present, same strength. Keeping '{other_field_name}' (longer), removing '{field_name}'.")
                            else:
                                match_strength_order = {
                                    'field_name_direct': 4,
                                    'field_name_word_by_word': 3,
                                    'field_name_llm': 2,
                                    'field_value': 1,
                                    '': 0
                                }
                                field_strength = match_strength_order.get(field_info.get('match_type', ''), 0)
                                other_strength = match_strength_order.get(other_field_info.get('match_type', ''), 0)

                                if field_strength > other_strength:
                                    fields_to_remove_from_foreign.append(other_field_name)
                                    removed_fields.add(other_field_name)
                                    print(f"      LLM: Neither definitively present. Keeping '{field_name}' (stronger match), removing '{other_field_name}'.")
                                elif other_strength > field_strength:
                                    fields_to_remove_from_foreign.append(field_name)
                                    removed_fields.add(field_name)
                                    print(f"      LLM: Neither definitively present. Keeping '{other_field_name}' (stronger match), removing '{field_name}'.")
                                else:
                                    if len(field_name) >= len(other_field_name):
                                        fields_to_remove_from_foreign.append(other_field_name)
                                        removed_fields.add(other_field_name)
                                        print(f"      LLM: Neither definitively present, same strength. Keeping '{field_name}' (longer), removing '{other_field_name}'.")
                                    else:
                                        fields_to_remove_from_foreign.append(field_name)
                                        removed_fields.add(field_name)
                                        print(f"      LLM: Neither definitively present, same strength. Keeping '{other_field_name}' (longer), removing '{field_name}'.")
                        else:
                            match_strength_order = {
                                'field_name_direct': 4,
                                'field_name_word_by_word': 3,
                                'field_name_llm': 2,
                                'field_value': 1,
                                '': 0
                            }
                            field_strength = match_strength_order.get(field_info.get('match_type', ''), 0)
                            other_strength = match_strength_order.get(other_field_info.get('match_type', ''), 0)

                            if field_strength > other_strength:
                                fields_to_remove_from_foreign.append(other_field_name)
                                removed_fields.add(other_field_name)
                                print(f"      Conflict: Same value. Keeping '{field_name}' (stronger match), removing '{other_field_name}'.")
                            elif other_strength > field_strength:
                                fields_to_remove_from_foreign.append(field_name)
                                removed_fields.add(field_name)
                                print(f"      Conflict: Same value. Keeping '{other_field_name}' (stronger match), removing '{field_name}'.")
                            else:
                                if len(field_name) >= len(other_field_name):
                                    fields_to_remove_from_foreign.append(other_field_name)
                                    removed_fields.add(other_field_name)
                                    print(f"      Conflict: Same value, same strength. Keeping '{field_name}' (longer), removing '{other_field_name}'.")
                                else:
                                    fields_to_remove_from_foreign.append(field_name)
                                    removed_fields.add(field_name)
                                    print(f"      Conflict: Same value, same strength. Keeping '{other_field_name}' (longer), removing '{field_name}'.")
                        continue
                    elif field_name in foreign_fields:
                        fields_to_remove_from_foreign.append(field_name)
                        removed_fields.add(field_name)
                        print(f"      Conflict: Same value '{field_value}'. Other field is primary, removing foreign field '{field_name}'.")
                        continue

        for field_name in fields_to_remove_from_primary:
            if field_name in primary_fields:
                primary_fields.remove(field_name)

        for field_name in fields_to_remove_from_foreign:
            if field_name in foreign_fields:
                foreign_fields.remove(field_name)

        return primary_fields, foreign_fields
