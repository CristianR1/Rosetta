"""Template creation pipeline for the DocumentTemplateSystem."""

import json
import os
import re
import string
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

from .text_utils import clean_field_string, is_misc_value, is_date_value, count_placeholders
from .config import MISC_CHARACTERS
from .models import SentenceTemplate
from .template_patterns import TemplatePatternMixin


class TemplateCreationMixin(TemplatePatternMixin):
    """Mixin providing the create_sentence_templates pipeline."""

    def create_sentence_templates(self, sentences: List[str], data_fields: Dict[str, str], database: str, table: str, context: str = None, num_fields_to_check=1000, base_dir: str = None, narrative_json_path: str = None, sentence_template_data: Dict[str, Any] = None, narrative_template_data: Dict[str, Any] = None, sentence_json_path: str = None):
        """Create templates for all sentences - one template per data field per sentence."""

        if context is None:
            context = self.detect_document_context(' '.join(sentences), data_fields)
            print(f"Auto-detected document context: {context}")

        self.document_context = context

        print("Processing field metadata...")
        field_metadata, field_values, field_data_types = self.identify_binary_null_fields(database, table, data_fields, num_fields_to_check)

        print(f"Creating templates for {len(sentences)} sentences with {len(data_fields)} data fields...")
        print("Pipeline: sentences WITH (Hash: …) -> single-column resolution; WITHOUT hash -> static variations only")
        print("Primary key tracking: Each field can only be primary once, but can be foreign multiple times")
        print("Binary field handling: Y/N and T/F fields will be matched by both full and natural names")

        num_primary_keys_used = 0
        primary_keys_used_list = set()
        common_fields = self.variation_generator.identify_common_language_fields(data_fields)
        primary_fields = []
        for name in common_fields:
            primary_fields.append(name)
        from .template_generator import TemplateGenerator as TG

        narrative_dirty = False
        i = 0
        while i < len(sentences):
            raw = (sentences[i] or "").strip()
            frag_hash = TG.extract_hash(raw)
            frag_text = TG.strip_hash(raw)
            word_count = len(re.findall(r'\b[a-zA-Z]+\b', frag_text))

            if not frag_hash and 0 < word_count < 4 and i + 1 < len(sentences):
                if frag_text.strip().upper() == "TBD":
                    i += 1
                    continue

                next_raw = (sentences[i + 1] or "").strip()
                next_hash = TG.extract_hash(next_raw)
                next_text = TG.strip_hash(next_raw)

                if next_text.strip().upper() == "TBD":
                    i += 1
                    continue

                if next_text and len(re.findall(r'\b[a-zA-Z]+\b', next_text)) >= 2:
                    frag_trimmed = frag_text.rstrip()
                    if frag_trimmed.endswith(','):
                        merged_body = frag_trimmed + ' ' + next_text[0].lower() + next_text[1:]
                    else:
                        merged_body = frag_trimmed + ' ' + next_text

                    if next_hash:
                        merged_entry = f"{merged_body} (Hash: {next_hash})"
                    else:
                        merged_entry = merged_body

                    print(f"  [FRAGMENT-MERGE] \"{frag_text}\" + \"{next_text[:60]}...\"")
                    print(f"    -> \"{merged_body[:100]}...\"")

                    if narrative_template_data is not None and narrative_json_path:
                        narrative_list = narrative_template_data.get('narrative', [])
                        if isinstance(narrative_list, list):
                            if next_hash:
                                next_with_hash = next_text + f" (Hash: {next_hash})"
                                merged_with_hash = merged_body + f" (Hash: {next_hash})"
                                merge_pattern = re.compile(
                                    r'\|\s*' + re.escape(frag_text) + r'\s*\|'
                                    r'\s*\|?\s*'
                                    + re.escape(next_with_hash) + r'\s*\|'
                                )
                                replacement_segment = f'| {merged_with_hash} |'
                            else:
                                merge_pattern = re.compile(
                                    r'\|\s*' + re.escape(frag_text) + r'\s*\|'
                                    r'\s*\|?\s*'
                                    + re.escape(next_text) + r'\s*\|'
                                )
                                replacement_segment = f'| {merged_body} |'

                            updated = []
                            replaced_once = False
                            for para in narrative_list:
                                if not replaced_once and merge_pattern.search(para):
                                    para = merge_pattern.sub(replacement_segment, para, count=1)
                                    replaced_once = True
                                updated.append(para)
                            narrative_template_data['narrative'] = updated
                            if replaced_once:
                                narrative_dirty = True

                    sentences[i] = merged_entry
                    sentences.pop(i + 1)
                    continue
            i += 1

        if narrative_dirty and narrative_json_path:
            try:
                with open(narrative_json_path, 'w', encoding='utf-8') as f:
                    json.dump(narrative_template_data, f, indent=2, ensure_ascii=False)
                print(f"  [FRAGMENT-MERGE] narrative template saved ({narrative_json_path})")
            except Exception as e:
                print(f"  [FRAGMENT-MERGE] WARNING: could not save narrative: {e}")

        for i, sentence in enumerate(sentences):
            sentence_stripped = (sentence or "").strip()
            if not sentence_stripped or not re.findall(r'\b[a-zA-Z]+\b', sentence_stripped):
                print(f"  SKIPPED - sentence {i+1} is empty or has no words; not included in template output")
                continue
            sentence_hash = TG.extract_hash(sentence_stripped)
            sentence = TG.strip_hash(sentence_stripped)

            if sentence == "TBD":
                print(f"  [sentence {i+1}/{len(sentences)}] SKIPPED — marked as TBD (complex embedded value)")
                continue

            if not sentence_hash:
                print(f"  [sentence {i+1}/{len(sentences)}] mode=STATIC (no Hash): {sentence[:80]}...")

                detected_leaks = []
                for field_name, field_value in data_fields.items():
                    field_value_str = str(field_value)

                    name_found, name_match_type = self.check_name_in_sentence(
                        sentence, field_name, field_value_str,
                        use_llm_fallback=False, data_fields=data_fields,
                    )

                    value_found = self._value_present_in_sentence_strict(sentence, field_value_str)

                    partial_phrase = ""
                    if not value_found and len(field_value_str) > 3:
                        tokens = re.split(r'[\s,\-/]+', field_value_str)
                        meaningful_tokens = [t for t in tokens if len(t) > 2]
                        has_partial = any(
                            self._value_present_in_sentence_strict(sentence, tok)
                            for tok in meaningful_tokens
                        )
                        if has_partial:
                            partial_phrase = self.variation_generator.confirm_partial_value_bleed(
                                sentence, field_name, field_value_str,
                                database=database, table=table,
                            )
                            if partial_phrase:
                                print(f"    [STATIC-GUARD] LLM confirmed partial leak for "
                                      f"'{field_name}': \"{partial_phrase}\"")

                    if name_found or value_found or partial_phrase:
                        leak_type = []
                        if name_found:
                            leak_type.append(f"name({name_match_type})")
                        if value_found:
                            leak_type.append("value")
                        if partial_phrase:
                            leak_type.append(f"partial(\"{partial_phrase}\")")
                        detected_leaks.append({
                            'field_name': field_name,
                            'field_value': partial_phrase if partial_phrase and not value_found else field_value_str,
                            'detected_as': '+'.join(leak_type),
                        })

                if detected_leaks:
                    leak_names = [d['field_name'] for d in detected_leaks]
                    print(f"    [STATIC-GUARD] data bleed detected — fields: {leak_names}")
                    sentence = self.variation_generator.scrub_static_sentence_data_bleed(
                        sentence, detected_leaks,
                    )
                    print(f"    [STATIC-GUARD] scrubbed sentence: {sentence[:80]}...")

                    still_contaminated = False
                    for field_name, field_value in data_fields.items():
                        field_value_str = str(field_value)
                        name_found, _ = self.check_name_in_sentence(
                            sentence, field_name, field_value_str,
                            use_llm_fallback=False, data_fields=data_fields,
                        )
                        value_found = self._value_present_in_sentence_strict(sentence, field_value_str)
                        if name_found or value_found:
                            still_contaminated = True
                            print(f"    [STATIC-GUARD] still contains data for field '{field_name}' after scrub")
                            break
                    if still_contaminated:
                        print(f"    [STATIC-GUARD] sentence still contaminated — skipping from template output")
                        continue
                    print(f"    [STATIC-GUARD] sentence clean after scrub — proceeding")

                print(f"    No hash tag — generating static variations only")

                static_variations = self.variation_generator.generate_structural_variations_static(
                    sentence, context=context
                )

                words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
                stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
                field_name_words = {
                    'code', 'name', 'number', 'type', 'status', 'count', 'id', 'identifier',
                    'date', 'time', 'year', 'percent', 'percentage', 'total', 'amount', 'value'
                }
                filtered_words = [w for w in words if w not in stop_words and w not in field_name_words and len(w) > 3]
                key_words = list(set(filtered_words))[:8]

                lexical_sets = {}
                if key_words:
                    print(f"    Generating lexical variations for static sentence words: {key_words}")
                    lexical_sets = self.variation_generator.generate_lexical_variations(key_words, context)

                static_template = SentenceTemplate(
                    original=sentence,
                    template_pattern=sentence,
                    primary_data_fields=[],
                    foreign_data_fields=[],
                    variations=static_variations,
                    counter_variations=[],
                    lexical_sets=lexical_sets,
                    field_data_types={},
                    is_static=True
                )
                self.templates.append(static_template)
                continue

            print(f"  [sentence {i+1}/{len(sentences)}] mode=HASH (hash={sentence_hash}): {sentence[:80]}...")

            if base_dir and narrative_json_path:
                from .data_loader import load_column_descriptors, get_sample_entries
                bd = base_dir if base_dir is not None else getattr(
                    self, "_docgen_base_dir", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                ds = getattr(self, "dataset_folder_name", "MINIDEV")
                column_descriptors = load_column_descriptors(bd)
                table_desc = column_descriptors.get(database, {}).get(table, {})
                if not table_desc and data_fields:
                    table_desc = {col: {"descriptor": f"Column: {col}", "data_type": "TEXT"} for col in data_fields}
                descriptors_lines = [f"- {col}: {info.get('descriptor', '')} ({info.get('data_type', '')})" for col, info in table_desc.items()]
                descriptors = "\n".join(descriptors_lines)
                sample_entries = get_sample_entries(bd, database, table, 1, dataset_folder_name=ds)
                values_str = json.dumps(sample_entries[0], indent=2) if sample_entries else "{}"
                corrected = self.variation_generator.context_bleed_preventer(sentence, descriptors, values_str)
                def _well_formed(s: str) -> bool:
                    s = (s or "").strip()
                    if not s or len(s) < 10:
                        return False
                    if not s[0].isupper():
                        return False
                    if s[-1] not in '.!?':
                        return False
                    return True
                if corrected and corrected.strip() != sentence.strip() and _well_formed(corrected):
                    print(f"    Replaced sentence due to context bleed:\n      Before: {sentence}\n      After:  {corrected}")
                    try:
                        with open(narrative_json_path, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                        narrative = template_data.get('narrative', [])
                        narrative_str = '\n\n'.join(narrative) if isinstance(narrative, list) else str(narrative)
                        new_narrative_str = re.sub(re.escape(sentence), corrected, narrative_str, count=1)
                        if new_narrative_str != narrative_str:
                            sentence = corrected
                    except Exception as e:
                        print(f"    Context bleed JSON update skipped: {e}")

            detected_fields = self.resolve_hashed_field(
                sentence, sentence_hash, data_fields, field_metadata
            )

            if not detected_fields:
                print(f"    Hash resolution returned no field — creating static template for this sentence")
                static_variations = self.variation_generator.generate_structural_variations_static(
                    sentence, context=context
                )
                static_template = SentenceTemplate(
                    original=sentence,
                    template_pattern=sentence,
                    primary_data_fields=[],
                    foreign_data_fields=[],
                    variations=static_variations,
                    counter_variations=[],
                    lexical_sets={},
                    field_data_types={},
                    is_static=True
                )
                self.templates.append(static_template)
                continue

            resolved_col = list(detected_fields.keys())[0]
            resolved_source = detected_fields[resolved_col].get('match_type', 'unknown')

            if resolved_source == 'needs_regeneration':
                print(f"    [REGEN-TRIGGER] sentence for column='{resolved_col}' flagged for regeneration")
                regen_sentence, regen_hash, regen_fields = self.regenerate_sentence_with_validation(
                    column=resolved_col,
                    data_fields=data_fields,
                    field_metadata=field_metadata,
                    database=database,
                    table=table,
                    sentence_index=i,
                    sentences=sentences,
                    sentence_template_data=sentence_template_data or {},
                    narrative_template_data=narrative_template_data,
                    narrative_json_path=narrative_json_path,
                    sentence_json_path=sentence_json_path,
                    base_dir=base_dir,
                )
                if regen_fields:
                    sentence = regen_sentence
                    sentence_hash = regen_hash
                    detected_fields = regen_fields
                    resolved_col = list(detected_fields.keys())[0]
                    resolved_source = detected_fields[resolved_col].get('match_type', 'unknown')
                    print(f"    [REGEN-RESULT] sentence regenerated — column='{resolved_col}' (source={resolved_source})")
                else:
                    print(f"    [REGEN-RESULT] regeneration exhausted — creating static template for this sentence")
                    static_variations = self.variation_generator.generate_structural_variations_static(
                        regen_sentence, context=context
                    )
                    static_template = SentenceTemplate(
                        original=regen_sentence,
                        template_pattern=regen_sentence,
                        primary_data_fields=[],
                        foreign_data_fields=[],
                        variations=static_variations,
                        counter_variations=[],
                        lexical_sets={},
                        field_data_types={},
                        is_static=True
                    )
                    self.templates.append(static_template)
                    continue

            print(f"    [HASH-RESULT] Resolved {len(detected_fields)} field: {resolved_col} (source={resolved_source})")

            primary_fields, foreign_fields = self.assign_field_roles(detected_fields, primary_keys_used_list)
            print(f"    Primary fields after assignment: {primary_fields}")
            print(f"    Foreign fields after assignment: {foreign_fields}")
            primary_fields, foreign_fields = self.remove_conflicting_fields(detected_fields, data_fields, primary_fields, foreign_fields, sentence, field_metadata)
            print(f"    Primary fields after conflicting fields removal: {primary_fields}")
            print(f"    Foreign fields after conflicting fields removal: {foreign_fields}")
            remaining_fields = primary_fields + foreign_fields
            detected_fields = {k: v for k, v in detected_fields.items() if k in remaining_fields}
            num_primary_keys_used = len(primary_keys_used_list)

            if not detected_fields:
                print(f"  All fields removed due to conflicts - creating static template")
                static_template = SentenceTemplate(
                    original=sentence,
                    template_pattern=sentence,
                    primary_data_fields=[],
                    foreign_data_fields=[],
                    variations=[],
                    counter_variations=[],
                    lexical_sets={},
                    field_data_types={},
                    is_static=True
                )
                self.templates.append(static_template)
                continue

            template_pattern = self.build_template_pattern(sentence, detected_fields, primary_fields, foreign_fields)

            all_fields_in_sentence = list(detected_fields.keys())

            print(f"  Generating structural variations...")
            variation_sentence = sentence
            for foreign_field in foreign_fields:
                foreign_value = str(data_fields[foreign_field])
                placeholder = f"[{foreign_field.upper()}]"
                variation_sentence = variation_sentence.replace(placeholder, foreign_value)

            primary_field_type = "STANDARD"
            foreign_field_type = "STANDARD"
            generating_field_name = primary_fields[0] if primary_fields else foreign_fields[0]
            field_value = ""
            generative_field_type = "STANDARD"
            if primary_fields:
                for primary_field in primary_fields:
                    pf_meta = field_metadata[primary_field]
                    if pf_meta == "NULLABLE_BINARY":
                        primary_field_type = "NULLABLE_BINARY"
                        field_value = data_fields[primary_field]
                        generating_field_name = primary_field
                        generative_field_type = "NULLABLE_BINARY"
                        break
                    elif pf_meta == "NULL":
                        primary_field_type = "NULL"
                        field_value = data_fields[primary_field]
                        generating_field_name = primary_field
                        generative_field_type = "NULL"
                        break
                    elif primary_field_type not in ("NULL", "NULLABLE_BINARY") and pf_meta == "BINARY":
                        field_value = data_fields[primary_field]
                        primary_field_type = "BINARY"
                        generating_field_name = primary_field
                        generative_field_type = "BINARY"
                        break
            else:
                if foreign_fields:
                    for foreign_field in foreign_fields:
                        ff_meta = field_metadata[foreign_field]
                        if ff_meta == "NULLABLE_BINARY":
                            foreign_field_type = "NULLABLE_BINARY"
                            field_value = data_fields[foreign_field]
                            generating_field_name = foreign_field
                            generative_field_type = "NULLABLE_BINARY"
                            break
                        elif ff_meta == "NULL":
                            foreign_field_type = "NULL"
                            field_value = data_fields[foreign_field]
                            generating_field_name = foreign_field
                            generative_field_type = "NULL"
                            break
                        elif foreign_field_type not in ("NULL", "NULLABLE_BINARY") and ff_meta == "BINARY":
                            foreign_field_type = "BINARY"
                            generating_field_name = foreign_field
                            generative_field_type = "BINARY"
                            field_value = data_fields[foreign_field]
                            break
            generating_field_data_type = field_data_types.get(generating_field_name, "string") if generating_field_name else "string"
            print(f"    Generating field: {generating_field_name} (data type: {generating_field_data_type})")

            if generating_field_data_type != "NULL" and generating_field_name and generative_field_type in ("NULL", "NULLABLE_BINARY"):
                dummy_value = self.fetch_dummy_value(generating_field_name, database, table, num_fields_to_check)

            if self.null_mode == "explicit":
                null_replacement_phrase = "NULL"
            else:
                null_replacement_phrase = "not specified"

            nullable_binary_null_variations = None

            if generative_field_type == "NULLABLE_BINARY":
                actual_value = str(data_fields[generating_field_name])
                is_actual_null = actual_value.upper() in ["NULL", "NONE"] or actual_value.strip() == ""

                if is_actual_null:
                    one_sentence = self.variation_generator.generate_sentence_for_field_value(
                        generating_field_name, "1", context, natural_mode=False, original_sentence=variation_sentence)
                    print(f"    Generated 1-case base sentence: {one_sentence[:80]}...")
                    variations = self.variation_generator.generate_structural_variations_standard_with_style(
                        one_sentence, generating_field_name, context, original_sentence=variation_sentence)

                    zero_sentence = self.variation_generator.generate_sentence_for_field_value(
                        generating_field_name, "0", context, natural_mode=False, original_sentence=variation_sentence)
                    print(f"    Generated 0-case base sentence: {zero_sentence[:80]}...")
                    counter_variations = self.variation_generator.generate_structural_variations_standard_with_style(
                        zero_sentence, generating_field_name, context, original_sentence=variation_sentence)

                    nullable_binary_null_variations = self.variation_generator.generate_null_variations_null(
                        variation_sentence, generating_field_name,
                        detected_fields[generating_field_name]['replacement_value'], context)
                elif actual_value == "1":
                    variations = self.variation_generator.generate_structural_variations_binary(
                        variation_sentence, generating_field_name, "1", context)
                    zero_sentence = self.variation_generator.generate_sentence_for_field_value(
                        generating_field_name, "0", context, natural_mode=False, original_sentence=variation_sentence)
                    print(f"      Generated 0-case base sentence: {zero_sentence[:80]}...")
                    counter_variations = self.variation_generator.generate_structural_variations_standard_with_style(
                        zero_sentence, generating_field_name, context, original_sentence=variation_sentence)
                    null_base_sentence = self.variation_generator.generate_sentence_for_field_value(
                        generating_field_name, null_replacement_phrase, context, natural_mode=False, original_sentence=variation_sentence)
                    nullable_binary_null_variations = self.variation_generator.generate_null_variations_null(
                        null_base_sentence, generating_field_name, null_replacement_phrase, context)
                else:
                    one_sentence = self.variation_generator.generate_sentence_for_field_value(
                        generating_field_name, "1", context, natural_mode=False, original_sentence=variation_sentence)
                    print(f"      Generated 1-case base sentence: {one_sentence[:80]}...")
                    variations = self.variation_generator.generate_structural_variations_standard_with_style(
                        one_sentence, generating_field_name, context, original_sentence=variation_sentence)
                    counter_variations = self.variation_generator.generate_structural_variations_binary(
                        variation_sentence, generating_field_name, "0", context)
                    null_base_sentence = self.variation_generator.generate_sentence_for_field_value(
                        generating_field_name, null_replacement_phrase, context, natural_mode=False, original_sentence=variation_sentence)
                    nullable_binary_null_variations = self.variation_generator.generate_null_variations_null(
                        null_base_sentence, generating_field_name, null_replacement_phrase, context)

                print(f"    Generated {len(variations)} standard(1) variations, {len(counter_variations)} counter(0) variations, and {len(nullable_binary_null_variations)} null variations for NULLABLE_BINARY field type")

            elif generative_field_type in ("NULL", "BINARY"):
                if(generative_field_type == "BINARY"):
                    actual_field_value = str(data_fields[generating_field_name])
                    if actual_field_value == "1":
                        print(f"    BINARY: actual value is 1 — standard variations use 1, counter variations use 0")
                        variations = self.variation_generator.generate_structural_variations_binary(variation_sentence, generating_field_name, "1", context)
                        counter_variations = self.variation_generator.generate_structural_variations_binary_counter(variation_sentence, generating_field_name, "1", context)
                    else:
                        print(f"    BINARY: actual value is 0 — generating 1-case sentence for standard, using existing for counter(0)")
                        one_sentence = self.variation_generator.generate_sentence_for_field_value(
                            generating_field_name, "1", context, natural_mode=False, original_sentence=variation_sentence)
                        print(f"    Generated 1-case base sentence: {one_sentence[:80]}...")
                        variations = self.variation_generator.generate_structural_variations_standard_with_style(
                            one_sentence, generating_field_name, context, original_sentence=variation_sentence)
                        counter_variations = self.variation_generator.generate_structural_variations_binary(
                            variation_sentence, generating_field_name, "0", context)
                    print(f"    Generated {len(variations)} standard(1) variations and {len(counter_variations)} counter(0) variations for BINARY field type (actual: {actual_field_value})")
                else:
                    null_variations = []
                    non_null_variations = []

                    if field_value != "NULL":
                        null_variations = self.variation_generator.generate_null_variations_non_null(variation_sentence, generating_field_name, "NULL", context, null_replacement_phrase=null_replacement_phrase)
                        non_null_variations = self.variation_generator.generate_nonnull_variations_non_null(variation_sentence, generating_field_name, field_value, context)
                    elif field_value == "NULL":
                        null_variations = self.variation_generator.generate_null_variations_null(variation_sentence, generating_field_name, detected_fields[generating_field_name]['replacement_value'], context)
                        non_null_variations = self.variation_generator.generate_nonnull_variations_null(variation_sentence, generating_field_name, dummy_value, context)
                    else:
                        print(f"    Warning: Unexpected field_value '{field_value}' for NULL field type. Using standard generation.")
                        non_null_variations = self.variation_generator.generate_structural_variations_standard(variation_sentence, generating_field_name, context)
                        null_variations = []

                    variations = non_null_variations
                    counter_variations = null_variations
                    print(f"    Generated {len(variations)} variations and {len(counter_variations)} counter variations for NULL field type")
            else:
                variations = self.variation_generator.generate_structural_variations_standard(variation_sentence, generating_field_name, context)
                counter_variations = None

            print(f"    Converting variations to use placeholders...")
            print(f"    Found {len(variations)} variations to process")
            if counter_variations:
                print(f"    Found {len(counter_variations)} counter variations to process")
            else:
                print(f"    No counter variations to process")
            if nullable_binary_null_variations:
                print(f"    Found {len(nullable_binary_null_variations)} null variations to process (NULLABLE_BINARY)")
            if generative_field_type == "STANDARD":
                print(f"    Field Type Standard Processing Variations as Normal")
                processed_variations = []
                processed_counter_variations = []
                total_replacements = 0

                ordered_fields = []
                for f in primary_fields:
                    if f in all_fields_in_sentence and f not in foreign_fields:
                        ordered_fields.append(f)
                for f in foreign_fields:
                    if f in all_fields_in_sentence and f not in primary_fields:
                        ordered_fields.append(f)
                for f in all_fields_in_sentence:
                    if f not in ordered_fields:
                        ordered_fields.append(f)

                expected_replacements_per_variation = len([f for f in ordered_fields if not (f in primary_fields and f in foreign_fields)])
                expected_total_replacements = expected_replacements_per_variation * len(variations)

                for var_idx, variation in enumerate(variations):
                    processed_variation = variation
                    variation_replacements = 0
                    fields_replaced = set()

                    for field_name in ordered_fields:
                        if field_name in primary_fields and field_name in foreign_fields:
                            continue

                        field_value_str = str(data_fields[field_name])
                        replacement_value = detected_fields[field_name]['replacement_value']
                        placeholder = f"[{field_name.upper()}]"

                        if placeholder in processed_variation:
                            fields_replaced.add(field_name)
                            variation_replacements += 1
                        else:
                            processed_variation, replaced, count = self.try_safe_replace_values(
                                processed_variation, field_name, field_value_str, replacement_value, placeholder
                            )
                            if replaced:
                                variation_replacements += count
                                fields_replaced.add(field_name)
                            else:
                                print(f"        Attempting LLM fallback for field '{field_name}' in variation {var_idx + 1}")
                                processed_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                    processed_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if llm_replacements > 0:
                                    variation_replacements += llm_replacements
                                    fields_replaced.add(field_name)

                    total_replacements += variation_replacements

                    actual_placeholders = count_placeholders(processed_variation)
                    if actual_placeholders < expected_replacements_per_variation:
                        print(f"        Regenerating variation {var_idx + 1} (placeholders: {actual_placeholders}/{expected_replacements_per_variation})...")
                        for retry in range(1):
                            new_variations = self.variation_generator.generate_structural_variations_standard(variation_sentence, generating_field_name, context, silent=True)
                            if new_variations:
                                retry_processed = new_variations[0]
                                for fn in ordered_fields:
                                    if fn in primary_fields and fn in foreign_fields:
                                        continue
                                    fv = str(data_fields[fn])
                                    rv = detected_fields[fn]['replacement_value']
                                    ph = f"[{fn.upper()}]"
                                    retry_processed, _, _ = self.try_safe_replace_values(retry_processed, fn, fv, rv, ph)
                                if count_placeholders(retry_processed) == expected_replacements_per_variation:
                                    processed_variation = retry_processed
                                    break

                    processed_variations.append(processed_variation)

                    print(f"      Variation {var_idx + 1}:")
                    print(f"        Before: {variation}")
                    print(f"        After:  {processed_variation}")
                    print(f"        Placeholders: {count_placeholders(processed_variation)}/{expected_replacements_per_variation}")

                print(f"    STANDARD: Replacements made in this sentence: {total_replacements} (expected: {expected_total_replacements})")

            elif generative_field_type == "NULL":
                print(f"    Field Type NULL Processing Variations as Null")
                processed_counter_variations = []
                processed_variations = []
                total_regular_replacements = 0
                total_counter_replacements = 0

                original_generating_field_value = str(data_fields.get(generating_field_name, "NULL"))
                original_was_null = (original_generating_field_value == "NULL" or original_generating_field_value.upper() == "NULL")

                ordered_fields = []
                for f in primary_fields:
                    if f in all_fields_in_sentence and f not in foreign_fields:
                        ordered_fields.append(f)
                for f in foreign_fields:
                    if f in all_fields_in_sentence and f not in primary_fields:
                        ordered_fields.append(f)
                for f in all_fields_in_sentence:
                    if f not in ordered_fields:
                        ordered_fields.append(f)

                expected_regular_replacements_per_variation = len([f for f in ordered_fields if not (f in primary_fields and f in foreign_fields)])
                expected_regular_total_replacements = expected_regular_replacements_per_variation * len(variations)

                for var_idx, variation in enumerate(variations):
                    processed_variation = variation
                    variation_replacements = 0
                    fields_replaced = set()

                    for field_name in ordered_fields:
                        if field_name in primary_fields and field_name in foreign_fields:
                            continue

                        placeholder = f"[{field_name.upper()}]"
                        if field_name == generating_field_name and original_was_null:
                            replacement_value = dummy_value
                        else:
                            replacement_value = detected_fields[field_name]['replacement_value']
                        field_value_str = str(data_fields[field_name])

                        if placeholder in processed_variation:
                            fields_replaced.add(field_name)
                            variation_replacements += 1
                        else:
                            processed_variation, replaced, count = self.try_safe_replace_values(
                                processed_variation, field_name, field_value_str, replacement_value, placeholder
                            )
                            if replaced:
                                variation_replacements += count
                                fields_replaced.add(field_name)
                            else:
                                print(f"        Attempting LLM fallback for field '{field_name}' in variation {var_idx + 1}")
                                processed_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                    processed_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if llm_replacements > 0:
                                    variation_replacements += llm_replacements
                                    fields_replaced.add(field_name)

                    total_regular_replacements += variation_replacements

                    actual_placeholders = count_placeholders(processed_variation)
                    if actual_placeholders < expected_regular_replacements_per_variation:
                        print(f"        Regenerating variation {var_idx + 1} (placeholders: {actual_placeholders}/{expected_regular_replacements_per_variation})...")
                        for retry in range(1):
                            new_variations = self.variation_generator.generate_structural_variations_standard(variation_sentence, generating_field_name, context, silent=True)
                            if new_variations:
                                retry_processed = new_variations[0]
                                for fn in ordered_fields:
                                    if fn in primary_fields and fn in foreign_fields:
                                        continue
                                    ph = f"[{fn.upper()}]"
                                    rv = dummy_value if (fn == generating_field_name and original_was_null) else detected_fields[fn]['replacement_value']
                                    fv = str(data_fields[fn])
                                    retry_processed, _, _ = self.try_safe_replace_values(retry_processed, fn, fv, rv, ph)
                                if count_placeholders(retry_processed) == expected_regular_replacements_per_variation:
                                    processed_variation = retry_processed
                                    break

                    processed_variations.append(processed_variation)

                    print(f"      Variation {var_idx + 1}:")
                    print(f"        Before: {variation}")
                    print(f"        After:  {processed_variation}")
                    print(f"        Placeholders: {count_placeholders(processed_variation)}/{expected_regular_replacements_per_variation}")

                generating_field_placeholder = f"[{generating_field_name.upper()}]"
                if counter_variations and len(counter_variations) > 0:
                    print(f"    Processing counter variations")
                    processed_counter_variations = []
                    expected_counter_replacements_per_variation = len([f for f in ordered_fields if f != generating_field_name and not (f in primary_fields and f in foreign_fields)])
                    expected_counter_total_replacements = expected_counter_replacements_per_variation * len(counter_variations)

                    for var_idx, counter_variation in enumerate(counter_variations):
                        processed_counter_variation = counter_variation
                        counter_variation_replacements = 0
                        fields_replaced = set()

                        for field_name in ordered_fields:
                            if field_name in primary_fields and field_name in foreign_fields:
                                continue
                            if field_name == generating_field_name:
                                continue

                            placeholder = f"[{field_name.upper()}]"
                            replacement_value = detected_fields[field_name]['replacement_value']
                            field_value_str = str(data_fields[field_name])

                            if placeholder in processed_counter_variation:
                                fields_replaced.add(field_name)
                                counter_variation_replacements += 1
                            else:
                                processed_counter_variation, replaced, count = self.try_safe_replace_values(
                                    processed_counter_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if replaced:
                                    counter_variation_replacements += count
                                    fields_replaced.add(field_name)
                                else:
                                    print(f"        Attempting LLM fallback for field '{field_name}' in counter variation {var_idx + 1}")
                                    processed_counter_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                        processed_counter_variation, field_name, field_value_str, replacement_value, placeholder
                                    )
                                    if llm_replacements > 0:
                                        counter_variation_replacements += llm_replacements
                                        fields_replaced.add(field_name)

                        total_counter_replacements += counter_variation_replacements

                        actual_counter_placeholders = count_placeholders(processed_counter_variation)
                        if actual_counter_placeholders < expected_counter_replacements_per_variation or generating_field_placeholder in processed_counter_variation:
                            print(f"        Regenerating counter variation {var_idx + 1} (placeholders: {actual_counter_placeholders}/{expected_counter_replacements_per_variation})...")
                            retry_null_phrase = "NULL" if self.null_mode == "explicit" else "not specified"
                            for retry in range(1):
                                new_counter_variations = self.variation_generator.generate_null_variations_null(variation_sentence, generating_field_name, retry_null_phrase, context, silent=True)
                                if new_counter_variations:
                                    retry_counter = new_counter_variations[0]
                                    for fn in ordered_fields:
                                        if fn in primary_fields and fn in foreign_fields:
                                            continue
                                        if fn == generating_field_name:
                                            continue
                                        ph = f"[{fn.upper()}]"
                                        rv = detected_fields[fn]['replacement_value']
                                        fv = str(data_fields[fn])
                                        retry_counter, _, _ = self.try_safe_replace_values(retry_counter, fn, fv, rv, ph)
                                    retry_placeholders = count_placeholders(retry_counter)
                                    if retry_placeholders == expected_counter_replacements_per_variation and generating_field_placeholder not in retry_counter:
                                        processed_counter_variation = retry_counter
                                        break

                        processed_counter_variations.append(processed_counter_variation)

                        print(f"      Counter Variation {var_idx + 1}:")
                        print(f"        Source: Null variation ")
                        print(f"        Before: {counter_variation}")
                        print(f"        After:  {processed_counter_variation}")
                        print(f"        Placeholders: {count_placeholders(processed_counter_variation)}/{expected_counter_replacements_per_variation}")
                else:
                    processed_counter_variations = []

                print(f"    NULL: Regular variations replacements in this sentence: {total_regular_replacements} (expected: {expected_regular_total_replacements})")
                print(f"    NULL: Counter variations replacements in this sentence: {total_counter_replacements}")
            elif generative_field_type == "NULLABLE_BINARY":
                print(f"    Field Type NULLABLE_BINARY Processing Variations (standard=1, counter=0, null=absent)")
                processed_variations = []
                processed_counter_variations = []
                processed_nullable_binary_null_variations = []
                nb_field_name = generating_field_name
                total_nb_standard_replacements = 0
                total_nb_counter_replacements = 0
                total_nb_null_replacements = 0

                nb_ordered_fields = []
                for f in primary_fields:
                    if f in all_fields_in_sentence and f not in foreign_fields and f != nb_field_name:
                        nb_ordered_fields.append(f)
                for f in foreign_fields:
                    if f in all_fields_in_sentence and f not in primary_fields and f != nb_field_name:
                        nb_ordered_fields.append(f)
                for f in all_fields_in_sentence:
                    if f not in nb_ordered_fields and f != nb_field_name:
                        nb_ordered_fields.append(f)

                expected_nb_replacements_per_variation = len([f for f in nb_ordered_fields if not (f in primary_fields and f in foreign_fields)])

                print(f"    Processing standard (1-case) variations...")
                for var_idx, variation in enumerate(variations):
                    processed_variation = variation
                    variation_replacements = 0
                    fields_replaced = set()

                    for field_name in nb_ordered_fields:
                        if field_name in primary_fields and field_name in foreign_fields:
                            continue

                        field_value_str = str(data_fields[field_name])
                        replacement_value = detected_fields[field_name]['replacement_value']
                        placeholder = f"[{field_name.upper()}]"

                        if placeholder in processed_variation:
                            fields_replaced.add(field_name)
                            variation_replacements += 1
                        else:
                            processed_variation, replaced, count = self.try_safe_replace_values(
                                processed_variation, field_name, field_value_str, replacement_value, placeholder
                            )
                            if replaced:
                                variation_replacements += count
                                fields_replaced.add(field_name)
                            else:
                                print(f"        Attempting LLM fallback for field '{field_name}' in standard variation {var_idx + 1}")
                                processed_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                    processed_variation, field_name, field_value_str, replacement_value, placeholder)
                                if llm_replacements > 0:
                                    variation_replacements += llm_replacements
                                    fields_replaced.add(field_name)

                    total_nb_standard_replacements += variation_replacements
                    processed_variations.append(processed_variation)
                    print(f"      Standard Variation {var_idx + 1}:")
                    print(f"        Before: {variation}")
                    print(f"        After:  {processed_variation}")
                    print(f"        Placeholders: {count_placeholders(processed_variation)}/{expected_nb_replacements_per_variation}")

                print(f"    Processing counter (0-case) variations...")
                if counter_variations:
                    for var_idx, variation in enumerate(counter_variations):
                        processed_counter_variation = variation
                        counter_variation_replacements = 0
                        fields_replaced = set()

                        for field_name in nb_ordered_fields:
                            if field_name in primary_fields and field_name in foreign_fields:
                                continue

                            field_value_str = str(data_fields[field_name])
                            replacement_value = detected_fields[field_name]['replacement_value']
                            placeholder = f"[{field_name.upper()}]"

                            if placeholder in processed_counter_variation:
                                fields_replaced.add(field_name)
                                counter_variation_replacements += 1
                            else:
                                processed_counter_variation, replaced, count = self.try_safe_replace_values(
                                    processed_counter_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if replaced:
                                    counter_variation_replacements += count
                                    fields_replaced.add(field_name)
                                else:
                                    print(f"        Attempting LLM fallback for field '{field_name}' in counter variation {var_idx + 1}")
                                    processed_counter_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                        processed_counter_variation, field_name, field_value_str, replacement_value, placeholder)
                                    if llm_replacements > 0:
                                        counter_variation_replacements += llm_replacements
                                        fields_replaced.add(field_name)

                        total_nb_counter_replacements += counter_variation_replacements

                        print(f"        Applying LLM refinement to binary counter variation {var_idx + 1}...")
                        refined_counter_variation = self.query_local_llm_for_counter_variation_refinement(
                            processed_counter_variation, self.null_mode, self.binary_mode)

                        actual_counter_placeholders = count_placeholders(refined_counter_variation)
                        if actual_counter_placeholders < expected_nb_replacements_per_variation:
                            refined_counter_variation = processed_counter_variation

                        processed_counter_variations.append(refined_counter_variation)
                        print(f"      Counter Variation {var_idx + 1}:")
                        print(f"        Before: {variation}")
                        print(f"        After replacement:  {processed_counter_variation}")
                        print(f"        After refinement:   {refined_counter_variation}")
                        print(f"        Placeholders: {count_placeholders(refined_counter_variation)}/{expected_nb_replacements_per_variation}")

                nb_generating_field_placeholder = f"[{nb_field_name.upper()}]"
                if nullable_binary_null_variations and len(nullable_binary_null_variations) > 0:
                    print(f"    Processing null (absent) variations...")
                    expected_nb_null_replacements_per_variation = len([f for f in nb_ordered_fields if f != nb_field_name and not (f in primary_fields and f in foreign_fields)])

                    for var_idx, null_var in enumerate(nullable_binary_null_variations):
                        processed_null_variation = null_var
                        null_variation_replacements = 0
                        fields_replaced = set()

                        for field_name in nb_ordered_fields:
                            if field_name in primary_fields and field_name in foreign_fields:
                                continue
                            if field_name == nb_field_name:
                                continue

                            placeholder = f"[{field_name.upper()}]"
                            replacement_value = detected_fields[field_name]['replacement_value']
                            field_value_str = str(data_fields[field_name])

                            if placeholder in processed_null_variation:
                                fields_replaced.add(field_name)
                                null_variation_replacements += 1
                            else:
                                processed_null_variation, replaced, count = self.try_safe_replace_values(
                                    processed_null_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if replaced:
                                    null_variation_replacements += count
                                    fields_replaced.add(field_name)
                                else:
                                    print(f"        Attempting LLM fallback for field '{field_name}' in null variation {var_idx + 1}")
                                    processed_null_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                        processed_null_variation, field_name, field_value_str, replacement_value, placeholder)
                                    if llm_replacements > 0:
                                        null_variation_replacements += llm_replacements
                                        fields_replaced.add(field_name)

                        total_nb_null_replacements += null_variation_replacements
                        processed_nullable_binary_null_variations.append(processed_null_variation)

                        print(f"      Null Variation {var_idx + 1}:")
                        print(f"        Source: Null variation (natural '{null_replacement_phrase}' language)")
                        print(f"        Before: {null_var}")
                        print(f"        After:  {processed_null_variation}")
                        print(f"        Placeholders: {count_placeholders(processed_null_variation)}/{expected_nb_null_replacements_per_variation}")

                print(f"    NULLABLE_BINARY: Standard(1) replacements: {total_nb_standard_replacements}")
                print(f"    NULLABLE_BINARY: Counter(0) replacements: {total_nb_counter_replacements}")
                print(f"    NULLABLE_BINARY: Null replacements: {total_nb_null_replacements}")
            else:
                print(f"    Field Type BINARY Processing Variations as Binary")
                processed_variations = []
                processed_counter_variations = []
                binary_field_name = generating_field_name
                total_binary_regular_replacements = 0
                total_binary_counter_replacements = 0

                ordered_fields = []
                for f in primary_fields:
                    if f in all_fields_in_sentence and f not in foreign_fields and f != binary_field_name:
                        ordered_fields.append(f)
                for f in foreign_fields:
                    if f in all_fields_in_sentence and f not in primary_fields and f != binary_field_name:
                        ordered_fields.append(f)
                for f in all_fields_in_sentence:
                    if f not in ordered_fields and f != binary_field_name:
                        ordered_fields.append(f)

                expected_binary_regular_replacements_per_variation = len([f for f in ordered_fields if not (f in primary_fields and f in foreign_fields)])
                expected_binary_regular_total_replacements = expected_binary_regular_replacements_per_variation * len(variations)

                for var_idx, variation in enumerate(variations):
                    processed_variation = variation
                    variation_replacements = 0
                    fields_replaced = set()

                    for field_name in ordered_fields:
                        if field_name in primary_fields and field_name in foreign_fields:
                            continue

                        field_value_str = str(data_fields[field_name])
                        replacement_value = detected_fields[field_name]['replacement_value']
                        placeholder = f"[{field_name.upper()}]"

                        if placeholder in processed_variation:
                            fields_replaced.add(field_name)
                            variation_replacements += 1
                        else:
                            processed_variation, replaced, count = self.try_safe_replace_values(
                                processed_variation, field_name, field_value_str, replacement_value, placeholder
                            )
                            if replaced:
                                variation_replacements += count
                                fields_replaced.add(field_name)
                            else:
                                print(f"        Attempting LLM fallback for field '{field_name}' in variation {var_idx + 1}")
                                processed_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                    processed_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if llm_replacements > 0:
                                    variation_replacements += llm_replacements
                                    fields_replaced.add(field_name)

                    total_binary_regular_replacements += variation_replacements

                    actual_placeholders = count_placeholders(processed_variation)
                    if actual_placeholders < expected_binary_regular_replacements_per_variation:
                        print(f"        Regenerating variation {var_idx + 1} (placeholders: {actual_placeholders}/{expected_binary_regular_replacements_per_variation})...")
                        for retry in range(1):
                            new_variations = self.variation_generator.generate_structural_variations_binary(variation_sentence, binary_field_name, str(data_fields[binary_field_name]), context, silent=True)
                            if new_variations:
                                retry_processed = new_variations[0]
                                for fn in ordered_fields:
                                    if fn in primary_fields and fn in foreign_fields:
                                        continue
                                    fv = str(data_fields[fn])
                                    rv = detected_fields[fn]['replacement_value']
                                    ph = f"[{fn.upper()}]"
                                    retry_processed, _, _ = self.try_safe_replace_values(retry_processed, fn, fv, rv, ph)
                                if count_placeholders(retry_processed) == expected_binary_regular_replacements_per_variation:
                                    processed_variation = retry_processed
                                    break

                    processed_variations.append(processed_variation)

                    print(f"      Variation {var_idx + 1}:")
                    print(f"        Before: {variation}")
                    print(f"        After:  {processed_variation}")
                    print(f"        Placeholders: {count_placeholders(processed_variation)}/{expected_binary_regular_replacements_per_variation}")

                if counter_variations:
                    expected_binary_counter_replacements_per_variation = len([f for f in ordered_fields if not (f in primary_fields and f in foreign_fields)])
                    expected_binary_counter_total_replacements = expected_binary_counter_replacements_per_variation * len(counter_variations)

                    for var_idx, variation in enumerate(counter_variations):
                        processed_counter_variation = variation
                        counter_variation_replacements = 0
                        fields_replaced = set()

                        for field_name in ordered_fields:
                            if field_name in primary_fields and field_name in foreign_fields:
                                continue

                            field_value_str = str(data_fields[field_name])
                            replacement_value = detected_fields[field_name]['replacement_value']
                            placeholder = f"[{field_name.upper()}]"

                            if placeholder in processed_counter_variation:
                                fields_replaced.add(field_name)
                                counter_variation_replacements += 1
                            else:
                                processed_counter_variation, replaced, count = self.try_safe_replace_values(
                                    processed_counter_variation, field_name, field_value_str, replacement_value, placeholder
                                )
                                if replaced:
                                    counter_variation_replacements += count
                                    fields_replaced.add(field_name)
                                else:
                                    print(f"        Attempting LLM fallback for field '{field_name}' in counter variation {var_idx + 1}")
                                    processed_counter_variation, llm_replacements = self.query_local_llm_for_value_replacement(
                                        processed_counter_variation, field_name, field_value_str, replacement_value, placeholder
                                    )
                                    if llm_replacements > 0:
                                        counter_variation_replacements += llm_replacements
                                        fields_replaced.add(field_name)

                        total_binary_counter_replacements += counter_variation_replacements

                        print(f"        Applying LLM refinement to binary counter variation {var_idx + 1}...")
                        refined_counter_variation = self.query_local_llm_for_counter_variation_refinement(
                            processed_counter_variation, self.null_mode, self.binary_mode
                        )

                        actual_counter_placeholders = count_placeholders(refined_counter_variation)
                        if actual_counter_placeholders < expected_binary_counter_replacements_per_variation:
                            print(f"        Regenerating counter variation {var_idx + 1}...")
                            actual_field_value = str(data_fields[binary_field_name])
                            for retry in range(1):
                                new_counter_variations = self.variation_generator.generate_structural_variations_binary_counter(variation_sentence, binary_field_name, actual_field_value, context, silent=True)
                                if new_counter_variations:
                                    retry_counter = new_counter_variations[0]
                                    for fn in ordered_fields:
                                        if fn in primary_fields and fn in foreign_fields:
                                            continue
                                        fv = str(data_fields[fn])
                                        rv = detected_fields[fn]['replacement_value']
                                        ph = f"[{fn.upper()}]"
                                        retry_counter, _, _ = self.try_safe_replace_values(retry_counter, fn, fv, rv, ph)
                                    retry_refined = self.query_local_llm_for_counter_variation_refinement(retry_counter, self.null_mode, self.binary_mode)
                                    if count_placeholders(retry_refined) == expected_binary_counter_replacements_per_variation:
                                        refined_counter_variation = retry_refined
                                        break

                        processed_counter_variations.append(refined_counter_variation)

                        print(f"     Counter Variation {var_idx + 1}:")
                        print(f"        Before: {variation}")
                        print(f"        After replacement:  {processed_counter_variation}")
                        print(f"        After refinement:   {refined_counter_variation}")
                        print(f"        Placeholders: {count_placeholders(refined_counter_variation)}/{expected_binary_counter_replacements_per_variation}")

                print(f"    BINARY: Regular variations replacements in this sentence: {total_binary_regular_replacements} (expected: {expected_binary_regular_total_replacements})")
                print(f"    BINARY: Counter variations replacements in this sentence: {total_binary_counter_replacements} (expected: {expected_binary_counter_total_replacements})")

            cleaned_variations = self.clean_malformed_variations(processed_variations)
            if processed_counter_variations:
                cleaned_counter_variations = self.clean_malformed_variations(processed_counter_variations)
            else:
                cleaned_counter_variations = []
            cleaned_null_variations = None
            if generative_field_type == "NULLABLE_BINARY" and processed_nullable_binary_null_variations:
                cleaned_null_variations = self.clean_malformed_variations(processed_nullable_binary_null_variations)
                print(f"    Successfully converted {len(cleaned_null_variations)} null variations to use placeholders (NULLABLE_BINARY)")
            print(f"    Successfully converted {len(cleaned_variations)} variations to use placeholders")
            print(f"    Successfully converted {len(cleaned_counter_variations)} counter variations to use placeholders")
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
            filtered_words = []
            stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}

            field_name_words = set()
            for field_name in data_fields.keys():
                field_words = re.findall(r'\b[a-zA-Z]+\b', field_name.lower())
                field_name_words.update(field_words)

            common_field_words = {'code', 'name', 'number', 'type', 'status', 'count', 'id', 'identifier',
                                'date', 'time', 'year', 'percent', 'percentage', 'total', 'amount', 'value'}
            field_name_words.update(common_field_words)

            proper_noun_words = set()
            for field_value in data_fields.values():
                field_value_str = str(field_value)
                words_in_value = field_value_str.split()
                for word_in_value in words_in_value:
                    word_clean = re.sub(r'[^\w]', '', word_in_value).lower()
                    if word_clean and len(word_clean) > 1:
                        proper_noun_words.add(word_clean)

            for word in words:
                if (word not in [str(v).lower() for v in data_fields.values()] and
                    word not in stop_words and
                    word not in field_name_words and
                    word not in proper_noun_words and
                    len(word) > 3):
                    filtered_words.append(word)

            key_words = list(set(filtered_words))[:8]

            print(f"    Generating lexical variations for words: {key_words}")
            lexical_sets = self.variation_generator.generate_lexical_variations(key_words, context)

            template_field_data_types = {}
            for field_name in all_fields_in_sentence:
                if field_name in field_data_types:
                    template_field_data_types[field_name] = field_data_types[field_name]

            template = SentenceTemplate(
                original=sentence,
                template_pattern=template_pattern,
                primary_data_fields=primary_fields,
                foreign_data_fields=foreign_fields,
                variations=cleaned_variations,
                counter_variations=cleaned_counter_variations,
                lexical_sets=lexical_sets,
                field_data_types=template_field_data_types,
                is_static=False,
                null_variations=cleaned_null_variations
            )

            self.templates.append(template)

            time.sleep(0.5)
