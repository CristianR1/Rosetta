"""Narrative parsing analysis - analyzes templates for column detection."""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cost_tracker import get_cost_tracker, track_openai_response

from .narrative_llm import NarrativeLLMMixin
from .narrative_detection import NarrativeDetectionMixin
from .config import (
    get_mode_folder_name,
    get_noise_ratio_folder_name,
    get_output_root,
    get_ground_truth_column_descriptors_enhanced_path,
    create_local_llm_openai_client,
)
from .models import ColumnAnalysis, NarrativeAnalysis
from .text_utils import identify_field_metadata, get_detection_patterns
from .template_generator import TemplateGenerator


class NarrativeParsingAnalyzer(NarrativeLLMMixin, NarrativeDetectionMixin):
    def __init__(self, templates_dir: str = None, enhanced_descriptors_path: str = None,
                 null_mode: str = "implicit", binary_mode: str = "implicit",
                 selected_tables: List[Tuple[str, str]] = None,
                 data_noise_x: int = 1, data_noise_y: int = 0,
                 dataset_folder_name: str = "MINIDEV"):
        self.null_mode = null_mode
        self.binary_mode = binary_mode
        self.data_noise_x = data_noise_x
        self.data_noise_y = data_noise_y
        self.selected_tables = selected_tables
        self.dataset_folder_name = dataset_folder_name

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir

        mode_folder = get_mode_folder_name(null_mode, binary_mode)
        ratio_folder = get_noise_ratio_folder_name(data_noise_x, data_noise_y)

        if templates_dir is None:
            templates_dir = os.path.join(get_output_root(base_dir), "templates", mode_folder, "sentence_templates")

        self.templates_dir = templates_dir
        self.narrative_templates_dir = os.path.join(get_output_root(base_dir), "templates", mode_folder, "narrative_templates", ratio_folder)

        if enhanced_descriptors_path is None:
            enhanced_descriptors_path = get_ground_truth_column_descriptors_enhanced_path(base_dir)

        self.enhanced_descriptors_path = enhanced_descriptors_path
        self.output_dir = get_output_root(base_dir)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.column_descriptors = self.load_column_descriptors()

        self.local_llm_client = create_local_llm_openai_client()
        self._init_detection_patterns()

    def _init_detection_patterns(self):
        """Initialize detection patterns based on null_mode and binary_mode."""
        self.null_patterns, self.binary_true_patterns, self.binary_false_patterns = get_detection_patterns(
            self.null_mode, self.binary_mode
        )

    def load_column_descriptors(self) -> Dict[str, Any]:
        try:
            with open(self.enhanced_descriptors_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading enhanced column descriptors: {e}")
            return {}

    def get_all_template_files(self) -> List[str]:
        """
        Get all template files from the nested folder structure.
        Templates are organized as: {templates_dir}/{db_name}/{table_name}_template.json

        If selected_tables is set, only returns templates for those specific tables.
        """
        template_files = []
        if os.path.exists(self.templates_dir):
            if self.selected_tables:
                for db_name, table_name in self.selected_tables:
                    template_file = os.path.join(self.templates_dir, db_name, f"{table_name}_template.json")
                    if os.path.exists(template_file):
                        template_files.append(template_file)
                    else:
                        print(f"Warning: Template file not found for {db_name}.{table_name}: {template_file}")
            else:
                for db_name in os.listdir(self.templates_dir):
                    db_path = os.path.join(self.templates_dir, db_name)
                    if os.path.isdir(db_path):
                        for file in os.listdir(db_path):
                            if file.endswith('_template.json'):
                                template_files.append(os.path.join(db_path, file))
        return sorted(template_files)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        sentences = re.split(r'(?<!\d)\.(?!\d)|[!?]+', text)
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def analyze_narrative(self, template_data: Dict[str, Any], narrative_data: Dict[str, Any] = None) -> NarrativeAnalysis:
        """Analyze a single narrative template for column detection using hash-based validation."""
        database = template_data.get('database', '')
        table = template_data.get('table', '')
        original_data = template_data.get('original_data', {})
        generated_sentences = template_data.get('generated_sentences', [])

        if narrative_data is not None:
            narrative = narrative_data.get('narrative', [])
        else:
            narrative = template_data.get('narrative', [])

        generator = TemplateGenerator(
            null_mode=self.null_mode, binary_mode=self.binary_mode,
            data_noise_x=self.data_noise_x, data_noise_y=self.data_noise_y,
            dataset_folder_name=self.dataset_folder_name,
        )

        if isinstance(narrative, list):
            narrative_text = ' '.join(narrative)
        else:
            narrative_text = str(narrative)

        print(f"\n{'='*60}")
        print(f"Analyzing: {database} -> {table}")
        print(f"{'='*60}")
        print(f"Columns to analyze: {len(original_data)}")
        print(f"Narrative length: {len(narrative_text)} characters")

        field_metadata = identify_field_metadata(original_data)

        field_to_sentence = {}
        field_names = list(original_data.keys())
        for idx, field_name in enumerate(field_names):
            if idx < len(generated_sentences):
                field_to_sentence[field_name] = generated_sentences[idx]

        column_analyses = []
        detected_count = 0
        generated_detections = {}
        generated_detected_count = 0
        tbd_columns = template_data.get('tbd_columns', [])

        for column_name, column_value in original_data.items():
            if column_name in tbd_columns:
                generated_detections[column_name] = {
                    'found': True,
                    'replacement_value': 'TBD',
                    'matched_name': column_name,
                    'detection_method': 'tbd_skipped',
                    'detected_sentence': 'TBD',
                    'replacement_attempted': False,
                    'replacement_succeeded': False,
                    'replaced_sentence': ''
                }
                generated_detected_count += 1
                continue

            found = False
            replacement_value = ""
            matched_name = ""
            detection_method = "not_detected"
            detected_sentence = ""
            replacement_attempted = False
            replacement_succeeded = False
            replaced_sentence = ""

            if column_name in field_to_sentence:
                generated_sentence = field_to_sentence[column_name]
                field_type = field_metadata.get(column_name, "STANDARD")

                found, replacement_value, matched_name, detection_method = self.check_field_in_sentence(
                    generated_sentence, column_name, column_value, field_metadata,
                    type_override=None, use_llm_fallback=True, data_fields=original_data
                )

                if found:
                    detected_sentence = generated_sentence
                    generated_detected_count += 1
                    replacement_attempted = True
                    replacement_succeeded, replaced_sentence = self.attempt_replacement(
                        generated_sentence, column_name, replacement_value, field_type
                    )

            generated_detections[column_name] = {
                'found': found,
                'replacement_value': replacement_value,
                'matched_name': matched_name,
                'detection_method': detection_method,
                'detected_sentence': detected_sentence,
                'replacement_attempted': replacement_attempted,
                'replacement_succeeded': replacement_succeeded,
                'replaced_sentence': replaced_sentence
            }

        hash_to_replacement = {}

        if self.data_noise_y == 0:
            print("  No noise, verifying hashes and hash count...")
            expected_hashes = []
            for column_name, column_value in original_data.items():
                if column_name in tbd_columns:
                    analysis = ColumnAnalysis(
                        column_name=column_name,
                        column_value=str(column_value),
                        detected=True,
                        detection_method='tbd_skipped',
                        matched_text='TBD',
                        confidence='high',
                        field_type='COMPLEX',
                        detected_sentence='TBD',
                        replacement_attempted=False,
                        replacement_succeeded=False,
                        replaced_sentence=''
                    )
                    column_analyses.append(analysis)
                    detected_count += 1
                    continue

                gen_det = generated_detections.get(column_name, {})
                gen_sentence = field_to_sentence.get(column_name, "")
                sentence_hash = TemplateGenerator.extract_hash(gen_sentence)

                found = gen_det.get('found', False)
                replacement_value = gen_det.get('replacement_value', '')
                detection_method = gen_det.get('detection_method', 'not_detected')
                replacement_attempted = gen_det.get('replacement_attempted', False)
                replacement_succeeded = gen_det.get('replacement_succeeded', False)
                replaced_sentence = gen_det.get('replaced_sentence', '')
                detected_sentence = gen_det.get('detected_sentence', gen_sentence)

                if sentence_hash:
                    expected_hashes.append(sentence_hash)
                    if found:
                        hash_to_replacement[sentence_hash] = replacement_value

                confidence = "high"
                if detection_method == "field_name_only":
                    confidence = "medium"
                elif detection_method == "field_value_only":
                    confidence = "low"
                elif detection_method == "not_detected":
                    confidence = "none"

                analysis = ColumnAnalysis(
                    column_name=column_name,
                    column_value=str(column_value),
                    detected=found,
                    detection_method=detection_method,
                    matched_text=replacement_value,
                    confidence=confidence,
                    field_type=field_metadata.get(column_name, "STANDARD"),
                    detected_sentence=detected_sentence if detected_sentence else gen_sentence,
                    replacement_attempted=replacement_attempted,
                    replacement_succeeded=replacement_succeeded,
                    replaced_sentence=replaced_sentence
                )
                column_analyses.append(analysis)
                if found:
                    detected_count += 1

            generated_hash_count = len(expected_hashes)
            replacement_hash_count = len(hash_to_replacement)
            non_tbd_columns = len([c for c in original_data.keys() if c not in tbd_columns])

            if generated_hash_count == replacement_hash_count and generated_hash_count == non_tbd_columns:
                print(f"    Hash verification passed: {generated_hash_count} hashes for {non_tbd_columns} columns")
            else:
                print(f"    WARNING: Hash count mismatch - expected {non_tbd_columns} columns, found {generated_hash_count} hashes, {replacement_hash_count} replacements")
                missing_hashes = [h for h in expected_hashes if h not in hash_to_replacement]
                if missing_hashes:
                    print(f"    Missing replacement hashes: {missing_hashes[:5]}{'...' if len(missing_hashes) > 5 else ''}")
        else:
            print("  Phase 2: Hash-based narrative validation...")

            for column_name, column_value in original_data.items():
                if column_name in tbd_columns:
                    analysis = ColumnAnalysis(
                        column_name=column_name,
                        column_value=str(column_value),
                        detected=True,
                        detection_method='tbd_skipped',
                        matched_text='TBD',
                        confidence='high',
                        field_type='COMPLEX',
                        detected_sentence='TBD',
                        replacement_attempted=False,
                        replacement_succeeded=False,
                        replaced_sentence=''
                    )
                    column_analyses.append(analysis)
                    detected_count += 1
                    continue

                gen_det = generated_detections.get(column_name, {})
                gen_sentence = field_to_sentence.get(column_name, "")
                sentence_hash = TemplateGenerator.extract_hash(gen_sentence)

                found = gen_det.get('found', False)
                replacement_value = gen_det.get('replacement_value', '')
                detection_method = gen_det.get('detection_method', 'not_detected')
                replacement_attempted = gen_det.get('replacement_attempted', False)
                replacement_succeeded = gen_det.get('replacement_succeeded', False)
                replaced_sentence = gen_det.get('replaced_sentence', '')
                detected_sentence = gen_det.get('detected_sentence', gen_sentence)

                narrative_validated = False
                if found and sentence_hash:
                    narrative_sentence = self._find_sentence_by_hash(narrative_text, sentence_hash)
                    if narrative_sentence:
                        print(f"Identified Sentence: {narrative_sentence}")
                        narr_found, narr_repl, _, _ = self.check_field_in_sentence(
                            narrative_sentence, column_name, column_value, field_metadata,
                            type_override=None, use_llm_fallback=False, data_fields=original_data
                        )
                        if narr_found:
                            narrative_validated = True
                            hash_to_replacement[sentence_hash] = narr_repl
                            print(f"    {column_name}: hash-validated in narrative (replacement: '{narr_repl[:50]}')")
                        else:
                            llm_found, llm_repl = self.query_local_llm_for_targeted_verification(
                                narrative_sentence, column_name, column_value,
                                field_metadata.get(column_name, "STANDARD")
                            )
                            if llm_found:
                                narrative_validated = True
                                hash_to_replacement[sentence_hash] = llm_repl
                                print(f"    {column_name}: hash-validated via targeted LLM (replacement: '{llm_repl[:50]}')")
                            else:
                                print(f"    {column_name}: narrative sentence lacks data - rewriting and molding...")
                                descriptor = generator.get_column_descriptor(column_name, database, table)
                                field_type = field_metadata.get(column_name, "STANDARD")
                                corrected = self.rewrite_sentence_for_detectability(
                                    narrative_sentence, column_name, column_value, field_type,
                                    descriptor, database, table, preserve_hash=sentence_hash
                                )
                                narrative_text = self.mold_sentence_into_narrative(
                                    narrative_sentence, corrected, narrative_text
                                )
                                new_narr_sentence = self._find_sentence_by_hash(narrative_text, sentence_hash)
                                if new_narr_sentence:
                                    narr_found2, narr_repl2, _, _ = self.check_field_in_sentence(
                                        new_narr_sentence, column_name, column_value, field_metadata,
                                        type_override=None, use_llm_fallback=False, data_fields=original_data
                                    )
                                    if narr_found2:
                                        narrative_validated = True
                                        hash_to_replacement[sentence_hash] = narr_repl2
                                        print(f"    {column_name}: validated after mold (replacement: '{narr_repl2[:50]}')")
                    else:
                        print(f"    {column_name}: hash {sentence_hash} NOT found in narrative")

                if found and sentence_hash and not narrative_validated:
                    hash_to_replacement[sentence_hash] = replacement_value
                    print(f"    {column_name}: using generated sentence replacement (hash present but narrative validation skipped)")
                elif found and not sentence_hash:
                    print(f"    {column_name}: no hash, using generated sentence replacement")

                confidence = "high"
                if detection_method == "field_name_only":
                    confidence = "medium"
                elif detection_method == "field_value_only":
                    confidence = "low"
                elif detection_method == "not_detected":
                    confidence = "none"

                analysis = ColumnAnalysis(
                    column_name=column_name,
                    column_value=str(column_value),
                    detected=found,
                    detection_method=detection_method,
                    matched_text=replacement_value,
                    confidence=confidence,
                    field_type=field_metadata.get(column_name, "STANDARD"),
                    detected_sentence=detected_sentence if detected_sentence else gen_sentence,
                    replacement_attempted=replacement_attempted,
                    replacement_succeeded=replacement_succeeded,
                    replaced_sentence=replaced_sentence
                )
                column_analyses.append(analysis)
                if found:
                    detected_count += 1

            expected_hashes = TemplateGenerator.extract_all_hashes(
                [field_to_sentence.get(fn, '') for fn in field_names if fn not in tbd_columns]
            )
            all_present, missing = TemplateGenerator.verify_hashes_in_narrative(narrative_text, expected_hashes)
            if not all_present:
                print(f"  WARNING: {len(missing)} hashes missing from narrative after validation: {missing[:5]}")

        sentence_file_path = os.path.join(self.templates_dir, database, f"{table}_template.json")
        try:
            with open(sentence_file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
            print(f"  Sentence template updated and saved")
        except Exception as e:
            print(f"  Warning: Could not save sentence template: {e}")

        if self.data_noise_y > 0:
            narrative_file_path = os.path.join(self.narrative_templates_dir, database, f"{table}_template.json")
            Path(os.path.dirname(narrative_file_path)).mkdir(parents=True, exist_ok=True)

            if narrative_data is None:
                narrative_data = {}

            narrative_data['narrative'] = generator.format_narrative_for_json(narrative_text) if isinstance(narrative_text, str) else narrative_text
            narrative_data['hash_to_replacement'] = hash_to_replacement
            narrative_data['database'] = database
            narrative_data['table'] = table

            try:
                with open(narrative_file_path, 'w', encoding='utf-8') as f:
                    json.dump(narrative_data, f, indent=2, ensure_ascii=False)
                print(f"  Narrative template updated and saved")
            except Exception as e:
                print(f"  Warning: Could not save narrative template: {e}")

        detection_rate = (detected_count / len(original_data)) * 100 if original_data else 0

        return NarrativeAnalysis(
            database=database,
            table=table,
            total_columns=len(original_data),
            detected_columns=detected_count,
            undetected_columns=len(original_data) - detected_count,
            detection_rate=detection_rate,
            column_analyses=column_analyses,
            narrative_text=narrative_text
        )

    def _load_narrative_data(self, database: str, table: str) -> Optional[Dict[str, Any]]:
        """Load narrative data from the narrative templates directory."""
        if self.data_noise_y == 0:
            return None
        narrative_file = os.path.join(self.narrative_templates_dir, database, f"{table}_template.json")
        if os.path.exists(narrative_file):
            try:
                with open(narrative_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Warning: Could not load narrative template: {e}")
        return None

    def analyze_all_templates(self, auto_backfill: bool = False, skip_phase1: bool = False) -> Dict[str, Any]:
        """Analyze all template files and generate comprehensive report."""
        print("Starting Narrative Parsing Analysis")
        print("=" * 60)

        template_files = self.get_all_template_files()

        if not template_files:
            print("No template files found!")
            return {'error': 'No template files found'}

        print(f"Found {len(template_files)} sentence template files to analyze")
        print(f"Data:Noise ratio: {self.data_noise_x}:{self.data_noise_y}")

        all_template_data = []
        global_generated_detected = 0
        global_generated_total = 0
        generator = TemplateGenerator(
            null_mode=self.null_mode, binary_mode=self.binary_mode,
            data_noise_x=self.data_noise_x, data_noise_y=self.data_noise_y,
            dataset_folder_name=self.dataset_folder_name,
        )
        max_rewrite_attempts = 5

        if skip_phase1:
            print("\n" + "="*60)
            print("PHASE 1 SKIPPED (templates assumed valid)")
            print("="*60)
            for template_file in template_files:
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                    if auto_backfill:
                        needs_backfill = any(
                            TemplateGenerator.extract_hash(s) is None
                            for s in template_data.get('generated_sentences', [])
                            if s != "TBD"
                        )
                        if needs_backfill:
                            print(f"  Auto-backfilling hashes for {template_data.get('database', '')}.{template_data.get('table', '')}")
                            backfill_gen = TemplateGenerator(
                                null_mode=template_data.get('null_mode', self.null_mode),
                                binary_mode=template_data.get('binary_mode', self.binary_mode),
                                dataset_folder_name=self.dataset_folder_name,
                            )
                            if backfill_gen.backfill_template_hashes(template_file):
                                with open(template_file, 'r', encoding='utf-8') as f:
                                    template_data = json.load(f)
                            else:
                                print(f"  Warning: Backfill failed for {template_file}")
                    original_data = template_data.get('original_data', {})
                    generated_sentences = template_data.get('generated_sentences', [])
                    field_names = list(original_data.keys())
                    field_to_sentence = {}
                    for idx, field_name in enumerate(field_names):
                        if idx < len(generated_sentences):
                            field_to_sentence[field_name] = generated_sentences[idx]
                    field_metadata = identify_field_metadata(original_data)
                    all_template_data.append({
                        'file': template_file,
                        'data': template_data,
                        'field_to_sentence': field_to_sentence,
                        'field_metadata': field_metadata
                    })
                    print(f"  Loaded: {template_data.get('database', '')}.{template_data.get('table', '')}")
                except Exception as e:
                    print(f"Error loading {template_file}: {e}")
        else:
            print("\n" + "="*60)
            print("GLOBAL PHASE 1: Checking detection in generated sentences for ALL templates")
            print("="*60)

            for template_file in template_files:
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)

                    if auto_backfill:
                        needs_backfill = any(
                            TemplateGenerator.extract_hash(s) is None
                            for s in template_data.get('generated_sentences', [])
                            if s != "TBD"
                        )
                        if needs_backfill:
                            print(f"  Auto-backfilling hashes for {template_data.get('database', '')}.{template_data.get('table', '')}")
                            backfill_gen = TemplateGenerator(
                                null_mode=template_data.get('null_mode', self.null_mode),
                                binary_mode=template_data.get('binary_mode', self.binary_mode),
                                dataset_folder_name=self.dataset_folder_name,
                            )
                            if backfill_gen.backfill_template_hashes(template_file):
                                with open(template_file, 'r', encoding='utf-8') as f:
                                    template_data = json.load(f)
                            else:
                                print(f"  Warning: Backfill failed for {template_file}")

                    database = template_data.get('database', '')
                    table = template_data.get('table', '')
                    original_data = template_data.get('original_data', {})
                    generated_sentences = template_data.get('generated_sentences', [])

                    print(f"\nChecking: {database}.{table}")

                    field_to_sentence = {}
                    field_names = list(original_data.keys())
                    for idx, field_name in enumerate(field_names):
                        if idx < len(generated_sentences):
                            field_to_sentence[field_name] = generated_sentences[idx]

                    field_metadata = identify_field_metadata(original_data)

                    template_detected = 0
                    template_total = len(original_data)
                    tbd_columns = template_data.get('tbd_columns', [])
                    template_updated = False
                    remediated_sentences = []

                    for column_name, column_value in original_data.items():
                        if column_name in tbd_columns:
                            template_detected += 1
                            continue
                        if column_name not in field_to_sentence:
                            continue

                        generated_sentence = field_to_sentence[column_name]
                        column_index = field_names.index(column_name)
                        descriptor = generator.get_column_descriptor(column_name, database, table)
                        field_type = field_metadata.get(column_name, "STANDARD")

                        found, replacement_value, matched_name, detection_method = self.check_field_in_sentence(
                            generated_sentence, column_name, column_value, field_metadata, type_override=None, use_llm_fallback=True, data_fields=original_data
                        )

                        if found:
                            template_detected += 1
                        else:
                            print(f"     FAILED: {column_name} = {column_value} - attempting remediation...")
                            print(f"        Generated sentence: {generated_sentence[:100]}...")

                            old_sentence = generated_sentence
                            sentence_hash = TemplateGenerator.extract_hash(generated_sentence)
                            generated_sentence = self.rewrite_sentence_for_detectability(
                                generated_sentence, column_name, column_value, field_type, descriptor, database, table,
                                preserve_hash=sentence_hash
                            )
                            rewrite_attempt = 0
                            while rewrite_attempt < max_rewrite_attempts:
                                check1_found, check1_repl, check1_matched_name, check1_method = self.check_field_in_sentence(
                                    generated_sentence, column_name, column_value, field_metadata,
                                    type_override=None, use_llm_fallback=True, data_fields=original_data
                                )
                                check2_found, check2_repl, _, _ = self.check_field_in_sentence(
                                    generated_sentence, column_name, column_value, field_metadata,
                                    type_override=None, use_llm_fallback=True, data_fields=original_data
                                )
                                if check1_found and check2_found:
                                    found = True
                                    replacement_value, matched_name, detection_method = check1_repl, check1_matched_name, check1_method
                                    break
                                if not check1_found and not check2_found:
                                    print(f"        Both checks NO - rewriting again (attempt {rewrite_attempt + 2}/{max_rewrite_attempts})")
                                elif check1_found != check2_found:
                                    print(f"        Ambiguous (one YES one NO) - rewriting again (attempt {rewrite_attempt + 2}/{max_rewrite_attempts})")
                                rewrite_attempt += 1
                                if rewrite_attempt < max_rewrite_attempts:
                                    generated_sentence = self.rewrite_sentence_for_detectability(
                                        generated_sentence, column_name, column_value, field_type, descriptor, database, table,
                                        preserve_hash=sentence_hash
                                    )

                            if not found and rewrite_attempt >= max_rewrite_attempts:
                                print(f"        Rewrite attempts exhausted - falling back to fresh regeneration")
                                generated_sentence = generator.generate_sentence_for_column(column_name, column_value, descriptor, append_hash=True)
                                if sentence_hash:
                                    generated_sentence = TemplateGenerator.strip_hash(generated_sentence)
                                    generated_sentence = TemplateGenerator.append_hash(generated_sentence, sentence_hash)
                                found, replacement_value, matched_name, detection_method = self.check_field_in_sentence(
                                    generated_sentence, column_name, column_value, field_metadata,
                                    type_override=None, use_llm_fallback=True, data_fields=original_data
                                )
                                if not found:
                                    print(f"        Fallback regeneration also failed - column may remain undetected")

                            if found:
                                template_detected += 1
                                template_updated = True
                                remediated_sentences.append((old_sentence, generated_sentence))
                                print(f"        DETECTED after remediation")
                                field_to_sentence[column_name] = generated_sentence
                                if column_index < len(generated_sentences):
                                    generated_sentences[column_index] = generated_sentence
                                else:
                                    generated_sentences.append(generated_sentence)
                                template_data['generated_sentences'] = generated_sentences

                    if template_updated and template_detected == template_total:
                        narrative = template_data.get('narrative', [])
                        narrative_text = ' '.join(narrative) if isinstance(narrative, list) else str(narrative)
                        for old_sent, new_sent in remediated_sentences:
                            narrative_text = self.mold_sentence_into_narrative(old_sent, new_sent, narrative_text)
                        expected_hashes = TemplateGenerator.extract_all_hashes(
                            [field_to_sentence.get(fn, '') for fn in field_names if fn not in tbd_columns]
                        )
                        all_present, missing = TemplateGenerator.verify_hashes_in_narrative(narrative_text, expected_hashes)
                        if not all_present:
                            print(f"  WARNING: {len(missing)} hashes missing from narrative after mold: {missing[:5]}")
                        template_data['narrative'] = generator.format_narrative_for_json(narrative_text)
                        try:
                            with open(template_file, 'w', encoding='utf-8') as f:
                                json.dump(template_data, f, indent=2, ensure_ascii=False)
                            print(f"  Template updated and saved (targeted mold replacement for {len(remediated_sentences)} sentence(s))")
                        except Exception as save_err:
                            print(f"  Warning: Could not save updated template: {save_err}")

                    global_generated_detected += template_detected
                    global_generated_total += template_total

                    detection_rate = (template_detected / template_total * 100) if template_total > 0 else 0
                    print(f"  Generated sentence detection: {template_detected}/{template_total} ({detection_rate:.1f}%)")

                    all_template_data.append({
                        'file': template_file,
                        'data': template_data,
                        'field_to_sentence': field_to_sentence,
                        'field_metadata': field_metadata
                    })

                except Exception as e:
                    print(f"Error analyzing {template_file}: {e}")

        global_detection_rate = (global_generated_detected / global_generated_total * 100) if global_generated_total > 0 else None

        if not skip_phase1:
            print("\n" + "="*60)
            print(f"GLOBAL GENERATED SENTENCE DETECTION: {global_generated_detected}/{global_generated_total} ({global_detection_rate:.1f}%)")
            print("="*60)

        print("\n" + "="*60)
        print("PHASE 2: Full template analysis (hash-based narrative validation)")
        print("="*60)

        all_analyses = []
        summary_stats = {
            'total_templates': len(template_files),
            'total_columns': 0,
            'total_detected': 0,
            'total_undetected': 0,
            'overall_detection_rate': 0,
            'generated_sentence_detection_rate': global_detection_rate,
            'narrative_analysis_performed': True,
            'replacement_stats': {
                'total_attempted': 0,
                'total_succeeded': 0,
                'replacement_rate': 0
            },
            'by_database': {},
            'by_field_type': {
                'STANDARD': {'total': 0, 'detected': 0, 'replaced': 0},
                'BINARY': {'total': 0, 'detected': 0, 'replaced': 0},
                'NULL': {'total': 0, 'detected': 0, 'replaced': 0},
                'MISC': {'total': 0, 'detected': 0, 'replaced': 0}
            },
            'by_detection_method': {},
            'by_confidence': {
                'high': 0,
                'medium': 0,
                'low': 0,
                'none': 0
            }
        }

        for template_info in all_template_data:
            try:
                template_data = template_info['data']

                analysis = self.analyze_narrative(template_data)
                all_analyses.append(analysis)

                summary_stats['total_columns'] += analysis.total_columns
                summary_stats['total_detected'] += analysis.detected_columns
                summary_stats['total_undetected'] += analysis.undetected_columns

                if analysis.database not in summary_stats['by_database']:
                    summary_stats['by_database'][analysis.database] = {
                        'tables': 0,
                        'columns': 0,
                        'detected': 0,
                        'detection_rate': 0
                    }

                db_stats = summary_stats['by_database'][analysis.database]
                db_stats['tables'] += 1
                db_stats['columns'] += analysis.total_columns
                db_stats['detected'] += analysis.detected_columns
                db_stats['detection_rate'] = (db_stats['detected'] / db_stats['columns']) * 100 if db_stats['columns'] > 0 else 0

                for col_analysis in analysis.column_analyses:
                    method = col_analysis.detection_method or 'not_detected'
                    summary_stats['by_detection_method'][method] = summary_stats['by_detection_method'].get(method, 0) + 1
                    confidence = col_analysis.confidence or 'none'
                    summary_stats['by_confidence'][confidence] = summary_stats['by_confidence'].get(confidence, 0) + 1

                    field_type = col_analysis.field_type if col_analysis.field_type else 'STANDARD'
                    if field_type in summary_stats['by_field_type']:
                        summary_stats['by_field_type'][field_type]['total'] += 1
                        if col_analysis.detected:
                            summary_stats['by_field_type'][field_type]['detected'] += 1
                        if col_analysis.replacement_succeeded:
                            summary_stats['by_field_type'][field_type]['replaced'] += 1

                    if col_analysis.replacement_attempted:
                        summary_stats['replacement_stats']['total_attempted'] += 1
                        if col_analysis.replacement_succeeded:
                            summary_stats['replacement_stats']['total_succeeded'] += 1

            except Exception as e:
                print(f"   Error analyzing template: {e}")

        summary_stats['overall_detection_rate'] = (summary_stats['total_detected'] / summary_stats['total_columns']) * 100 if summary_stats['total_columns'] > 0 else 0

        if summary_stats['replacement_stats']['total_attempted'] > 0:
            summary_stats['replacement_stats']['replacement_rate'] = (
                summary_stats['replacement_stats']['total_succeeded'] /
                summary_stats['replacement_stats']['total_attempted'] * 100
            )

        detailed_analyses = []

        for analysis in all_analyses:
            detected_columns = []
            undetected_columns = []

            for col in analysis.column_analyses:
                col_data = {
                    'column_name': col.column_name,
                    'column_value': col.column_value,
                    'field_type': col.field_type,
                    'detection_method': col.detection_method,
                    'matched_text': col.matched_text,
                    'confidence': col.confidence,
                    'detected_sentence': col.detected_sentence,
                    'replacement_attempted': col.replacement_attempted,
                    'replacement_succeeded': col.replacement_succeeded,
                    'replaced_sentence': col.replaced_sentence
                }

                if col.detected:
                    detected_columns.append(col_data)
                else:
                    undetected_columns.append(col_data)

            detailed_analyses.append({
                'database': analysis.database,
                'table': analysis.table,
                'total_columns': analysis.total_columns,
                'detected_columns_count': analysis.detected_columns,
                'undetected_columns_count': analysis.undetected_columns,
                'detection_rate': analysis.detection_rate,
                'detected_columns': detected_columns,
                'undetected_columns': undetected_columns,
                'narrative_sample': analysis.narrative_text[:200] + "..." if len(analysis.narrative_text) > 200 else analysis.narrative_text
            })

        report = {
            'summary': summary_stats,
            'detailed_analyses': detailed_analyses,
            'problematic_tables': [
                {
                    'database': analysis.database,
                    'table': analysis.table,
                    'detection_rate': analysis.detection_rate,
                    'undetected_count': analysis.undetected_columns,
                    'undetected_columns': [
                        {
                            'column_name': col.column_name,
                            'column_value': col.column_value,
                            'reason': 'Field name and value not found in narrative'
                        }
                        for col in analysis.column_analyses if not col.detected
                    ]
                }
                for analysis in all_analyses if analysis.detection_rate < 70
            ],
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
        }

        return report

    def save_report(self, report: Dict[str, Any]) -> str:
        """Save the analysis report to JSON file."""
        report_file = os.path.join(self.output_dir, "narrative_parsing_analysis.json")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"   Analysis report saved to: {report_file}")
            return report_file
        except Exception as e:
            print(f"   Error saving report: {e}")
            return ""

    def create_simple_detection_report(self, all_analyses: List[NarrativeAnalysis]) -> Dict[str, Any]:
        """Create a simple report with just column names and detection status."""
        simple_report = {}

        for analysis in all_analyses:
            table_key = f"{analysis.database}.{analysis.table}"
            simple_report[table_key] = {
                'detection_rate': f"{analysis.detection_rate:.1f}%",
                'detected_count': f"{analysis.detected_columns}/{analysis.total_columns}",
                'columns': {}
            }

            for col_analysis in analysis.column_analyses:
                simple_report[table_key]['columns'][col_analysis.column_name] = {
                    'status': 'DETECTED' if col_analysis.detected else 'NOT DETECTED',
                    'value': col_analysis.column_value,
                    'field_type': col_analysis.field_type,
                    'method': col_analysis.detection_method if col_analysis.detected else 'none',
                    'detected_sentence': col_analysis.detected_sentence,
                    'replacement_attempted': col_analysis.replacement_attempted,
                    'replacement_succeeded': col_analysis.replacement_succeeded,
                    'replaced_sentence': col_analysis.replaced_sentence
                }

        return simple_report

    def save_simple_report(self, simple_report: Dict[str, Any]) -> str:
        """Save the simplified detection report."""
        simple_file = os.path.join(self.output_dir, "column_detection_simple.json")

        try:
            with open(simple_file, 'w', encoding='utf-8') as f:
                json.dump(simple_report, f, indent=2, ensure_ascii=False)

            print(f"    Simple detection report saved to: {simple_file}")
            return simple_file
        except Exception as e:
            print(f"    Error saving simple report: {e}")
            return ""

    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the analysis results."""
        summary = report['summary']

        print(f"\n{'='*80}")
        print("   NARRATIVE PARSING ANALYSIS SUMMARY")
        print(f"{'='*80}")

        print(f"    Templates analyzed: {summary['total_templates']}")
        print(f"    Total columns: {summary['total_columns']}")
        print(f"    Detected columns: {summary['total_detected']}")
        print(f"    Undetected columns: {summary['total_undetected']}")
        print(f"    Overall detection rate: {summary['overall_detection_rate']:.1f}%")

        print(f"\n  Detection Methods:")
        for method, count in summary['by_detection_method'].items():
            percentage = (count / summary['total_columns']) * 100 if summary['total_columns'] > 0 else 0
            print(f"  {method}: {count} ({percentage:.1f}%)")

        print(f"\n  Confidence Levels:")
        for confidence, count in summary['by_confidence'].items():
            percentage = (count / summary['total_columns']) * 100 if summary['total_columns'] > 0 else 0
            print(f"  {confidence}: {count} ({percentage:.1f}%)")

        print(f"\n  By Field Type (Detection & Replacement):")
        for field_type, stats in summary.get('by_field_type', {}).items():
            total = stats['total']
            detected = stats['detected']
            replaced = stats.get('replaced', 0)
            detection_rate = (detected / total * 100) if total > 0 else 0
            replacement_rate = (replaced / detected * 100) if detected > 0 else 0
            print(f"  {field_type}: {detected}/{total} detected ({detection_rate:.1f}%), {replaced}/{detected} replaced ({replacement_rate:.1f}%)")

        replacement_stats = summary.get('replacement_stats', {})
        if replacement_stats:
            print(f"\n  Replacement Statistics:")
            print(f"    Total attempted: {replacement_stats.get('total_attempted', 0)}")
            print(f"    Total succeeded: {replacement_stats.get('total_succeeded', 0)}")
            print(f"    Replacement rate: {replacement_stats.get('replacement_rate', 0):.1f}%")

        print(f"\n  By Database:")
        for db_name, db_stats in summary['by_database'].items():
            print(f"  {db_name}: {db_stats['detected']}/{db_stats['columns']} ({db_stats['detection_rate']:.1f}%) across {db_stats['tables']} tables")

        print(f"\n  Tables with Low Detection Rates (<70%):")
        for analysis in report['detailed_analyses']:
            if analysis['detection_rate'] < 70:
                print(f"  {analysis['database']}.{analysis['table']}: {analysis['detection_rate']:.1f}% ({analysis['detected_columns_count']}/{analysis['total_columns']})")


def main():
    """Main function to run the narrative parsing analysis."""
    parser = argparse.ArgumentParser(description="Narrative Parsing Analysis")
    parser.add_argument("--auto-backfill", action="store_true", help="Auto-backfill hashes on templates that lack them before analysis")
    args = parser.parse_args()

    print("    Starting Narrative Parsing Analysis")
    print("=" * 60)

    analyzer = NarrativeParsingAnalyzer()

    if not os.path.exists(analyzer.templates_dir):
        print(f"    Error: Templates directory not found at {analyzer.templates_dir}")
        print("Please run generate_all_templates.py first!")
        return

    if not os.path.exists(analyzer.enhanced_descriptors_path):
        print(f"    Error: Enhanced descriptors file not found at {analyzer.enhanced_descriptors_path}")
        print("Please run generate_column_descriptors.py first!")
        return

    report = analyzer.analyze_all_templates(auto_backfill=args.auto_backfill)

    if 'error' in report:
        print(f"    Analysis failed: {report['error']}")
        return

    all_analyses = []
    for analysis_data in report['detailed_analyses']:
        column_analyses = []
        for col in analysis_data.get('detected_columns', []):
            column_analyses.append(ColumnAnalysis(
                column_name=col['column_name'],
                column_value=col['column_value'],
                detected=True,
                detection_method=col['detection_method'],
                matched_text=col['matched_text'],
                confidence=col['confidence'],
                field_type=col.get('field_type', 'STANDARD'),
                detected_sentence=col.get('detected_sentence', ''),
                replacement_attempted=col.get('replacement_attempted', False),
                replacement_succeeded=col.get('replacement_succeeded', False),
                replaced_sentence=col.get('replaced_sentence', '')
            ))
        for col in analysis_data.get('undetected_columns', []):
            column_analyses.append(ColumnAnalysis(
                column_name=col['column_name'],
                column_value=col['column_value'],
                detected=False,
                detection_method=col['detection_method'],
                matched_text=col['matched_text'],
                confidence=col['confidence'],
                field_type=col.get('field_type', 'STANDARD'),
                detected_sentence=col.get('detected_sentence', ''),
                replacement_attempted=col.get('replacement_attempted', False),
                replacement_succeeded=col.get('replacement_succeeded', False),
                replaced_sentence=col.get('replaced_sentence', '')
            ))

        all_analyses.append(NarrativeAnalysis(
            database=analysis_data['database'],
            table=analysis_data['table'],
            total_columns=analysis_data['total_columns'],
            detected_columns=analysis_data['detected_columns_count'],
            undetected_columns=analysis_data['undetected_columns_count'],
            detection_rate=analysis_data['detection_rate'],
            column_analyses=column_analyses,
            narrative_text=analysis_data.get('narrative_sample', '')
        ))

    simple_report = analyzer.create_simple_detection_report(all_analyses)
    simple_file = analyzer.save_simple_report(simple_report)

    report_file = analyzer.save_report(report)

    analyzer.print_summary(report)

    print(f"\n  Detailed report saved to: {report_file}")
    print(f"    Simple column list saved to: {simple_file}")


if __name__ == "__main__":
    main()
