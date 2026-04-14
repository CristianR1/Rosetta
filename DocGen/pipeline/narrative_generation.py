"""Narrative generation and backfill methods for TemplateGenerator."""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from utils.cost_tracker import track_openai_response
from .config import (
    get_mode_folder_name, get_noise_ratio_folder_name,
    get_sentence_template_path, get_narrative_template_path,
    get_sentence_templates_dir, get_narrative_templates_dir,
    variation_templates_exist, compute_expected_transition_count,
    resolve_backfill_noise_xy_and_narrative_path,
    parse_data_noise_ratio_str,
)
from .data_loader import detect_complex_columns


class NarrativeGenerationMixin:
    """Mixin providing narrative generation, smoothing, and backfill.

    Expects the host class to provide:
        self.base_dir, self.null_mode, self.binary_mode
        self.data_noise_x, self.data_noise_y
        self.query_gpt4o()
        self.generate_sentences_for_entry()
        self.append_hash(), self.extract_hash(), etc.
    """

    @staticmethod
    def _split_into_batches(sentences: List[str], n: int) -> List[List[str]]:
        """Split sentences into n roughly equal batches. Uses min(n, len) if n > len."""
        if not sentences:
            return []
        n = min(n, len(sentences))
        if n <= 1:
            return [sentences]
        batch_size = (len(sentences) + n - 1) // n
        batches = []
        for i in range(0, len(sentences), batch_size):
            batches.append(sentences[i:i + batch_size])
        return batches

    def _parse_numbered_list_and_add_pipes(self, llm_output: str) -> str:
        """Parse numbered list from LLM, extract each sentence, wrap with pipes."""
        lines = llm_output.strip().split('\n')
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if m:
                sentence = m.group(1).strip()
                if sentence:
                    sentences.append(f"| {sentence} |")
        return ' '.join(sentences)

    def _generate_single_narrative_batch(self, sentences: List[str], db_name: str, table_name: str,
                                          expected_transition_count: int = None) -> str:
        """Generate narrative for a single batch of sentences. LLM outputs numbered list; we parse and add pipes."""
        if not sentences:
            return ""

        num_data = len(sentences)
        sentences_text = "\n".join([f"- {sentence}" for sentence in sentences])

        if expected_transition_count is not None and expected_transition_count > 0:
            total_expected = num_data + expected_transition_count
            transition_instruction = f"""4. Add EXACTLY {expected_transition_count} transition sentences (without hashes) between the provided sentences.
   - Transition sentences provide narrative flow and cohesion.
   - Transition sentences must NOT contain any data or factual claims from the provided sentences.
   - The final output must have EXACTLY {total_expected} sentences total ({num_data} data + {expected_transition_count} transitions)."""
        else:
            transition_instruction = """4. You may add transitional sentences between the provided sentences. Transitional sentences have no hash."""

        prompt = f"""You are writing a compelling narrative as a NUMBERED LIST. Output a numbered list of sentences.

CRITICAL REQUIREMENTS:
1. Output ONLY a numbered list. Each line must be: N. (sentence)
2. Include EVERY sentence below EXACTLY as written — no changes, no combining, no splitting, no paraphrasing.
3. For provided sentences, include (Hash: XXXXXXXX) exactly at the end. Do NOT remove, modify, or relocate any (Hash: ...) tag.
{transition_instruction}
5. Make the narrative fit the nature of the data from database '{db_name}' and table '{table_name}', without mentioning a database.
6. DO NOT use information from the given sentences to form transitional sentences to avoid context bleeding.
7. Use blank lines to separate paragraphs if desired.
8. Return ONLY the numbered list — no other text, no explanations.

SENTENCES TO INCLUDE (preserve each one word-for-word including the hash):
{sentences_text}

Output your numbered list now:"""
        system_message = "You output a numbered list of sentences for a narrative. Include every provided sentence exactly as written with its (Hash: ...) tag. Add the specified number of transitional sentences between them. Return ONLY the numbered list."
        llm_output = self.query_gpt4o(prompt, system_message)
        llm_output = llm_output if isinstance(llm_output, str) else ' '.join(llm_output)
        return self._parse_numbered_list_and_add_pipes(llm_output)

    def _generate_narrative_batched(self, sentences: List[str], db_name: str, table_name: str, num_batches: int,
                                     expected_transition_count: int = None) -> str:
        """Generate narrative by splitting into batches. Each batch outputs numbered list; we parse, add pipes, concatenate."""
        batches = self._split_into_batches(sentences, num_batches)
        actual_num_batches = len(batches)

        transition_per_batch = []
        if expected_transition_count is not None and expected_transition_count > 0 and actual_num_batches > 0:
            base = expected_transition_count // actual_num_batches
            remainder = expected_transition_count % actual_num_batches
            for i in range(actual_num_batches):
                if i < actual_num_batches - 1:
                    transition_per_batch.append(base)
                else:
                    transition_per_batch.append(base + remainder)
        else:
            transition_per_batch = [None] * actual_num_batches

        sub_narratives = []
        for i, batch in enumerate(batches):
            sub_narr = self._generate_single_narrative_batch(batch, db_name, table_name, transition_per_batch[i])
            sub_narratives.append(sub_narr.strip())
        return ' '.join(sub_narratives)

    def _smooth_paragraph_transitions(self, narrative_text: str) -> str:
        """Add transitional phrases between paragraphs without modifying paragraph content."""
        prompt = f"""Given this narrative with multiple paragraphs (separated by blank lines), add transitional phrases or short sentences ONLY BETWEEN paragraphs to improve flow. Do NOT modify any content inside a paragraph. Preserve all (Hash: ...) identifiers and | pipe delimiters exactly. Return the full narrative with improved transitions.

NARRATIVE:
{narrative_text}

Return the improved narrative with transitions between paragraphs only."""
        system_message = "You improve narrative flow by adding transitions between paragraphs. You must NOT change any content within paragraphs. Preserve every (Hash: ...) tag and | pipe delimiter exactly."
        try:
            result = self.query_gpt4o(prompt, system_message)
            return result if isinstance(result, str) else ' '.join(result)
        except Exception as e:
            print(f"  Smoothing failed: {e}")
            return narrative_text

    def _smooth_sentence_transitions(self, paragraph_text: str) -> str:
        """Add transitional words/phrases between sentences without adding new sentences or removing words."""
        prompt = f"""Add transitional words or short phrases (e.g. "Furthermore,", "Additionally,") between sentences to improve flow. You may ONLY append or prepend words to existing sentences. Do NOT add new sentences. Do NOT remove any existing words. Preserve every (Hash: ...) tag and | pipe delimiter exactly. Return the improved text.

        TEXT:
        {paragraph_text}"""
        system_message = "You add transitions between sentences. Only append or prepend words. Never add new sentences or remove words. Preserve all (Hash: ...) tags and | pipe delimiters."
        try:
            result = self.query_gpt4o(prompt, system_message)
            return result if isinstance(result, str) else ' '.join(result)
        except Exception as e:
            print(f"  Sentence smoothing failed: {e}")
            return paragraph_text

    def _verify_smoothing_integrity(
        self, original: str, smoothed: str, expected_hashes: List[str]
    ) -> Tuple[bool, str]:
        """
        Verify that smoothing preserved all hashes and sentence counts.
        Returns (is_valid, error_message).
        """
        orig_total, orig_hashed = self.count_sentences_in_narrative(original)
        smooth_total, smooth_hashed = self.count_sentences_in_narrative(smoothed)

        orig_hash_set = set(re.findall(r'\(Hash:\s*([a-f0-9]+)\)', original))
        smooth_hash_set = set(re.findall(r'\(Hash:\s*([a-f0-9]+)\)', smoothed))

        if orig_hash_set != smooth_hash_set:
            missing = orig_hash_set - smooth_hash_set
            added = smooth_hash_set - orig_hash_set
            return False, f"Hash mismatch: missing={missing}, added={added}"

        if len(expected_hashes) != len(smooth_hash_set):
            return False, f"Expected {len(expected_hashes)} hashes, found {len(smooth_hash_set)}"

        if orig_total != smooth_total:
            return False, f"Sentence count changed: {orig_total} -> {smooth_total}"

        if orig_hashed != smooth_hashed:
            return False, f"Hashed sentence count changed: {orig_hashed} -> {smooth_hashed}"

        return True, ""

    def _apply_smoothing_with_verification(
        self, narrative_text: str, expected_hashes: List[str], db_name: str, table_name: str
    ) -> str:
        """
        Apply sentence smoothing and verify integrity. Returns original if verification fails.
        """
        print(f"  Applying sentence smoothing for {db_name}.{table_name}...")
        smoothed = self._smooth_sentence_transitions(narrative_text)

        is_valid, error = self._verify_smoothing_integrity(narrative_text, smoothed, expected_hashes)
        if is_valid:
            print(f"  Smoothing verified: hashes and sentence counts preserved")
            return smoothed
        else:
            print(f"  Smoothing verification failed: {error}")
            print(f"  Reverting to unsmoothed narrative")
            return narrative_text

    def generate_narrative_from_sentences(self, sentences: List[str], db_name: str, table_name: str,
                                           expected_transition_count: int = None, max_retries: int = 3) -> str:
        """
        Generate a narrative weaving the given sentences together.

        If expected_transition_count is 0 or None and self.data_noise_y == 0, returns raw dump (no LLM).
        Otherwise, generates narrative with the specified number of transition sentences.
        """
        filtered_sentences = [s for s in sentences if s != "TBD"]
        expected_hashes = self.extract_all_hashes(filtered_sentences)
        if not filtered_sentences:
            return ""

        if expected_transition_count is None:
            expected_transition_count = compute_expected_transition_count(
                len(filtered_sentences), self.data_noise_x, self.data_noise_y
            )

        if expected_transition_count == 0:
            print(f"  Data:Noise ratio {self.data_noise_x}:0 - skipping LLM narrative generation")
            return '\n\n'.join([f"| {s} |" for s in filtered_sentences])

        def try_stage(narrative_text: str) -> Tuple[Optional[str], List[str]]:
            all_present, missing = self.verify_hashes_in_narrative(narrative_text, expected_hashes)
            if not all_present:
                return (None, missing + ["missing hashes"])
            if not self.verify_delimiters_in_narrative(narrative_text):
                return (None, ["delimiters malformed"])
            valid, err = self.verify_transition_count(narrative_text, expected_hashes, expected_transition_count)
            if not valid:
                return (None, [err])
            return (narrative_text, [])

        print(f"  Generating narrative with {expected_transition_count} transition sentences...")
        narrative_text = self._generate_single_narrative_batch(filtered_sentences, db_name, table_name, expected_transition_count)
        result, missing = try_stage(narrative_text)
        if result is not None:
            smoothed = self._apply_smoothing_with_verification(result, expected_hashes, db_name, table_name)
            final_result, _ = try_stage(smoothed)
            return final_result if final_result is not None else result
        print(f"  Narrative generation attempt 1: full (failed - {missing})")

        for num_batches in [3, 9, 27]:
            print(f"  Narrative generation attempt: {num_batches} batches...")
            narrative_text = self._generate_narrative_batched(filtered_sentences, db_name, table_name, num_batches, expected_transition_count)
            result, errs = try_stage(narrative_text)
            if result is not None:
                smoothed = self._apply_smoothing_with_verification(result, expected_hashes, db_name, table_name)
                final_result, _ = try_stage(smoothed)
                return final_result if final_result is not None else result
            print(f"  Batched narrative failed: {errs}")

        print(f"  Fail-safe: using raw sentence dump + sentence transition smoothing...")
        raw_dump = " ".join([f"| {s} |" for s in filtered_sentences])
        smoothed = self._smooth_sentence_transitions(raw_dump)
        result, _ = try_stage(smoothed)
        if result is not None:
            return result

        print(f"  Final fallback: raw paragraph dump (all hashes and delimiters preserved)")
        return '\n\n'.join([f"| {s} |" for s in filtered_sentences])

    def format_narrative_for_json(self, narrative: str) -> List[str]:
        if not narrative:
            return []

        paragraphs = narrative.split('\n\n')

        formatted_paragraphs = []
        for paragraph in paragraphs:
            cleaned = paragraph.strip().replace('\n', ' ')
            if cleaned:
                formatted_paragraphs.append(cleaned)

        return formatted_paragraphs

    def backfill_template_hashes(self, sentence_template_path: str, narrative_template_path: str = None) -> bool:
        """
        Upgrade an existing template by appending hashes to sentences and regenerating the narrative.

        This method assumes the narrative file exists (if data_noise_y > 0). Callers should use
        generate_narrative_only() first if the narrative doesn't exist for the target ratio.
        """
        print(f"    Loading sentence template: {sentence_template_path}")
        try:
            with open(sentence_template_path, 'r', encoding='utf-8') as f:
                sentence_data = json.load(f)
        except Exception as e:
            print(f"  Error loading sentence template {sentence_template_path}: {e}")
            return False

        generated_sentences = sentence_data.get('generated_sentences', [])
        original_data = sentence_data.get('original_data', {})
        tbd_columns = sentence_data.get('tbd_columns', [])

        if not original_data or not generated_sentences:
            print(f"  Template missing original_data or generated_sentences: {sentence_template_path}")
            return False

        db_name = sentence_data.get('database', '')
        table_name = sentence_data.get('table', '')

        use_x = self.data_noise_x
        use_y = self.data_noise_y

        if not self.needs_backfill(
            sentence_data, self.base_dir, self.null_mode, self.binary_mode,
            use_x, use_y
        ):
            return True

        field_names = list(original_data.keys())
        hash_to_column = {}
        append_count = 0

        for i, column_name in enumerate(field_names):
            if column_name in tbd_columns:
                continue
            if i >= len(generated_sentences):
                continue
            sentence = generated_sentences[i]
            if sentence == "TBD":
                continue
            if self.extract_hash(sentence) is None:
                generated_sentences[i] = self.append_hash(sentence)
                sentence = generated_sentences[i]
                append_count += 1
            h = self.extract_hash(sentence)
            if h:
                hash_to_column[h] = column_name

        print(f"    Appending hashes to {append_count} sentences")

        filtered_sentences = [s for s in generated_sentences if s != "TBD"]
        num_data_sentences = len(filtered_sentences)
        expected_transitions = compute_expected_transition_count(
            num_data_sentences, use_x, use_y
        )
        ratio_str = f"{use_x}:{use_y}"

        sentence_data['generated_sentences'] = generated_sentences
        sentence_data['hash_to_column'] = hash_to_column
        sentence_data['data_noise_ratio'] = ratio_str
        sentence_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"    Saving updated sentence template...")
        try:
            with open(sentence_template_path, 'w', encoding='utf-8') as f:
                json.dump(sentence_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Backfill failed (sentence save): {e}")
            return False

        if use_y == 0:
            txt_file = sentence_template_path.replace('_template.json', '_sentences.txt')
            try:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    for s in filtered_sentences:
                        f.write(f"{s}\n")
                print(f"    Saved raw sentences to: {txt_file}")
            except Exception as e:
                print(f"    Warning: Could not save text dump: {e}")
            print(f"    Backfill succeeded for {db_name}.{table_name} (no narrative needed)")
            return True

        narrative_out_path = narrative_template_path or get_narrative_template_path(
            self.base_dir, self.null_mode, self.binary_mode,
            use_x, use_y, db_name, table_name
        )

        print(f"    Regenerating narrative with {expected_transitions} transitions (ratio {ratio_str})...")
        narrative_text = self.generate_narrative_from_sentences(
            filtered_sentences, db_name, table_name, expected_transitions
        )

        print(f"    Narrative generated, formatting...")
        formatted_narrative = self.format_narrative_for_json(narrative_text)

        narrative_data = {
            'database': db_name,
            'table': table_name,
            'narrative': formatted_narrative,
            'data_noise_ratio': ratio_str,
            'data_sentence_count': num_data_sentences,
            'expected_transition_count': expected_transitions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        Path(os.path.dirname(narrative_out_path)).mkdir(parents=True, exist_ok=True)
        print(f"    Saving narrative template to {narrative_out_path}...")
        try:
            with open(narrative_out_path, 'w', encoding='utf-8') as f:
                json.dump(narrative_data, f, indent=2, ensure_ascii=False)
            print(f"    Backfill succeeded for {db_name}.{table_name}")
            return True
        except Exception as e:
            print(f"  Backfill failed (narrative save): {e}")
            return False

    def generate_narrative_only(self, sentence_template_path: str, data_noise_x: int, data_noise_y: int) -> bool:
        """
        Generate only the narrative for a given data:noise ratio when the sentence template exists
        but the narrative does not. Does NOT regenerate sentences.

        For ratio X:0 (no noise), creates a raw dump narrative with sentences wrapped in pipe delimiters.
        For ratio X:Y where Y>0, generates narrative with LLM-added transitions.
        """
        print(f"    Loading sentence template for narrative generation: {sentence_template_path}")
        try:
            with open(sentence_template_path, 'r', encoding='utf-8') as f:
                sentence_data = json.load(f)
        except Exception as e:
            print(f"  Error loading sentence template {sentence_template_path}: {e}")
            return False

        generated_sentences = sentence_data.get('generated_sentences', [])
        if not generated_sentences:
            print(f"  Template missing generated_sentences: {sentence_template_path}")
            return False

        db_name = sentence_data.get('database', '')
        table_name = sentence_data.get('table', '')

        filtered_sentences = [s for s in generated_sentences if s != "TBD"]
        if not filtered_sentences:
            print(f"  No valid sentences found in template: {sentence_template_path}")
            return False

        if any(self.extract_hash(s) is None for s in filtered_sentences):
            print(f"  Sentences missing hashes, adding hashes first...")
            field_names = list(sentence_data.get('original_data', {}).keys())
            tbd_columns = sentence_data.get('tbd_columns', [])
            hash_to_column = {}
            for i, column_name in enumerate(field_names):
                if column_name in tbd_columns:
                    continue
                if i >= len(generated_sentences):
                    continue
                sentence = generated_sentences[i]
                if sentence == "TBD":
                    continue
                if self.extract_hash(sentence) is None:
                    generated_sentences[i] = self.append_hash(sentence)
                    sentence = generated_sentences[i]
                h = self.extract_hash(sentence)
                if h:
                    hash_to_column[h] = column_name

            sentence_data['generated_sentences'] = generated_sentences
            sentence_data['hash_to_column'] = hash_to_column
            sentence_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open(sentence_template_path, 'w', encoding='utf-8') as f:
                    json.dump(sentence_data, f, indent=2, ensure_ascii=False)
                print(f"    Updated sentence template with hashes")
            except Exception as e:
                print(f"  Error saving sentence template: {e}")
                return False
            filtered_sentences = [s for s in generated_sentences if s != "TBD"]

        num_data_sentences = len(filtered_sentences)
        expected_transitions = compute_expected_transition_count(num_data_sentences, data_noise_x, data_noise_y)
        ratio_str = f"{data_noise_x}:{data_noise_y}"

        narrative_out_path = get_narrative_template_path(
            self.base_dir, self.null_mode, self.binary_mode,
            data_noise_x, data_noise_y, db_name, table_name
        )

        if os.path.isfile(narrative_out_path):
            print(f"    Narrative already exists at {narrative_out_path}, skipping")
            return True

        if data_noise_y == 0:
            print(f"    Ratio {ratio_str}: Creating narrative with no noise")
            raw_narrative = ' '.join([f"| {s} |" for s in filtered_sentences])
            formatted_narrative = self.format_narrative_for_json(raw_narrative)
        else:
            _save_dx, _save_dy = self.data_noise_x, self.data_noise_y
            self.data_noise_x, self.data_noise_y = data_noise_x, data_noise_y
            try:
                print(f"    Generating narrative with {expected_transitions} transitions (ratio {ratio_str})...")
                narrative_text = self.generate_narrative_from_sentences(
                    filtered_sentences, db_name, table_name, expected_transitions
                )
            finally:
                self.data_noise_x, self.data_noise_y = _save_dx, _save_dy
            print(f"    Narrative generated, formatting...")
            formatted_narrative = self.format_narrative_for_json(narrative_text)

        narrative_data = {
            'database': db_name,
            'table': table_name,
            'narrative': formatted_narrative,
            'data_noise_ratio': ratio_str,
            'data_sentence_count': num_data_sentences,
            'expected_transition_count': expected_transitions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        Path(os.path.dirname(narrative_out_path)).mkdir(parents=True, exist_ok=True)
        print(f"    Saving narrative template to {narrative_out_path}...")
        try:
            with open(narrative_out_path, 'w', encoding='utf-8') as f:
                json.dump(narrative_data, f, indent=2, ensure_ascii=False)
            print(f"    Narrative generation succeeded for {db_name}.{table_name}")
            return True
        except Exception as e:
            print(f"  Narrative generation failed (save): {e}")
            return False

    def backfill_all_templates(self, null_mode: str = None, binary_mode: str = None,
                                data_noise_x: int = None, data_noise_y: int = None) -> Dict[str, Any]:
        """Backfill hashes for all existing sentence templates in the output directory."""
        if null_mode is not None:
            self.null_mode = null_mode
        if binary_mode is not None:
            self.binary_mode = binary_mode
        if data_noise_x is not None:
            self.data_noise_x = data_noise_x
        if data_noise_y is not None:
            self.data_noise_y = data_noise_y

        sentence_dir = get_sentence_templates_dir(self.base_dir, self.null_mode, self.binary_mode)

        if not os.path.isdir(sentence_dir):
            print(f"Sentence templates directory not found: {sentence_dir}")
            return {
                'success': False,
                'backfilled_count': 0,
                'skipped_count': 0,
                'failed_count': 0,
                'details': {}
            }

        template_files = list(Path(sentence_dir).rglob("*_template.json"))
        print(f"Processing sentence templates in {sentence_dir}")
        print(f"Found {len(template_files)} sentence template files")
        print(f"Target ratio: {self.data_noise_x}:{self.data_noise_y}")

        backfilled_count = 0
        generated_count = 0
        skipped_count = 0
        failed_count = 0
        details = {}

        for template_path in template_files:
            path_str = str(template_path)
            try:
                with open(path_str, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)

                db_name = template_data.get('database', '')
                table_name = template_data.get('table', '')
                rel_path = os.path.relpath(path_str, sentence_dir)

                narrative_path_for_ratio = get_narrative_template_path(
                    self.base_dir, self.null_mode, self.binary_mode,
                    self.data_noise_x, self.data_noise_y, db_name, table_name
                )

                narrative_exists = os.path.isfile(narrative_path_for_ratio)

                if not narrative_exists:
                    print(f"\n  {db_name}.{table_name}: Narrative missing for ratio {self.data_noise_x}:{self.data_noise_y}, generating...")
                    success = self.generate_narrative_only(path_str, self.data_noise_x, self.data_noise_y)
                    if success:
                        generated_count += 1
                        details[rel_path] = 'narrative_generated'
                    else:
                        failed_count += 1
                        details[rel_path] = 'narrative_generation_failed'
                    continue

                if not self.needs_backfill(
                    template_data, self.base_dir, self.null_mode, self.binary_mode,
                    self.data_noise_x, self.data_noise_y
                ):
                    skipped_count += 1
                    details[rel_path] = 'skipped'
                    continue

                print(f"\n  {db_name}.{table_name}: Backfilling hashes...")
                success = self.backfill_template_hashes(path_str, narrative_path_for_ratio)
                if success:
                    backfilled_count += 1
                    details[rel_path] = 'backfilled'
                else:
                    failed_count += 1
                    details[rel_path] = 'failed'
            except Exception as e:
                failed_count += 1
                rel_path = os.path.relpath(path_str, sentence_dir) if os.path.exists(path_str) else path_str
                details[rel_path] = f'error: {str(e)}'
                print(f"  Error processing {path_str}: {e}")

        print(f"\nComplete: Generated={generated_count}, Backfilled={backfilled_count}, Skipped={skipped_count}, Failed={failed_count}")
        return {
            'success': failed_count == 0,
            'generated_count': generated_count,
            'backfilled_count': backfilled_count,
            'skipped_count': skipped_count,
            'failed_count': failed_count,
            'details': details
        }

    def backfill_templates_for_tables(
        self,
        selected_tables: List[Tuple[str, str]],
        null_mode: str = None,
        binary_mode: str = None,
        data_noise_x: int = None,
        data_noise_y: int = None
    ) -> Dict[str, Any]:
        """Check and backfill hashes for templates of the given tables. Used when run from document_generation."""
        if null_mode is not None:
            self.null_mode = null_mode
        if binary_mode is not None:
            self.binary_mode = binary_mode
        if data_noise_x is not None:
            self.data_noise_x = data_noise_x
        if data_noise_y is not None:
            self.data_noise_y = data_noise_y

        backfilled_count = 0
        generated_count = 0
        skipped_count = 0
        failed_count = 0
        details = {}

        for db_name, table_name in selected_tables:
            sentence_path = get_sentence_template_path(self.base_dir, self.null_mode, self.binary_mode, db_name, table_name)

            if not os.path.isfile(sentence_path):
                details[f"{db_name}.{table_name}"] = 'not_found'
                continue
            try:
                with open(sentence_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)

                narrative_path_for_ratio = get_narrative_template_path(
                    self.base_dir, self.null_mode, self.binary_mode,
                    self.data_noise_x, self.data_noise_y, db_name, table_name
                )

                narrative_exists = os.path.isfile(narrative_path_for_ratio)

                if not narrative_exists:
                    success = self.generate_narrative_only(sentence_path, self.data_noise_x, self.data_noise_y)
                    if success:
                        generated_count += 1
                        details[f"{db_name}.{table_name}"] = 'narrative_generated'
                    else:
                        failed_count += 1
                        details[f"{db_name}.{table_name}"] = 'narrative_generation_failed'
                    continue

                if not self.needs_backfill(
                    template_data, self.base_dir, self.null_mode, self.binary_mode,
                    self.data_noise_x, self.data_noise_y
                ):
                    skipped_count += 1
                    details[f"{db_name}.{table_name}"] = 'skipped'
                    continue

                success = self.backfill_template_hashes(sentence_path, narrative_path_for_ratio)
                if success:
                    backfilled_count += 1
                    details[f"{db_name}.{table_name}"] = 'backfilled'
                else:
                    failed_count += 1
                    details[f"{db_name}.{table_name}"] = 'failed'
            except Exception as e:
                failed_count += 1
                details[f"{db_name}.{table_name}"] = f'error: {str(e)}'
                print(f"  Error processing {db_name}.{table_name}: {e}")

        if generated_count > 0 or backfilled_count > 0 or failed_count > 0:
            print(f"  Generated={generated_count}, Backfilled={backfilled_count}, Skipped={skipped_count}, Failed={failed_count}")
        return {
            'success': failed_count == 0,
            'generated_count': generated_count,
            'backfilled_count': backfilled_count,
            'skipped_count': skipped_count,
            'failed_count': failed_count,
            'details': details
        }
