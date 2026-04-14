"""Variation bank generation for document template pipelines.

Provides the VariationBankGenerator class which creates structurally diverse
sentence variations, lexical synonym sets, and null/binary-aware rewrites
via a local LLM backend.  Also exposes the _ensure_nltk_wordnet helper
used to lazily bootstrap the WordNet corpus.
"""

import json
import openai
import random
import re
import time
import sys
import os
import string
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from utils.cost_tracker import get_cost_tracker, track_openai_response, track_local_llm_operation

from .config import create_local_llm_openai_client, get_local_llm_model

_wordnet_corpus_ready = False


def _ensure_nltk_wordnet() -> None:
    """Ensure WordNet (+ OMW) are available; run at most once per process."""
    global _wordnet_corpus_ready
    if _wordnet_corpus_ready:
        return
    for resource in ("corpora/wordnet", "corpora/wordnet.zip"):
        try:
            nltk.data.find(resource)
            _wordnet_corpus_ready = True
            return
        except LookupError:
            continue
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _wordnet_corpus_ready = True


class VariationBankGenerator:

    def __init__(self, num_variations: int = 15, null_mode: str = "implicit", binary_mode: str = "implicit"):
        self.client = openai.OpenAI()
        self.local_llm_client = create_local_llm_openai_client()
        self.num_variations = num_variations
        self.max_retries = 3
        self.null_mode = null_mode
        self.binary_mode = binary_mode

    def query_local_llm(self, prompt: str, system_message: str) -> str:
        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=16000,
            )
            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('structural_variation')
            if response and response.choices:
                message = response.choices[0].message
                content = message.content
                if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                    content = message.reasoning_content
                return content if content else ""
            return ""
        except Exception as e:
            print(f"Error querying local LLM: {str(e)}")
            return ""

    def count_placeholders(self, text: str) -> int:
        return len(re.findall(r'\[[^\]]+\]', text))

    def validate_variation(self, variation: str, expected_placeholders: int, field_values: List[str] = None) -> bool:
        actual_placeholders = self.count_placeholders(variation)
        if actual_placeholders != expected_placeholders:
            return False
        if field_values:
            for value in field_values:
                if value.lower() in variation.lower() and value not in ["not specified", "unspecified"]:
                    return False
        return True

    def validate_counter_variation(self, variation: str, expected_placeholders: int, primary_field_name: str) -> bool:
        actual_placeholders = self.count_placeholders(variation)
        if actual_placeholders != expected_placeholders:
            return False
        primary_placeholder = f"[{primary_field_name.upper()}]"
        if primary_placeholder in variation:
            return False
        return True

    def context_bleed_preventer(self, sentence: str, descriptors: str, values: str) -> str:
        """Detect and correct context bleeding: when literal data values are replaced by expansions/interpretations (e.g. EUR → European), making parsing difficult."""
        prompt = f"""The following sentence may contain embedded data values from a synthetically generated narrative. Your task is to detect "context bleeding" and, if present, output a corrected sentence.

CONTEXT BLEEDING occurs when the actual data value (the literal string that would appear in the database or column) is replaced or paraphrased in the sentence by an unabbreviated form, interpretation, or expansion. The reader sees the meaning (e.g. "European") but not the parseable value (e.g. "EUR"), which makes extraction and parsing difficult.
CONTEXT BLEEDING can also occur in transitional sentences which don't directly make reference to the data value implied but might reveal some specification about that value. If the descriptors include locations, dates, times, or other finite value types, the narrative is subject to more context bleeding. These sentences which imply data of this type must be analyzed to ensure that the content in reference to this data is removed and the sentence is better generalized for multiple contexts. 

DESCRIPTORS (column names and expected data types/values):
{descriptors}

RULES:
- Identify any word or phrase in the sentence that implicitly or explicitly stands in for a descriptor's data value but is NOT the literal value itself (e.g. expansions, demonyms, interpretations).
- If context bleeding is found, output a corrected sentence that preserves the exact literal data value(s) while keeping the sentence natural. Do not remove or add non-data content; only replace the "bled" wording with the actual value. If a data value is implied with no mention to the column name from the descriptors, remove the bled word entirely.
- If no context bleeding is found, output the sentence unchanged.

FEW-SHOT EXAMPLES:

Example 1 — Descriptor: Currency column (ISO currency codes, e.g. EUR, USD, GBP).
Before (context bleed): According to the Currency column, the transaction today was executed in EUR, revealing European customer spending dynamics.
After (corrected): According to the Currency column, the transaction today was executed in EUR, revealing EUR-region customer spending dynamics.
(Explanation: "European" unabbreviates EUR and bleeds context; the literal value EUR must remain parseable.)

Example 2 — Descriptor: Country column (ISO country codes, e.g. USA, DEU, JPN).
Before (context bleed): The Country field shows USA, reflecting American market preferences.
After (corrected): The Country field shows USA, reflecting USA market preferences.
(Explanation: "American" replaces the literal value USA.)

Example 3 — Descriptor: Gender column (codes M, F, or similar).
Before (context bleed): The record indicates M, consistent with male respondents in the cohort.
After (corrected): The record indicates M, consistent with M respondents in the cohort.
(Explanation: "male" expands M and obscures the actual stored value.)

Example 4 — No bleed.
Sentence: The Status column contains PENDING and the Amount is 150.00.
Output: The Status column contains PENDING and the Amount is 150.00.
(No change; literal values are already present.)

TASK:
Sentence to check: {sentence}
Values the narrative was generated on: {values}

Output only the corrected sentence (or the original sentence if no context bleeding is detected). Do not include explanation or "Before/After" labels."""
        system_message = "You are an expert at detecting context bleeding in data-derived text. You identify when literal data values have been replaced by expansions or interpretations and output a corrected sentence that preserves the exact, parseable data values."
        response = self.query_local_llm(prompt, system_message)
        return response.strip() if response else sentence

    def scrub_static_sentence_data_bleed(self, sentence: str, detected_items: List[Dict[str, str]]) -> str:
        """Remove leaked field names and data values from a static (no-hash) sentence.

        Unlike context_bleed_preventer (which preserves literal data values),
        this method instructs the LLM to *remove* them entirely so the sentence
        reads as a generic transitional or contextual statement with no data
        specificity.
        """
        items_desc = "\n".join(
            f"- field_name=\"{item['field_name']}\", field_value=\"{item['field_value']}\", "
            f"detected_as=\"{item['detected_as']}\""
            for item in detected_items
        )
        prompt = f"""The following sentence is a TRANSITIONAL / CONTEXTUAL sentence in a narrative 
generated from structured data. It is NOT supposed to contain any data-bearing field 
names or field values, but some have leaked in.

SENTENCE:
{sentence}

DETECTED DATA LEAKS:
{items_desc}

TASK:
Rewrite the sentence so that every detected field name, field value, or derived 
reference to them is removed. The rewritten sentence must:
1. Read as natural, well-formed English prose.
2. Preserve any non-data transitional or contextual phrasing.
3. NOT mention any column name, field name, or data value (even paraphrased).
4. Keep roughly the same length — if removing a phrase would collapse the 
   sentence, replace it with a generic filler that maintains flow 
   (e.g. "this metric", "the relevant figure", "certain factors").
5. End with proper punctuation.
6. Return ONLY the rewritten sentence — no explanation, no labels.

Rewritten sentence:"""
        system_message = (
            "You rewrite sentences to remove all data-specific content (field names and values) "
            "while preserving natural sentence flow. Return only the cleaned sentence."
        )
        response = self.query_local_llm(prompt, system_message)
        if response:
            cleaned = response.strip()
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
            if cleaned and len(cleaned) > 10:
                return cleaned
        return sentence

    def generate_replacement_noise_sentence(
        self,
        prev_sentence: str,
        next_sentence: str,
        database: str = "",
        table: str = "",
    ) -> str:
        """Generate a full transitional/contextual noise sentence to replace a
        degenerate fragment (e.g. a bare transition word like 'Additionally,').

        The replacement should be a complete, natural-sounding sentence that
        bridges the two surrounding sentences without containing any
        data-specific field names or values.
        """
        prompt = f"""You are writing a narrative document about a "{database}" database, 
specifically the "{table}" table.

Between the two sentences below there should be a TRANSITIONAL or CONTEXTUAL 
sentence that connects them naturally. Write exactly ONE complete sentence 
(subject + verb + object/complement) that fits between them.

PREVIOUS SENTENCE:
{prev_sentence}

NEXT SENTENCE:
{next_sentence}

REQUIREMENTS:
1. The sentence must be a full, grammatically correct English sentence (not a 
   fragment or a lone transition word).
2. It should bridge the ideas of the previous and next sentences naturally.
3. Do NOT include any specific data field names, column names, values, or 
   numbers from the database — keep it generic and contextual.
4. Aim for 10-25 words.
5. End with proper punctuation.
6. Return ONLY the sentence — no labels, no quotes, no explanation.

Your sentence:"""

        system_message = (
            "You write natural transitional sentences for data narrative documents. "
            "Return only the single replacement sentence."
        )
        response = self.query_local_llm(prompt, system_message)
        if response:
            cleaned = response.strip().strip('"').strip("'")
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
            words = re.findall(r'\b[a-zA-Z]+\b', cleaned)
            if cleaned and len(words) >= 4:
                if not cleaned.endswith(('.', '!', '?')):
                    cleaned += '.'
                return cleaned
        return ""

    def confirm_partial_value_bleed(
        self,
        sentence: str,
        field_name: str,
        field_value: str,
        database: str = "",
        table: str = "",
    ) -> str:
        """Ask the local LLM whether a partial match of *field_value* in *sentence*
        constitutes a real data leak in the context of the database being generated.

        Returns the offending word/phrase that should be removed if the LLM
        confirms a leak, or an empty string if there is no leak.
        """
        prompt = f"""A transitional sentence in a narrative generated from the database 
"{database}" (table "{table}") may contain a PARTIAL reference to a data value 
that should NOT appear in this sentence.

SENTENCE:
{sentence}

FIELD NAME: {field_name}
FIELD VALUE: {field_value}

TASK:
Determine whether any word or phrase in the sentence is clearly derived from, 
or closely references, an important part of the field value above 
(e.g. a location name, a date component, a person name, an institution, a 
code segment, or any other semantically significant fragment of the value).

Single common English words that happen to overlap with a short field value 
(e.g. the letter "D" appearing inside "education") are NOT data leaks.

If a genuine data leak exists, respond with ONLY the exact word or phrase 
from the sentence that needs to be removed — nothing else.
If there is no genuine data leak, respond with exactly: No

Examples:
- field_value="Alameda County", sentence contains "Alameda" → respond: Alameda
- field_value="D", sentence contains "education" → respond: No
- field_value="2014-2015", sentence contains "2014" → respond: 2014
- field_value="Springfield Elementary", sentence contains "Springfield" → respond: Springfield
- field_value="1", sentence contains "providing" → respond: No

Your answer:"""
        system_message = (
            "You determine if a data value has partially leaked into a sentence. "
            "Return ONLY the offending word/phrase, or 'No'. No explanation."
        )
        response = self.query_local_llm(prompt, system_message)
        if not response:
            return ""
        answer = response.strip().strip('"').strip("'")
        if answer.upper() == "NO" or len(answer) < 1:
            return ""
        if answer.lower() in sentence.lower():
            return answer
        return ""

    def generate_structural_variations_standard(self, sentence: str, field_name: str, context: str = "", silent: bool = False) -> List[str]:
        n = self.num_variations
        context_prompt = f"Context: {context}" if context else "Context: This is part of a professional data report or document."
        natural_field_name = field_name.replace("_", " ")

        mode_guidance = ""
        if self.binary_mode == "explicit":
            mode_guidance += "\n13. For binary values (0/1), preserve the LITERAL value (e.g., '0', '1') — do NOT convert to yes/no or true/false"
        else:
            mode_guidance += "\n13. For binary values, use natural language forms (e.g., 'is a charter school', 'is not a charter school') — do NOT use literal '0' or '1'"

        if self.null_mode == "explicit":
            mode_guidance += "\n14. For null/missing values, preserve the literal word 'NULL' — do NOT convert to 'not specified' or 'not available'"
        else:
            mode_guidance += "\n14. For null/missing values, use natural language (e.g., 'not specified', 'not available', 'unknown') — do NOT use literal 'NULL'"

        prompt = f"""Generate {n} structurally diverse variations of this sentence. Each variation must look visually different while preserving the exact meaning and all data values.

Original sentence: "{sentence}"

{context_prompt}

CRITICAL REQUIREMENTS:
1. Keep the field name "{natural_field_name}" in its NATURAL form with spaces - NEVER convert to underscores like "{field_name}"
2. **ABSOLUTE VALUE PRESERVATION**: Every data value in the original sentence MUST appear EXACTLY and LITERALLY in each variation
3. NEVER substitute, paraphrase, abbreviate, expand, round, or modify ANY data value in ANY way
4. This includes but is not limited to: IDs, codes, numbers, names, dates, percentages, identifiers, serial numbers, reference codes
5. Example: If the sentence contains "ID 12345", every variation MUST contain exactly "12345" — NEVER "twelve thousand" or similar
6. Example: If the sentence contains "Code ABC-123", every variation MUST contain exactly "ABC-123" — NEVER rephrase it
7. Each variation must use different word choices and sentence structures for NON-DATA words only
8. Use synonyms ONLY for non-data words to create visual diversity
9. Vary between active and passive voice
10. Restructure clauses and phrases differently in each variation
11. Return only the {n} variations, one per line
12. No numbering or explanations{mode_guidance}

DIVERSIFICATION TECHNIQUES (apply ONLY to non-data words):
- Rearrange clause order (but keep data values intact)
- Use different verbs with similar meaning
- Vary adjectives and adverbs
- Change prepositional phrases
- Alternate between formal and semi-formal tone
- Use different sentence openings

Variations:"""

        system_message = "You are an expert at creating structurally diverse sentence variations while preserving semantic meaning. Focus on making each variation look distinctly different through word choice, structure, and phrasing while keeping all data values intact. Always keep field names in natural readable form with spaces, never use underscores."

        variations = []
        if not silent:
            print(f"      Generating variations one by one...")
        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                lines = response.strip().split('\n')
                for idx, line in enumerate(lines):
                    clean_line = line.strip()
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                    if clean_line and len(clean_line) > 10:
                        variations.append(clean_line)
                        if not silent:
                            print(f"        [{len(variations)}/{n}] {clean_line[:80]}{'...' if len(clean_line) > 80 else ''}")
            if len(variations) >= n:
                break

        return variations[:n]

    def generate_structural_variations_static(self, sentence: str, context: str = "", silent: bool = False) -> List[str]:
        """Generate structurally diverse variations for sentences without explicit data fields.
        
        Focus on word and structure diversity while preserving the core semantic meaning.
        """
        n = self.num_variations
        context_prompt = f"Context: {context}" if context else "Context: This is part of a professional data report or document."

        mode_guidance = ""
        if self.binary_mode == "explicit":
            mode_guidance += "\n13. For any binary values (0/1), preserve the LITERAL value — do NOT convert to yes/no or true/false"
        else:
            mode_guidance += "\n13. For any binary values, use natural language forms — do NOT use literal '0' or '1'"

        if self.null_mode == "explicit":
            mode_guidance += "\n14. For any null/missing values, preserve the literal word 'NULL' — do NOT convert to 'not specified'"
        else:
            mode_guidance += "\n14. For any null/missing values, use natural language (e.g., 'not specified', 'not available') — do NOT use literal 'NULL'"

        prompt = f"""Generate {n} structurally diverse variations of this sentence. Each variation must look visually different while preserving the same core meaning.

Original sentence: "{sentence}"

{context_prompt}

CRITICAL REQUIREMENTS:
1. Preserve the original factual content and meaning of the sentence.
2. Do NOT add new facts or remove existing facts.
3. **ABSOLUTE VALUE PRESERVATION**: If ANY data values appear in the sentence (IDs, codes, numbers, names, dates, etc.), they MUST appear EXACTLY and LITERALLY in each variation.
4. NEVER substitute, paraphrase, abbreviate, expand, round, or modify ANY data value.
5. Example: If the sentence contains "ID 12345", every variation MUST contain exactly "12345" — NEVER "twelve thousand".
6. Each variation must use different word choices and sentence structures for NON-DATA words only.
7. Use synonyms ONLY for non-critical, non-data words to create visual diversity.
8. Vary between active and passive voice.
9. Restructure clauses and phrases differently in each variation.
10. Use different transitional words and connectors.
11. Return only the {n} variations, one per line.
12. No numbering or explanations.{mode_guidance}

DIVERSIFICATION TECHNIQUES (apply ONLY to non-data words):
- Rearrange clause order (but keep data values intact).
- Use different verbs with similar meaning.
- Vary adjectives and adverbs.
- Change prepositional phrases.
- Alternate between formal and semi-formal tone.
- Use different sentence openings.

Variations:"""

        system_message = "You are an expert at creating structurally diverse sentence variations while preserving semantic meaning. Focus on word choice and syntactic diversity while keeping the underlying message intact."

        variations: List[str] = []
        if not silent:
            print(f"      Generating static sentence variations...")
        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                lines = response.strip().split('\n')
                for idx, line in enumerate(lines):
                    clean_line = line.strip()
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                    if clean_line and len(clean_line) > 10:
                        variations.append(clean_line)
                        if not silent:
                            print(f"        [static {len(variations)}/{n}] {clean_line[:80]}{'...' if len(clean_line) > 80 else ''}")
            if len(variations) >= n:
                break

        return variations[:n]

    def generate_structural_variations_standard_with_style(self, sentence: str, field_name: str, context: str = "", original_sentence: str = "", silent: bool = False) -> List[str]:
        n = self.num_variations
        context_prompt = f"Context: {context}" if context else "Context: This is part of a professional data report or document."
        original_style_reference = f"\nStyle reference: \"{original_sentence}\"" if original_sentence else ""
        natural_field_name = field_name.replace("_", " ")

        mode_guidance = ""
        if self.binary_mode == "explicit":
            mode_guidance += "\n12. For binary values (0/1), preserve the LITERAL value (e.g., '0', '1') — do NOT convert to yes/no or true/false"
        else:
            mode_guidance += "\n12. For binary values, use natural language forms (e.g., 'is a charter school', 'is not a charter school') — do NOT use literal '0' or '1'"

        if self.null_mode == "explicit":
            mode_guidance += "\n13. For null/missing values, preserve the literal word 'NULL' — do NOT convert to 'not specified' or 'not available'"
        else:
            mode_guidance += "\n13. For null/missing values, use natural language (e.g., 'not specified', 'not available', 'unknown') — do NOT use literal 'NULL'"

        prompt = f"""Generate {n} structurally diverse variations of this sentence. Each variation must look visually different while preserving the exact meaning and all data values.{original_style_reference}

Original sentence: "{sentence}"

{context_prompt}

CRITICAL REQUIREMENTS:
1. Keep the field name "{natural_field_name}" in its NATURAL form with spaces - NEVER convert to underscores like "{field_name}"
2. **ABSOLUTE VALUE PRESERVATION**: Every data value in the original sentence MUST appear EXACTLY and LITERALLY in each variation
3. NEVER substitute, paraphrase, abbreviate, expand, round, or modify ANY data value in ANY way
4. This includes but is not limited to: IDs, codes, numbers, names, dates, percentages, identifiers, serial numbers, reference codes
5. Example: If the sentence contains "ID 12345", every variation MUST contain exactly "12345" — NEVER "twelve thousand" or similar
6. Example: If the sentence contains "Code ABC-123", every variation MUST contain exactly "ABC-123" — NEVER rephrase it
7. Each variation must use different word choices and sentence structures for NON-DATA words only
8. Use synonyms ONLY for non-data words to create visual diversity
9. Vary between active and passive voice
10. Match the tone and formality of the style reference if provided
11. Return only the {n} variations, one per line. No numbering or explanations{mode_guidance}

DIVERSIFICATION TECHNIQUES (apply ONLY to non-data words):
- Rearrange clause order (but keep data values intact)
- Use different verbs with similar meaning
- Vary adjectives and adverbs
- Change prepositional phrases
- Use different sentence openings

Variations:"""

        system_message = "You are an expert at creating structurally diverse sentence variations while preserving semantic meaning. Focus on making each variation look distinctly different through word choice, structure, and phrasing while keeping all data values intact. Always keep field names in natural readable form with spaces, never use underscores."

        variations = []
        if not silent:
            print(f"      Generating variations one by one...")
        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                lines = response.strip().split('\n')
                for idx, line in enumerate(lines):
                    clean_line = line.strip()
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                    if clean_line and len(clean_line) > 10:
                        variations.append(clean_line)
                        if not silent:
                            print(f"        [{len(variations)}/{n}] {clean_line[:80]}{'...' if len(clean_line) > 80 else ''}")
            if len(variations) >= n:
                break

        return variations[:n]

    def generate_sentence_for_field_value(self, field_name: str, field_value: str, context: str = "", natural_mode: bool = False, original_sentence: str = "") -> str:
        context_prompt = f"Context: {context}" if context else "Context: This is part of a professional data report or document."
        natural_field_name = field_name.replace("_", " ")

        is_binary_value = field_value in ["0", "1"]
        is_null_value = field_value.upper() in ["NULL", "NONE", "NOT SPECIFIED", "UNSPECIFIED", "NOT AVAILABLE", "UNAVAILABLE", "UNKNOWN"]

        binary_guidance = ""
        value_rule = f'2. **CRITICAL**: The EXACT value "{field_value}" MUST appear LITERALLY in the sentence — NO substitutions, NO paraphrasing, NO abbreviations, NO expansions'
        value_emphasis = f'3. The value "{field_value}" must be included CHARACTER-FOR-CHARACTER as it appears — this includes IDs, codes, numbers, identifiers'

        if is_binary_value:
            if self.binary_mode == "explicit":
                binary_guidance = "4. For this binary value, use the LITERAL value (e.g., 'the value is 0' or 'the value is 1') — do NOT convert to yes/no or true/false"
            else:
                if field_value == "1":
                    binary_guidance = "4. For this binary value (1 = yes/true), express it using NATURAL LANGUAGE (e.g., 'is a charter school', 'operates as', 'qualifies as') — do NOT use the literal '1'"
                    value_rule = "2. Express the affirmative/positive state of this binary field using natural language — do NOT include the literal '1'"
                    value_emphasis = "3. Use natural language to convey the 'yes/true' meaning of this binary field"
                else:
                    binary_guidance = "4. For this binary value (0 = no/false), express it using NATURAL LANGUAGE (e.g., 'is not a charter school', 'does not operate as', 'does not qualify as') — do NOT use the literal '0'"
                    value_rule = "2. Express the negative state of this binary field using natural language — do NOT include the literal '0'"
                    value_emphasis = "3. Use natural language to convey the 'no/false' meaning of this binary field"
        else:
            if self.binary_mode == "explicit":
                binary_guidance = "4. For any binary values (0/1), use the LITERAL value — do NOT convert to yes/no or true/false"
            else:
                binary_guidance = "4. For any binary values (0/1), use natural language (e.g., 'is a charter school' or 'is not a charter school')"

        null_guidance = ""
        if is_null_value:
            if self.null_mode == "explicit":
                null_guidance = "\n10. For this null/missing value, use the literal word 'NULL' — do NOT use 'not specified' or 'not available'"
                value_rule = "2. Use the literal word 'NULL' for this missing value"
                value_emphasis = "3. The word 'NULL' must appear exactly as written"
            else:
                null_guidance = "\n10. For this null/missing value, use natural language (e.g., 'not specified', 'not available', 'unknown') — do NOT use literal 'NULL'"
                value_emphasis = "3. Use natural language to indicate the value is missing/unspecified"
        else:
            if self.null_mode == "explicit":
                null_guidance = "\n10. For null/missing values, use the literal word 'NULL' — do NOT use 'not specified' or 'not available'"
            else:
                null_guidance = "\n10. For null/missing values, use natural language (e.g., 'not specified', 'not available', 'unknown')"

        style_ref = f'\nStyle reference (match this complexity and length): "{original_sentence}"' if original_sentence else ""
        prompt = f"""Generate a single cohesive sentence for the field "{natural_field_name}" with value {field_value}.

Context: {context_prompt}{style_ref}

RULES:
1. Include the field name "{natural_field_name}" in NATURAL form with spaces - NEVER use underscores
{value_rule}
{value_emphasis}
{binary_guidance}
5. Do not explain what the value means
6. Do not mention databases or records
7. Return only one sentence of 15-25 words
8. Use natural, professional language
9. The sentence MUST match the semantic complexity and reading difficulty of the style reference if provided — use rich vocabulary, varied clause structure, and professional phrasing rather than simple declarative statements{null_guidance}

IMPORTANT: For non-binary, non-null values like IDs, codes, and identifiers, the EXACT value "{field_value}" MUST appear in the output sentence. Do NOT convert numbers to words, do NOT abbreviate, do NOT expand codes.

Sentence:"""

        system_message = "You are a sentence generator that creates natural, professional sentences incorporating field names and values. Sentences must be semantically rich and structurally complex — never trivially short or simplistic. Always use natural field names with spaces, never underscores."

        print(f"      Generating sentence for {natural_field_name}...")
        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                result = response.strip()
                result = re.sub(r'^\d+[\.\)]\s*', '', result)
                result = re.sub(r'^[-•*]\s*', '', result)
                if result and len(result) > 10:
                    print(f"        Generated: {result[:80]}{'...' if len(result) > 80 else ''}")
                    return result

        return f"The {natural_field_name} value is {field_value}."

    def generate_null_variations_non_null(self, sentence: str, field_name: str, field_value: str, context: str = "", null_replacement_phrase: str = None) -> List[str]:
        if null_replacement_phrase is None:
            null_replacement_phrase = "NULL" if self.null_mode == "explicit" else "not specified"
        print(f"      Generating null variations for non-null field: {field_name} (encoding: '{null_replacement_phrase}')")
        variation_sentence = self.generate_sentence_for_field_value(field_name, null_replacement_phrase, context, natural_mode=False, original_sentence=sentence)
        print(f"      Generated variation sentence: {variation_sentence[:100]}...")

        result = self.generate_null_variations_null(variation_sentence, field_name, null_replacement_phrase, context)
        print(f"      Generated {len(result)} null variations")
        return result

    def generate_nonnull_variations_non_null(self, sentence: str, field_name: str, field_value: str, context: str = "") -> List[str]:
        print(f"      Generating non-null variations for field: {field_name}")
        result = self.generate_structural_variations_standard(sentence, field_name, context)
        print(f"      Generated {len(result)} non-null variations")
        return result

    def generate_null_variations_null(self, sentence: str, field_name: str, original_replacement_value: str, context: str = "", silent: bool = False) -> List[str]:
        n = self.num_variations
        context_prompt = f"Context: {context}" if context else "Context: This is part of a professional data report or document."
        natural_field_name = field_name.replace("_", " ")

        if self.null_mode == "explicit":
            null_guidance = """2. Every variation MUST use the literal word "NULL" for the missing value
3. Do NOT substitute with "not specified", "unspecified", "missing", "absent", "unavailable", or any other natural language term"""
        else:
            null_guidance = f"""2. Every variation MUST contain the exact phrase "{original_replacement_value}" or a similar natural language expression
3. You MAY use synonyms like "unspecified", "not available", "unknown" — but keep them consistent within each variation"""

        prompt = f"""Generate {n} structurally diverse variations of this sentence. Each variation must look visually different while preserving the null/missing value indication.

Original sentence: "{sentence}"

{context_prompt}

CRITICAL REQUIREMENTS:
1. Keep field name "{natural_field_name}" in NATURAL form with spaces - NEVER use underscores
{null_guidance}
4. **ABSOLUTE VALUE PRESERVATION**: ALL other data values (IDs, codes, numbers, names, dates, etc.) MUST appear EXACTLY and LITERALLY in each variation
5. NEVER substitute, paraphrase, abbreviate, expand, round, or modify ANY non-null data value
6. Example: If the sentence contains "ID 12345", every variation MUST contain exactly "12345"
7. Each variation must use different word choices and structures for NON-DATA words only
8. Vary sentence structure and clause order
9. Return only the {n} variations, one per line
10. No numbering or explanations
11. Each variation MUST match the semantic complexity and reading difficulty of standard data-bearing variations — use rich vocabulary, varied clause structures, subordinate clauses, and professional phrasing. Do NOT produce trivially short or simplistic sentences

DIVERSIFICATION TECHNIQUES (apply ONLY to non-data words):
- Rearrange clause order (but keep data values intact)
- Use different verbs with similar meaning
- Vary adjectives and adverbs
- Change prepositional phrases
- Alternate between formal and semi-formal tone
- Use different sentence openings

Variations:"""

        system_message = "You are an expert at creating structurally diverse sentence variations while preserving required phrases exactly. Focus on visual diversity while maintaining semantic meaning. Variations must be semantically rich and complex — never trivially short or simplistic. Always use natural field names with spaces, never underscores."

        variations = []
        if not silent:
            print(f"      Generating null variations one by one...")

        if self.null_mode == "explicit":
            null_indicators = ["null"]
        else:
            null_indicators = ["not specified", "unspecified", "not available", "unavailable",
                               "unknown", "missing", "not provided", "not recorded",
                               original_replacement_value.lower()]

        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                lines = response.strip().split('\n')
                for idx, line in enumerate(lines):
                    clean_line = line.strip()
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
                    clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                    if clean_line and len(clean_line) > 10:
                        line_lower = clean_line.lower()
                        has_null_indicator = any(ind in line_lower for ind in null_indicators)
                        if has_null_indicator:
                            variations.append(clean_line)
                            if not silent:
                                print(f"        [{len(variations)}/{n}] {clean_line[:80]}{'...' if len(clean_line) > 80 else ''}")
            if len(variations) >= n:
                break

        return variations[:n]

    def generate_nonnull_variations_null(self, sentence: str, field_name: str, dummy_value: str, context: str = "") -> List[str]:
        variation_sentence = self.generate_sentence_for_field_value(field_name, dummy_value, context, natural_mode=False, original_sentence=sentence)
        return self.generate_structural_variations_standard(variation_sentence, field_name, context)

    def _parse_variations_response(self, response: str, max_variations: Optional[int] = None) -> List[str]:
        variations = []

        if response:
            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                clean_line = re.sub(r'^Variation \d+:?\s*', '', clean_line)

                if clean_line and len(clean_line) > 10:
                    variations.append(clean_line)

        cap = self.num_variations if max_variations is None else max_variations
        return variations[:cap]

    def generate_structural_variations_binary(self, sentence: str, field_name: str, field_value: str, context: str = "", silent: bool = False) -> List[str]:
        if not silent:
            print(f"      Generating binary variations for field: {field_name} with value: {field_value}")
        result = self.generate_structural_variations_standard_with_style(sentence, field_name, context, original_sentence=sentence, silent=silent)
        if not silent:
            print(f"      Generated {len(result)} binary variations")
        return result

    def generate_structural_variations_binary_counter(self, sentence: str, field_name: str, field_value: str, context: str = "", silent: bool = False) -> List[str]:
        opposite_value = "1" if field_value == "0" else "0"
        if not silent:
            print(f"      Generating binary counter variations for field: {field_name} with opposite value: {opposite_value}")

        counter_sentence = self.generate_simple_binary_counter_sentence(sentence, field_name, opposite_value)
        if not silent:
            print(f"      Generated counter variation sentence: {counter_sentence[:100]}...")

        result = self.generate_structural_variations_standard_with_style(counter_sentence, field_name, context, original_sentence=sentence, silent=silent)
        if not silent:
            print(f"      Generated {len(result)} binary counter variations")
        return result

    def generate_simple_binary_counter_sentence(self, original_sentence: str, field_name: str, opposite_value: str) -> str:
        natural_field_name = field_name.replace("_", " ")

        if self.binary_mode == "explicit":
            value_guidance = f"1. The rewritten sentence MUST contain the LITERAL value {opposite_value} for \"{natural_field_name}\" — use '0' or '1' explicitly, do NOT convert to yes/no or true/false"
        else:
            value_guidance = f"1. The rewritten sentence MUST express the opposite state for \"{natural_field_name}\" using natural language (e.g., 'is not', 'does not have', 'lacks') — do NOT use literal '0' or '1'"

        prompt = f"""Rewrite the following sentence so that the binary field "{natural_field_name}" has the opposite value instead.

Original sentence: "{original_sentence}"

RULES:
{value_guidance}
2. Keep the field name "{natural_field_name}" in NATURAL form with spaces — NEVER use underscores
3. The rewritten sentence MUST match the semantic complexity, length, and reading difficulty of the original — use rich vocabulary and varied clause structure
4. Preserve all other information and the professional tone of the original
5. Do NOT produce a trivially short sentence like "The value is {opposite_value}."
6. Return ONLY the rewritten sentence with no explanation

Rewritten sentence:"""

        system_message = "You are a precise sentence rewriter. You change a specific field value in a sentence while preserving the sentence's complexity, structure, and professional tone. Never simplify or shorten the output."

        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                result = response.strip()
                result = re.sub(r'^\d+[\.\)]\s*', '', result)
                result = re.sub(r'^[-•*]\s*', '', result)
                if result and len(result) > 15:
                    return result

        return f"The {natural_field_name} value is {opposite_value}."

    def generate_lexical_variations(self, base_words: List[str], context: str = "professional data report") -> Dict[str, List[str]]:
        if not base_words:
            return {}

        _ensure_nltk_wordnet()

        words_needing_filter: Dict[str, List[str]] = {}
        words_no_filter: Dict[str, List[str]] = {}

        for word in base_words:
            synonyms = set([word])
            synsets = wordnet.synsets(word)
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if (len(synonym) > 2 and
                        synonym.isalpha() and
                        synonym != word and
                        '_' not in lemma.name()):
                        synonyms.add(synonym)

            raw_synonyms = list(synonyms)
            if len(raw_synonyms) > 1:
                words_needing_filter[word] = raw_synonyms
            else:
                words_no_filter[word] = raw_synonyms

        batch_results = self._batch_filter_synonyms_with_llm(words_needing_filter, context)

        lexical_sets = {}
        for word in base_words:
            if word in batch_results:
                synonym_list = batch_results[word][:6]
            elif word in words_no_filter:
                synonym_list = words_no_filter[word]
            else:
                synonym_list = [word]

            if word in synonym_list:
                synonym_list.remove(word)
            synonym_list.insert(0, word)
            lexical_sets[word] = synonym_list
            print(f"    Filtered synonyms for '{word}': {synonym_list}")

        return lexical_sets

    def _batch_filter_synonyms_with_llm(
        self, word_synonym_map: Dict[str, List[str]], context: str
    ) -> Dict[str, List[str]]:
        """Filter synonyms for many words in a single LLM call.

        Returns a dict mapping each original word to its filtered synonym list.
        Falls back to returning the raw synonyms for any word whose entry
        cannot be parsed from the LLM response.
        """
        if not word_synonym_map:
            return {}

        entries = []
        for idx, (word, syns) in enumerate(word_synonym_map.items(), start=1):
            entries.append(f"{idx}. \"{word}\": {', '.join(syns)}")
        entries_text = "\n".join(entries)

        prompt = f"""Filter the synonym candidates for each word below.

Context: {context}

For every word, keep only synonyms that:
1. Make sense in professional documents
2. Have similar meaning to the original word
3. Are commonly used
4. Match the original word's form (plural/singular)

Exclude synonyms that are archaic, unusual, change the meaning, or sound
awkward in formal writing. Always include the original word itself.

Words and candidates:
{entries_text}

Return your answer as a numbered list in EXACTLY this format (one line per word,
same numbering, pipe-delimited synonyms):
1. "word": syn1|syn2|syn3
2. "word": syn1|syn2

Do NOT add any extra text, explanation, or blank lines.

Filtered:"""

        system_message = (
            "You are a synonym filter. Return ONLY the numbered list with "
            "pipe-delimited synonyms for each word. No explanation."
        )

        ordered_words = list(word_synonym_map.keys())

        for attempt in range(self.max_retries):
            try:
                response = self.query_local_llm(prompt, system_message)
                if not response:
                    continue
                parsed = self._parse_batch_synonym_response(response, ordered_words)
                if parsed:
                    for w in ordered_words:
                        if w not in parsed:
                            parsed[w] = word_synonym_map[w]
                    return parsed
            except Exception as e:
                print(f"    Batch synonym filter attempt {attempt + 1} failed: {e}")

        return {w: syns for w, syns in word_synonym_map.items()}

    @staticmethod
    def _parse_batch_synonym_response(
        response: str, ordered_words: List[str]
    ) -> Dict[str, List[str]]:
        """Parse the numbered, pipe-delimited synonym list returned by the LLM."""
        result: Dict[str, List[str]] = {}
        response = re.sub(r'^Filtered:?\s*', '', response.strip(), flags=re.IGNORECASE)

        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            word_key = None
            payload = None

            m = re.match(r'^(\d+)\.\s*"?([^":\|]+?)"?\s*:\s*(.+)$', line)
            if m:
                word_key = m.group(2).strip()
                payload = m.group(3).strip()
            else:
                m = re.match(r'^(\d+)\.\s*(.+)$', line)
                if m:
                    payload = m.group(2).strip()
                else:
                    continue

            parts = re.split(r'\|', payload)
            cleaned = [
                p.strip().strip('"').strip("'")
                for p in parts if p.strip() and len(p.strip()) > 1
            ]
            if not cleaned:
                continue

            matched_word = None

            if word_key:
                wk_lower = word_key.lower()
                for ow in ordered_words:
                    if ow.lower() == wk_lower:
                        matched_word = ow
                        break

            if not matched_word:
                for c in cleaned:
                    for ow in ordered_words:
                        if ow.lower() == c.lower() and ow not in result:
                            matched_word = ow
                            break
                    if matched_word:
                        break

            if not matched_word:
                line_idx_m = re.match(r'^(\d+)', line)
                if line_idx_m:
                    idx = int(line_idx_m.group(1)) - 1
                    if 0 <= idx < len(ordered_words) and ordered_words[idx] not in result:
                        matched_word = ordered_words[idx]

            if matched_word and matched_word not in result:
                if matched_word not in cleaned:
                    cleaned.insert(0, matched_word)
                result[matched_word] = cleaned[:6]

        return result if result else {}

    def identify_common_language_fields(self, data_fields: Dict[str, str]) -> List[str]:
        print("Checking for field values that are too commonly found in natural language...")

        field_list = []
        for field_name, field_value in data_fields.items():
            field_list.append(f"- {field_name}: {field_value}")

        field_list_text = "\n".join(field_list)

        prompt = f"""Analyze these field names and values. Identify which fields have values that are TOO COMMONLY FOUND in natural language.

Field names and values:
{field_list_text}

Examples of commonly found values: "traditional", "modern", "standard", "basic", "common", "regular", "normal", "typical", "general", "special"

Return ONLY the field names (one per line) whose values are common words. Return "NONE" if no fields qualify.

Field names:"""

        system_message = "You identify database field values that are too common in natural language. Return only field names, one per line."

        for attempt in range(self.max_retries):
            response = self.query_local_llm(prompt, system_message)
            if response:
                break

        if not response:
            print("    No response from LLM for common language field identification")
            return []

        common_field_names = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            line = re.sub(r'^Field\s+name[s]?:\s*', '', line, flags=re.IGNORECASE)

            if line and line.upper() != "NONE" and line in data_fields:
                common_field_names.append(line)
                print(f"    Identified common language field: {line} (value: {data_fields[line]})")

        if not common_field_names:
            print("    No field names identified as having commonly found values")
        else:
            print(f"    Found {len(common_field_names)} field names with commonly found values: {common_field_names}")

        return common_field_names
