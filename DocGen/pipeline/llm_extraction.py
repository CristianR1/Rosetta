"""LLM extraction methods for the DocumentTemplateSystem."""

import re
import traceback
from typing import Dict, List, Any, Tuple
from utils.cost_tracker import track_openai_response, track_local_llm_operation

from .config import get_local_llm_model


class LLMExtractionMixin:
    """Mixin providing LLM-based field extraction and verification methods.

    Expects the host class to provide:
        self.local_llm_client - OpenAI-compatible client
        self.null_mode - "explicit" or "implicit"
        self.binary_mode - "explicit" or "implicit"
        self.clean_llm_response(text) - strip markdown/quotes from LLM output
        self.clean_field_string(text) - normalise field name strings
    """

    def query_local_llm_for_field_verification(self, sentence: str, field_name: str, field_value: str, field_type: str = None, null_mode: str = "implicit", binary_mode: str = "implicit") -> bool:
        """
        Query local LLM to verify if BOTH field name AND field value appear in the sentence.
        This is stricter than field detection - requires both to be present for verification.
        Returns True only if BOTH field name AND field value are present, False otherwise.
        """
        value_format_guidance = ""
        percentage_equivalents = ""
        if field_value.replace('.', '', 1).isdigit():
            try:
                numeric_value = float(field_value)
                value_format_guidance = f"""
   - Numeric value: "{field_value}"
   - Integer form: "{int(numeric_value) if numeric_value == int(numeric_value) else field_value}"
   - Natural language numbers: "{field_value}" might appear as written words (e.g., "88" = "eighty-eight", "1" = "one")
   - Comma-formatted: Large numbers may have commas (e.g., "1000" = "1,000")
   - Ordinal forms: Numbers may appear as ordinals (e.g., "1" = "first", "2" = "second")"""

                if 0 < numeric_value < 1:
                    percentage_value = numeric_value * 100
                    percentage_2dec = round(percentage_value, 2)
                    percentage_1dec = round(percentage_value, 1)
                    percentage_0dec = round(percentage_value, 0)
                    percentage_equivalents = f"""
   
   CRITICAL: This is a decimal value between 0 and 1. It MUST be converted to a percentage:
   - Decimal: {field_value}
   - Percentage (2 decimals): {percentage_2dec}%
   - Percentage (1 decimal): {percentage_1dec}%
   - Percentage (rounded): {percentage_0dec}%
   - The sentence may contain ANY of these percentage forms: "{percentage_2dec}%", "{percentage_1dec}%", "{percentage_0dec}%", "{percentage_2dec} percent", "{percentage_1dec} percent", "approximately {percentage_0dec}%", "about {percentage_0dec} percent", etc.
   - Example: If field value is 0.657773689052438, look for "65.78%", "65.8%", "66%", "65.78 percent", "approximately 66%", etc. in the sentence."""
            except ValueError:
                pass

        mode_guidance = ""
        if field_type in ("BINARY", "NULLABLE_BINARY"):
            if binary_mode == "explicit":
                mode_guidance = "\n   NOTE: Binary mode is EXPLICIT - the value may appear as literal '0', '1', 'true', 'false', 'yes', 'no'"
            else:
                mode_guidance = "\n   NOTE: Binary mode is IMPLICIT - the value may appear as natural language (e.g., 'is a', 'operates as', 'does not operate as')"
        if field_type in ("NULL", "NULLABLE_BINARY"):
            if null_mode == "explicit":
                mode_guidance += "\n   NOTE: Null mode is EXPLICIT - the value may appear as literal 'NULL', 'NONE', 'N/A'"
            else:
                mode_guidance += "\n   NOTE: Null mode is IMPLICIT - the value may appear as natural language (e.g., 'not specified', 'unspecified', 'not available', 'unknown')"

        few_shot_examples = ""
        if field_value.replace('.', '', 1).isdigit():
            try:
                numeric_value = float(field_value)
                if 0 < numeric_value < 1:
                    percentage_value = numeric_value * 100
                    rounded_percentage = round(percentage_value, 2)
                    few_shot_examples = f"""

FEW-SHOT EXAMPLES:

Example 1:
Sentence: "The Percent Eligible FRPM K-12 recorded at 65.78% shows the proportion of students eligible."
Field name: "Percent (%) Eligible FRPM (K-12)"
Field value: "0.657773689052438"
Analysis: Field name "Percent Eligible FRPM K-12" matches (parentheses and % symbol variations are acceptable). Field value 0.657773689052438 = 65.78% (when multiplied by 100 and rounded). The sentence contains "65.78%" which matches the decimal value.
Answer: YES

Example 2:
Sentence: "The school has a Percent Eligible Free K-12 of 51.98 percent."
Field name: "Percent (%) Eligible Free (K-12)"
Field value: "0.519779208831647"
Analysis: Field name "Percent Eligible Free K-12" matches (parentheses variations acceptable). Field value 0.519779208831647 = 51.98% (when multiplied by 100). The sentence contains "51.98 percent" which matches.
Answer: YES

Example 3:
Sentence: "The enrollment count is 1087 students."
Field name: "Percent (%) Eligible FRPM (K-12)"
Field value: "0.657773689052438"
Analysis: Field name "Percent Eligible FRPM" is NOT present (only "enrollment" mentioned). Field value 0.657773689052438 is NOT present (only "1087" mentioned). Neither matches.
Answer: NO

Example 4:
Sentence: "The percentage eligible for FRPM in grades K-12 is approximately 66%."
Field name: "Percent (%) Eligible FRPM (K-12)"
Field value: "0.657773689052438"
Analysis: Field name "percentage eligible for FRPM in grades K-12" matches (word order variation, "grades" instead of parentheses). Field value 0.657773689052438 = 65.78% ≈ 66% (rounded). The sentence contains "66%" which is a reasonable approximation.
Answer: YES

Example 5:
Sentence: "Eighty-eight students participated in the SAT test."
Field name: "NumTstTakr"
Field value: "88"
Analysis: Field name "NumTstTakr" is NOT present (no mention of test takers or similar). Field value "88" appears as "Eighty-eight" (written form). However, field name is missing.
Answer: NO

KEY CONVERSION RULES:
- Decimal values between 0 and 1 should be converted to percentages: multiply by 100
- 0.657773689052438 = 65.78% (rounded to 2 decimal places) or approximately 66%
- 0.519779208831647 = 51.98% or approximately 52%
- Field names with parentheses like "Percent (%) Eligible FRPM (K-12)" match variations like "Percent Eligible FRPM K-12" or "percentage eligible FRPM K-12"
- Percentages can appear as "65.78%", "65.78 percent", "approximately 66%", etc.
- Rounding is acceptable: 0.6577... can match "65.78%", "66%", "approximately 66 percent", etc.

"""
            except ValueError:
                pass

        prompt = f"""Analyze the following sentence and determine if it contains BOTH the specified field name AND the field value.

Sentence: "{sentence}"

Field name to detect: {field_name}
Field value to detect: {field_value}
Field type: {field_type if field_type else "STANDARD"}{mode_guidance}

IMPORTANT INSTRUCTIONS:
1. You must verify that BOTH the field name AND the field value are present in the sentence
2. Consider ALL possible forms of the field name: abbreviated, unabbreviated, extended, or shortened versions
3. The field name may appear in various forms such as:
   - With or without parenthetical information (e.g., "Charter School (Y/N)" might appear as just "Charter School")
   - Abbreviated (e.g., "FRPM" for "Free Reduced Price Meal" might appear as "FRPM" or "Free Reduced Price Meal")
   - Expanded (e.g., "Percent Eligible Free K-12" might appear as "percentage eligible" or "percent of eligible")
   - Partial (e.g., "Student Count" might appear as just "students" or "count")
   - With different word order or insertions (e.g., "School Type" might appear as "type of school")
   - Pluralized or singularized versions
   - Parentheses and symbols can be omitted (e.g., "Percent (%) Eligible FRPM (K-12)" matches "Percent Eligible FRPM K-12")
4. The field value may appear in various forms:{value_format_guidance}{percentage_equivalents}
   - Exact match: "{field_value}"
   - Abbreviated or natural language forms (e.g., "88" might appear as "eighty-eight")
   - Numeric variations (e.g., "1" might appear as "one")
   - For numeric values: Consider all formats handled by parsing logic (integers, decimals, percentages, comma-formatted, ordinals, written words)
5. Return YES ONLY if BOTH the field name (in any form) AND the field value (in any form) are present in the sentence
6. Return NO if only the field name is present (without the value)
7. Return NO if only the field value is present (without the field name)
8. Return NO if NEITHER the field name NOR the field value is present
9. Look for semantic matches, not just exact string matches
10. Do NOT include any markdown, explanations, or additional text in your response
11. Return ONLY "YES" or "NO"{few_shot_examples}

Response:"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise field verification assistant. You verify that BOTH a field name AND its value appear in a sentence. You return only YES or NO, with no additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('field_verification')

            if not response or not response.choices:
                print(f"      LLM verification returned no response for field '{field_name}'")
                return False

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                print(f"      LLM verification returned None for field '{field_name}'")
                return False

            result = content.strip()
            print(f"      LLM verification response for '{field_name}': '{result[:100]}...' (truncated)" if len(result) > 100 else f"      LLM verification response for '{field_name}': '{result}'")

            result_upper = result.upper()

            if "YES" in result_upper and "NO" not in result_upper:
                print(f"      Extracted answer: YES (both field name and value present)")
                return True
            elif "NO" in result_upper and "YES" not in result_upper:
                print(f"      Extracted answer: NO (field name or value missing)")
                return False
            elif result_upper == "YES":
                return True
            elif result_upper == "NO":
                return False
            else:
                print(f"      Could not extract clear YES/NO from verification response")
                return False
        except Exception as e:
            print(f"      Exception in LLM field verification: {e}")
            print(f"      Traceback: {traceback.format_exc()}")
            return False

    def query_local_llm_for_field_detection(self, sentence: str, field_name: str, field_value: str = None, other_field_names: List[str] = None) -> bool:
        """
        Query local LLM to identify if a field is eligible for placeholder replacement in the sentence.

        Returns True only if the sentence is eligible for placeholder replacement:
        - Both field name AND field value present: YES (eligible)
        - Only field value present AND value is not part of another field's name: YES (eligible)
        - Only field name present (no value): NO (nothing to replace)
        - Neither present: NO (not detected)
        - Field value is part of another field's name: NO (ambiguous)

        Args:
            sentence: The sentence to analyze
            field_name: The field name to detect
            field_value: The field value to detect (optional)
            other_field_names: List of other field names in context (to check for ambiguity)
        """
        field_value_info = ""
        if field_value is not None:
            field_value_info = f"\nField value to detect: {field_value}"

        other_fields_context = ""
        if other_field_names:
            other_fields_str = ", ".join([f'"{fn}"' for fn in other_field_names[:10]])
            other_fields_context = f"\nOther field names in this record (for context): {other_fields_str}"

        prompt = f"""Analyze the following sentence to determine if a field is ELIGIBLE FOR PLACEHOLDER REPLACEMENT.

Sentence: "{sentence}"

Field name to detect: {field_name}{field_value_info}{other_fields_context}

ANALYSIS INSTRUCTIONS:
1. First, check if the FIELD NAME appears in the sentence (in any form: abbreviated, expanded, partial, different word order, pluralized, etc.)
   - Examples: "Charter School (Y/N)" might appear as "Charter School", "charter status", etc.
   - Examples: "Enrollment (K-12)" might appear as "enrollment", "K-12 enrollment", "students enrolled", etc.
   - IMPORTANT: A generic word like "enrollment" appearing alone does NOT mean "Enrollment (K-12)" is present unless there's context indicating K-12 or the specific enrollment being discussed.

2. Second, check if the FIELD VALUE appears in the sentence (exact match or equivalent representation)
   - For numbers: consider written forms ("88" = "eighty-eight"), percentages, comma formatting
   - For text: consider exact matches or close paraphrases

3. Determine eligibility based on these rules:
   A) BOTH field name AND field value are present → Return "YES" (eligible for replacement)
   B) ONLY field value is present (field name is absent):
      - Check if the field value appears as part of any other field name listed in context
      - If field value is NOT part of another field name → Return "YES" (eligible for replacement)
      - If field value IS part of another field name → Return "NO" (ambiguous - cannot safely replace)
   C) ONLY field name is present (field value is absent) → Return "NO" (nothing to replace)
   D) NEITHER field name NOR field value is present → Return "NO" (not detected)

4. CRITICAL: Be strict about field name detection. The word "enrollment" alone does NOT confirm "Enrollment (K-12)" unless there is clear context about K-12 grades or the specific type of enrollment.

5. Return ONLY "YES" or "NO" - no explanations, markdown, or additional text.

Response:"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise field detection assistant that determines if a field is eligible for placeholder replacement. You analyze whether BOTH the field name AND field value appear in a sentence. You return only YES or NO, with no additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('field_detection')

            if not response or not response.choices:
                print(f"      LLM returned no response or empty choices for field '{field_name}'")
                print(f"      Response object: {response}")
                return False

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
                print(f"      Using reasoning_content for field '{field_name}'")

            if content is None:
                print(f"      LLM returned None for both content and reasoning_content for field '{field_name}'")
                print(f"      This might indicate: LLM timeout, empty response, or endpoint issue")
                return False

            result = content.strip()
            print(f"      LLM field detection response for '{field_name}': '{result[:100]}...' (truncated)" if len(result) > 100 else f"      LLM field detection response for '{field_name}': '{result}'")

            result_upper = result.upper()

            if "YES" in result_upper and "NO" not in result_upper:
                print(f"      Extracted answer: YES (eligible for replacement)")
                return True
            elif "NO" in result_upper and "YES" not in result_upper:
                print(f"      Extracted answer: NO (not eligible for replacement)")
                return False
            elif result_upper == "YES":
                return True
            elif result_upper == "NO":
                return False
            else:
                print(f"      Could not extract clear YES/NO from response")
                return False
        except Exception as e:
            print(f"      Exception querying local LLM for field detection: {e}")
            print(f"      Exception type: {type(e).__name__}")
            print(f"      Traceback: {traceback.format_exc()}")
            return False

    def query_local_llm_for_date_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """Query local LLM to extract date representations in various formats."""
        prompt = f"""Analyze the following sentence and identify the EXACT text portion that represents the date value.

Sentence: "{sentence}"

Field name: {field_name}
Field value (database format): {field_value}

IMPORTANT INSTRUCTIONS:
1. The date value may be in various formats:
   - Standard: "2020-01-15", "01/15/2020", "15-01-2020"
   - Written: "January 15, 2020", "15th January 2020", "Jan 15 2020"
   - Abbreviated: "15-Jan-20", "Jan-15-2020"
2. Find the EXACT text span in the sentence that represents this date
3. Return ONLY the matched date text from the sentence - no explanations, no markdown, no quotes
4. If the date is NOT present in the sentence, return "NOT_FOUND"

Your response (only the matched date text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise date extraction assistant. You identify exact text spans that match date values in various formats. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('date_extraction')

            if not response or not response.choices:
                return ""

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                return ""

            result = self.clean_llm_response(content)

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND", "NONE", "NULL"]:
                return ""

            if result and result.lower() in sentence.lower():
                return result
            else:
                return ""

        except Exception as e:
            print(f"      Exception in LLM date extraction: {e}")
            return ""

    def query_local_llm_for_complex_list_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """Query local LLM to extract the natural language text that represents a complex field value."""
        prompt = f"""Analyze the following sentence and identify the EXACT text portion that describes or represents the field value in natural language format.

Sentence: "{sentence}"

Field name: {field_name}
Field value (database format): {field_value}

IMPORTANT INSTRUCTIONS:
1. The field value may be:
   - A simple comma-separated list: "item1,item2,item3"
   - A complex structure (JSON, dict): {{'key': 'value', 'nested': {{'data': 123}}}}
   - A nested list or dictionary with multiple levels
2. The sentence describes this data in NATURAL LANGUAGE format
3. For simple lists: May appear as "item1, item2 and item3" or "item1 and item2"
4. For complex structures: May describe parts/subsets like "1 basic, 10 common, 1 rare"
5. Find the EXACT text span in the sentence that describes ANY PART or ALL of the field value
6. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
7. If the field value is NOT represented in the sentence, return "NOT_FOUND"

Example (URL list unpacked in sentence):
Sentence: "The Purchase URLs are available at Card Kingdom, Cardmarket, and TCGPlayer via the links: https://example.com/1, https://example.com/2, and https://example.com/3 respectively."
Field value: {{'cardKingdom': 'url1', 'cardmarket': 'url2', 'tcgplayer': 'url3'}}
Response: Card Kingdom, Cardmarket, and TCGPlayer via the links: https://example.com/1, https://example.com/2, and https://example.com/3 respectively

Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that match field values in natural language. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('complex_list_extraction')

            if not response or not response.choices:
                return ""

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                return ""

            result = self.clean_llm_response(content)

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND", "NONE", "NULL"]:
                return ""

            if result and result.lower() in sentence.lower():
                return result
            else:
                return ""

        except Exception as e:
            print(f"      Exception in LLM complex list extraction: {e}")
            return ""

    def query_local_llm_for_null_value_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """Query local LLM to extract text indicating a NULL/missing value in the sentence."""
        prompt = f"""Analyze the following sentence and identify the EXACT text portion that indicates a NULL, missing, or unspecified value for the given field.

Sentence: "{sentence}"

Field name: {field_name}
Field value (database format): {field_value}

IMPORTANT INSTRUCTIONS:
1. Look for text patterns that indicate the field value is NULL, missing, absent, or unspecified
2. Common patterns include but are not limited to:
   - "not specified", "unspecified", "not detailed"
   - "not available", "unavailable", "N/A"
   - "unknown", "not known"
   - "not provided", "not given", "not recorded"
   - "missing", "absent", "lacks"
   - "none", "null", "empty"
3. Find the EXACT text span in the sentence that indicates this null/missing status
4. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
5. If no null/missing indicator is present in the sentence, return "NOT_FOUND"

Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that indicate NULL or missing values. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('null_extraction')

            if not response or not response.choices:
                return ""

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                return ""

            result = self.clean_llm_response(content)

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND"]:
                return ""

            if result and result.lower() in sentence.lower():
                return result
            else:
                return ""

        except Exception as e:
            print(f"      Exception in LLM null value extraction: {e}")
            return ""

    def query_local_llm_for_binary_value_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """Query local LLM to extract text indicating a binary (0 or 1) value in the sentence."""
        is_positive = field_value == '1'
        value_description = "positive/true/yes/1" if is_positive else "negative/false/no/0"

        prompt = f"""Analyze the following sentence and identify the EXACT text portion that indicates a binary field value.

Sentence: "{sentence}"

Field name: {field_name}
Field value (database format): {field_value}
Expected meaning: {value_description}

IMPORTANT INSTRUCTIONS:
1. The binary value (0 or 1) could be:
   - Directly embedded: "0", "1", "yes", "no", "true", "false"
   - Implied through context: "is a [field]", "is not a [field]", "operates as", "does not operate as"
2. For value 1 (positive/true), look for:
   - Affirmative statements: "is a", "operates as", "functions as", "designated as"
   - Positive indicators: "yes", "true", "1", "active", "enabled"
3. For value 0 (negative/false), look for:
   - Negative statements: "is not a", "not operating as", "does not function as"
   - Negative indicators: "no", "false", "0", "inactive", "disabled", "non-"
4. Find the EXACT text span in the sentence that indicates this binary value
5. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
6. If the binary value is NOT indicated in the sentence, return "NOT_FOUND"

Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that indicate binary (0/1, yes/no, true/false) values. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('binary_extraction')

            if not response or not response.choices:
                return ""

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                return ""

            result = self.clean_llm_response(content)

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND"]:
                return ""

            if result and result.lower() in sentence.lower():
                return result
            else:
                return ""

        except Exception as e:
            print(f"      Exception in LLM binary value extraction: {e}")
            return ""

    def query_local_llm_for_misc_value_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """Query local LLM to extract text indicating a miscellaneous character value in the sentence."""
        readable_chars = []
        char_map = {
            '-': 'dash/hyphen', '/': 'forward slash', '\\': 'backslash', '*': 'asterisk',
            '#': 'hash/pound', '@': 'at symbol', '&': 'ampersand', '%': 'percent',
            '$': 'dollar sign', '!': 'exclamation mark', '?': 'question mark',
            '.': 'period/dot', ',': 'comma', ':': 'colon', ';': 'semicolon',
            '_': 'underscore', '+': 'plus sign', '=': 'equals sign', '~': 'tilde',
            '`': 'backtick', '^': 'caret', '|': 'pipe/vertical bar',
            '<': 'less than', '>': 'greater than', '\u2013': 'en-dash', '\u2014': 'em-dash',
            '\u2026': 'ellipsis', '\u2022': 'bullet point'
        }

        for char in field_value:
            if char in char_map:
                readable_chars.append(char_map[char])
            elif char in '()':
                readable_chars.append('parenthesis')
            elif char in '[]':
                readable_chars.append('bracket')
            elif char in '{}':
                readable_chars.append('brace/curly bracket')
            elif char in '\'"':
                readable_chars.append('quote')
            else:
                readable_chars.append(f'"{char}"')

        char_description = ', '.join(set(readable_chars)) if readable_chars else field_value

        prompt = f"""Analyze the following sentence and identify the EXACT text portion that represents or describes the miscellaneous character value.

Sentence: "{sentence}"

Field name: {field_name}
Field value (database format): {field_value}
Character types: {char_description}

IMPORTANT INSTRUCTIONS:
1. The miscellaneous character value may appear in various forms:
   - Directly embedded: "{field_value}" appearing as-is in the text
   - Described in words: "dash", "hyphen", "slash", "asterisk", etc.
   - As a placeholder indicator: "marked with", "indicated by", "represented by"
2. CRITICAL: Misc characters like "-" or "+" are OFTEN MISREPRESENTED as null/missing values:
   - "-" is commonly written as: "not specified", "unspecified", "not available", "unavailable", "unknown", "missing", "not provided", "none", "absent", "no data", "N/A"
   - "+" is commonly written as: "positive", "plus", "added", "additional", "admitted", "hospitalized", "confirmed"
3. Find the EXACT text span in the sentence that represents this character value (including null-like phrases for "-" and "+")
4. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
5. If the miscellaneous character value is NOT present in the sentence, return "NOT_FOUND"

Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that represent or describe miscellaneous character values. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('misc_extraction')

            if not response or not response.choices:
                return ""

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                return ""

            result = self.clean_llm_response(content)

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND"]:
                return ""

            if result and result.lower() in sentence.lower():
                return result
            else:
                return ""

        except Exception as e:
            print(f"      Exception in LLM misc value extraction: {e}")
            return ""

    def query_local_llm_for_value_replacement(self, sentence: str, field_name: str, field_value: str, replacement_value: str, placeholder: str, max_attempts: int = 3) -> Tuple[str, int]:
        """
        Query local LLM to find and replace field value in sentence when direct parsing fails.
        Retries up to max_attempts before giving up.

        Returns:
            Tuple[str, int]: (modified_sentence, number_of_replacements_made)
        """
        prompt = f"""Analyze the following sentence and find any text that represents the field value, including abbreviations, natural language variations, or near-exact matches.

Sentence: "{sentence}"

Field name: {field_name}
Field value (database format): {field_value}
Replacement value (detected format): {replacement_value}
Target placeholder: {placeholder}

IMPORTANT INSTRUCTIONS:
1. The field value may appear in various forms:
   - Exact match: "{field_value}" or "{replacement_value}"
   - Abbreviated form (e.g., "88" might appear as "eighty-eight" or "Eighty-eight")
   - Natural language variation (e.g., numeric values written as words)
   - Near-exact matches with slight variations in formatting
2. Find ALL occurrences of the value in the sentence
3. Replace each occurrence with the placeholder: {placeholder}
4. Return ONLY the modified sentence with replacements made - no explanations, no markdown, no quotes
5. If the value is NOT found in any form, return the original sentence unchanged
6. Preserve all other text exactly as it appears

Example:
Original: "Eighty-eight students were involved in the SAT test at this school."
Field value: "88"
Replacement: "88"
Placeholder: "[NUMTSTTAKR]"
Response: "[NUMTSTTAKR] students were involved in the SAT test at this school."

Your response (only the modified sentence):"""

        system_message = "You are a precise text replacement assistant. You find field values in sentences (including abbreviations and natural language forms) and replace them with placeholders. You return ONLY the modified sentence with no additional formatting or explanation."

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.local_llm_client.chat.completions.create(
                    model=get_local_llm_model(),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16000,
                )

                track_openai_response(response, get_local_llm_model())
                track_local_llm_operation('value_replacement')

                if not response or not response.choices:
                    print(f"        LLM fallback attempt {attempt}/{max_attempts} returned no response for field '{field_name}'")
                    continue

                message = response.choices[0].message
                content = message.content

                if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                    content = message.reasoning_content

                if content is None:
                    print(f"        LLM fallback attempt {attempt}/{max_attempts} returned None for field '{field_name}'")
                    continue

                result = self.clean_llm_response(content)

                original_placeholder_count = sentence.count(placeholder)
                new_placeholder_count = result.count(placeholder)
                replacements_made = new_placeholder_count - original_placeholder_count

                if replacements_made > 0:
                    print(f"        LLM fallback attempt {attempt}/{max_attempts} made {replacements_made} replacement(s) for field '{field_name}'")
                    return result, replacements_made

                if result != sentence:
                    print(f"        LLM fallback attempt {attempt}/{max_attempts} modified sentence for field '{field_name}' (replacement count unclear)")
                    return result, 1

                print(f"        LLM fallback attempt {attempt}/{max_attempts} found no matches for field '{field_name}'")

            except Exception as e:
                print(f"        LLM fallback attempt {attempt}/{max_attempts} exception for field '{field_name}': {e}")

        print(f"        LLM fallback exhausted {max_attempts} attempts for field '{field_name}' — no replacement made")
        return sentence, 0

    def query_local_llm_for_counter_variation_refinement(self, sentence: str, null_mode: str = "implicit", binary_mode: str = "implicit") -> str:
        """
        Refine a counter variation sentence to improve grammar and wording.
        Ensures null/binary language is used correctly based on the exposure modes.
        Handles both NULL and BINARY counter variations.
        """
        mode_guidance = ""
        if null_mode == "explicit":
            mode_guidance += " For null values, use explicit language if appropriate (e.g., 'NULL', 'N/A')."
        else:
            mode_guidance += " For null values, use natural language (e.g., 'not specified', 'unspecified', 'not available')."

        if binary_mode == "explicit":
            mode_guidance += " For binary values, use explicit language if appropriate (e.g., '0', '1', 'false', 'true')."
        else:
            mode_guidance += " For binary values, use natural language (e.g., 'does not operate as', 'is not a')."

        prompt = f"""Refine the following sentence to improve its grammar, wording, and natural flow. The sentence is a counter variation indicating that a field value is not present or has an opposite state.

Original sentence: "{sentence}"

IMPORTANT INSTRUCTIONS:
1. Keep the core meaning and all field placeholders (e.g., [FIELD_NAME]) exactly as they are
2. Improve grammar, word choice, and sentence flow to make it read naturally
3. Ensure "not specified" (or appropriate null/binary language) is used correctly and naturally{mode_guidance}
4. Fix any awkward phrasing or grammatical errors (e.g., "that is not specified" should become "not specified" where appropriate)
5. Maintain the same tone and style as the original
6. Do NOT change field placeholders or remove any information
7. The refined sentence MUST retain the same semantic complexity and reading difficulty as standard data-bearing variations — use rich vocabulary, subordinate clauses, and professional phrasing. Do NOT simplify or shorten the sentence
8. Return ONLY the refined sentence with no explanations, no markdown, no quotes

Refined sentence:"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text refinement assistant. You improve grammar and wording of sentences while preserving all field placeholders and core meaning. The refined sentence must be semantically rich and complex — never trivially short or simplistic. You return only the refined sentence with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=16000,
            )

            track_openai_response(response, get_local_llm_model())
            track_local_llm_operation('counter_refinement')

            if not response or not response.choices:
                print(f"        LLM refinement returned no response, using original sentence")
                return sentence

            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content

            if content is None:
                print(f"        LLM refinement returned None, using original sentence")
                return sentence

            result = self.clean_llm_response(content)

            original_placeholders = set(re.findall(r'\[([^\]]+)\]', sentence))
            refined_placeholders = set(re.findall(r'\[([^\]]+)\]', result))

            if original_placeholders != refined_placeholders:
                print(f"        Warning: Placeholders changed during refinement. Original: {original_placeholders}, Refined: {refined_placeholders}. Using original.")
                return sentence

            if result and len(result.strip()) > 0:
                print(f"        Refined counter variation")
                return result
            else:
                print(f"        LLM refinement returned empty result, using original sentence")
                return sentence

        except Exception as e:
            print(f"        Exception in LLM counter variation refinement: {e}")
            print(f"        Traceback: {traceback.format_exc()}")
            return sentence
