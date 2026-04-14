"""LLM query methods for NarrativeParsingAnalyzer."""
import re
from typing import Tuple, Optional
from utils.cost_tracker import track_openai_response
from .config import get_local_llm_model
from .text_utils import clean_llm_response
from .template_generator import TemplateGenerator


class NarrativeLLMMixin:
    """Mixin providing LLM query and sentence rewriting methods.

    Expects the host class to provide:
        self.local_llm_client
        self.null_mode, self.binary_mode
    """

    def query_local_llm_for_date_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """
        Query a local LLM to extract date representations in various formats.
        Returns the extracted date text portion, or empty string if not found.
        """
        prompt = f"""Analyze the following sentence and identify the EXACT text portion that represents the date value.

            Sentence: "{sentence}"

            Field name: {field_name}
            Field value (database format): {field_value}

            IMPORTANT INSTRUCTIONS:
            1. The date value may be in various formats:
            - Standard: "2020-01-15", "01/15/2020", "15-01-2020"
            - Written: "January 15, 2020", "15th January 2020", "Jan 15 2020"
            - Abbreviated: "15-Jan-20", "Jan-15-2020"
            2. The sentence may describe the same date in a different format
            3. Find the EXACT text span in the sentence that represents this date
            4. Return ONLY the matched date text from the sentence - no explanations, no markdown, no quotes
            5. If the date is NOT present in the sentence, return "NOT_FOUND"
            6. Do NOT add any text that's not in the original sentence

            Example 1:
            Sentence: "The event occurred on January 15, 2020 in the afternoon."
            Field value: "2020-01-15"
            Response: January 15, 2020

            Example 2:
            Sentence: "Updated on 15th Jan 2020."
            Field value: "2020-01-15"
            Response: 15th Jan 2020

            Example 3:
            Sentence: "The record has no date."
            Field value: "2020-01-15"
            Response: NOT_FOUND

            Your response (only the matched date text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise date extraction assistant. You identify exact text spans that match date values in various formats. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"      LLM returned no response for date extraction")
                return ""

            message = response.choices[0].message
            content = message.content
            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            if content is None:
                print(f"      LLM returned None for date extraction")
                return ""

            result = content.strip()
            result = result.strip('"\'`')
            result = result.replace('**', '')
            print(f"      LLM date extraction: '{result}'")
            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND", "NONE", "NULL"]:
                return ""
            if result and result.lower() in sentence.lower():
                return result
            else:
                print(f"      LLM extracted date not found in sentence: '{result}'")
                return ""

        except Exception as e:
            print(f"      Exception in LLM date extraction: {e}")
            return ""

    def query_local_llm_for_complex_list_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """
        Query a local LLM to extract the natural language text that represents a complex field value.
        Handles simple lists, complex JSON/dict structures, and nested data.
        Returns the extracted text portion, or empty string if not found.
        """
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
            4. For complex structures: May describe parts/subsets like "1 basic, 10 common, 1 rare" from {{'basic': 1, 'common': 10, 'rare': 1}}
            5. Find the EXACT text span in the sentence that describes ANY PART or ALL of the field value
            6. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
            7. If the field value is NOT represented in the sentence, return "NOT_FOUND"
            8. Do NOT add any text that's not in the original sentence
            9. The text may describe only a subset of the data - that's OK, extract what's there

            Example 1 (Simple list):
            Sentence: "The colors available are red, blue and green in stock."
            Field value: "red,blue,green"
            Response: red, blue and green

            Example 2 (Simple list with context):
            Sentence: "This card is available on both mtgo and paper formats."
            Field value: "mtgo,paper"
            Response: mtgo and paper

            Example 3 (Complex structure):
            Sentence: "The pack contains 1 basic, 10 common, 1 rare, and 3 uncommon cards, weighted at 1913922."
            Field value: {{'basic': 1, 'common': 10, 'rare': 1, 'uncommon': 3, 'weight': 1913922}}
            Response: 1 basic, 10 common, 1 rare, and 3 uncommon cards, weighted at 1913922

            Example 4 (URL list unpacked in sentence):
            Sentence: "The Purchase URLs for this trading card are available at Card Kingdom, Cardmarket, and TCGPlayer via the links: https://mtgjson.com/links/9fb51af0ad6f0736, https://mtgjson.com/links/ace8861194ee0b6a, and https://mtgjson.com/links/4843cea124a0d515 respectively."
            Field value: {{'cardKingdom': 'https://mtgjson.com/links/9fb51af0ad6f0736', 'cardmarket': 'https://mtgjson.com/links/ace8861194ee0b6a', 'tcgplayer': 'https://mtgjson.com/links/4843cea124a0d515'}}
            Response: Card Kingdom, Cardmarket, and TCGPlayer via the links: https://mtgjson.com/links/9fb51af0ad6f0736, https://mtgjson.com/links/ace8861194ee0b6a, and https://mtgjson.com/links/4843cea124a0d515 respectively

            Example 5 (Dict keys become list items):
            Sentence: "This item can be purchased from cardKingdom, cardmarket, or tcgplayer online stores."
            Field value: {{'cardKingdom': 'url1', 'cardmarket': 'url2', 'tcgplayer': 'url3'}}
            Response: cardKingdom, cardmarket, or tcgplayer

            Example 6 (Not found):
            Sentence: "The item is unavailable."
            Field value: "red,blue,green"
            Response: NOT_FOUND

            Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that match field values in natural language. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"      LLM returned no response for complex list extraction")
                return ""
            message = response.choices[0].message
            content = message.content

            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            if content is None:
                print(f"      LLM returned None for complex list extraction")
                return ""
            result = content.strip()
            result = result.strip('"\'`')
            result = result.replace('**', '')
            print(f"      LLM complex list extraction: '{result}'")

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND", "NONE", "NULL"]:
                return ""
            if result and result.lower() in sentence.lower():
                return result
            else:
                print(f"      LLM extracted text not found in sentence: '{result}'")
                return ""

        except Exception as e:
            print(f"      Exception in LLM complex list extraction: {e}")
            return ""

    def query_local_llm_for_field_detection(self, sentence: str, field_name: str, field_value: str = None, other_field_names: list = None) -> bool:
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
                max_tokens=1000,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"        LLM returned no response or empty choices for field '{field_name}'")
                print(f"      Response object: {response}")
                return False

            message = response.choices[0].message
            content = message.content
            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
                print(f"        Using reasoning_content for field '{field_name}'")
            if content is None:
                print(f"        LLM returned None for both content and reasoning_content for field '{field_name}'")
                print(f"      This might indicate: LLM timeout, empty response, or endpoint issue")
                return False

            result = content.strip()
            print(f"      LLM field detection response for '{field_name}': '{result[:100]}...' (truncated)" if len(result) > 100 else f"      LLM field detection response for '{field_name}': '{result}'")
            result_upper = result.upper()
            if "YES" in result_upper and "NO" not in result_upper:
                print(f"        Extracted answer: YES (eligible for replacement)")
                return True
            elif "NO" in result_upper and "YES" not in result_upper:
                print(f"        Extracted answer: NO (not eligible for replacement)")
                return False
            elif result_upper == "YES":
                return True
            elif result_upper == "NO":
                return False
            else:
                print(f"        Could not extract clear YES/NO from response")
                return False
        except Exception as e:
            print(f"        Exception querying local LLM for field detection: {e}")
            print(f"      Exception type: {type(e).__name__}")
            import traceback
            print(f"      Traceback: {traceback.format_exc()}")
            return False

    def query_local_llm_for_targeted_verification(
        self, sentence: str, column_name: str, column_value: str, field_type: str = "STANDARD"
    ) -> Tuple[bool, str]:
        """
        Single LLM call for Phase 2: verify if sentence contains column data and extract replacement value.
        Returns (found, replacement_value). Used when rule-based check fails.
        """
        field_value_str = str(column_value).strip()
        type_hint = ""
        if field_type == "NULL":
            type_hint = " The value indicates NULL/missing (e.g., 'not specified', 'unavailable')."
        elif field_type == "BINARY":
            type_hint = " The value is binary (0/1, yes/no, true/false, or implicit like 'charter school')."

        prompt = f"""This sentence is from a narrative and should contain data for a specific column.

SENTENCE: "{sentence}"

COLUMN: {column_name}
EXPECTED VALUE: {field_value_str}{type_hint}

TASK:
1. Does this sentence contain the data for column "{column_name}" with value "{field_value_str}"?
2. If YES: extract the EXACT text span from the sentence that represents this value. Return it on a new line after "YES:".
3. If NO: return only "NO".

FORMAT:
- If found: "YES:\n<exact text from sentence>"
- If not found: "NO"

Return ONLY your response, no other text."""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You verify if a sentence contains specific column data and extract the exact text. Return YES with extraction or NO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )
            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                return False, ""
            content = response.choices[0].message.content
            if content is None:
                return False, ""
            result = content.strip().upper()
            if result == "NO" or (result.startswith("NO") and len(result) <= 5):
                return False, ""
            if "YES" in result:
                stripped = content.strip()
                if "YES:" in stripped or "YES :" in stripped.upper():
                    parts = re.split(r"YES\s*:\s*", stripped, 1, flags=re.IGNORECASE)
                    if len(parts) > 1 and parts[1].strip():
                        extraction = parts[1].strip().split("\n")[0].strip()
                        if extraction and extraction.upper() != "NO":
                            return True, extraction
                lines = stripped.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().upper().startswith("YES"):
                        if i + 1 < len(lines):
                            extraction = lines[i + 1].strip()
                            if extraction and extraction.upper() != "NO":
                                return True, extraction
                        return True, field_value_str
            return False, ""
        except Exception as e:
            print(f"      Exception in query_local_llm_for_targeted_verification: {e}")
            return False, ""

    def query_local_llm_for_null_value_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """
        Query a local LLM to extract text indicating a NULL/missing value in the sentence.
        Returns the exact text portion indicating null, or empty string if not found.
        """
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
            - "not listed", "not mentioned", "not defined"
            - "no [field name]", "without [field name]"
            - "does not have", "has no", "is without"
            - "non-existent", "does not exist"
            3. Find the EXACT text span in the sentence that indicates this null/missing status
            4. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
            5. If no null/missing indicator is present in the sentence, return "NOT_FOUND"
            6. Do NOT add any text that's not in the original sentence

            Example 1:
            Sentence: "The student's grade level is not specified in the records."
            Field name: "Grade Level"
            Field value: "NULL"
            Response: not specified

            Example 2:
            Sentence: "The address information remains unavailable for this entry."
            Field name: "Address"
            Field value: "NULL"
            Response: unavailable

            Example 3:
            Sentence: "This record lacks a valid phone number."
            Field name: "Phone Number"
            Field value: "-"
            Response: lacks a valid phone number

            Example 4:
            Sentence: "The student attends Lincoln High School."
            Field name: "Grade Level"
            Field value: "NULL"
            Response: NOT_FOUND

            Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that indicate NULL or missing values. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"      LLM returned no response for null value extraction")
                return ""

            message = response.choices[0].message
            content = message.content
            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            if content is None:
                print(f"      LLM returned None for null value extraction")
                return ""
            result = clean_llm_response(content)

            print(f"      LLM null value extraction: '{result}'")

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND"]:
                return ""
            if result and result.lower() in sentence.lower():
                return result
            else:
                print(f"      LLM extracted null text not found in sentence: '{result}'")
                return ""

        except Exception as e:
            print(f"      Exception in LLM null value extraction: {e}")
            return ""

    def query_local_llm_for_binary_value_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """
        Query a local LLM to extract text indicating a binary (0 or 1) value in the sentence.
        The value could be directly embedded or implied through context.
        Returns the exact text portion indicating the binary value, or empty string if not found.
        """
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
            - Positive indicators: "yes", "true", "1", "active", "enabled","includes"
            3. For value 0 (negative/false), look for:
            - Negative statements: "is not a", "not operating as", "does not function as"
            - Negative indicators: "no", "false", "0", "inactive", "disabled", "non-","does not include"
            4. Find the EXACT text span in the sentence that indicates this binary value
            5. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
            6. If the binary value is NOT indicated in the sentence, return "NOT_FOUND"
            7. Do NOT add any text that's not in the original sentence

            Example 1:
            Sentence: "This school operates as a charter school with special funding."
            Field name: "Charter School (Y/N)"
            Field value: "1"
            Response: operates as a charter school

            Example 2:
            Sentence: "The institution is not a magnet school."
            Field name: "Magnet School (Y/N)"
            Field value: "0"
            Response: is not a magnet school

            Example 3:
            Sentence: "The charter school designation is active for this campus."
            Field name: "Charter School (Y/N)"
            Field value: "1"
            Response: charter school designation is active

            Example 4:
            Sentence: "The school has 500 students enrolled."
            Field name: "Charter School (Y/N)"
            Field value: "1"
            Response: NOT_FOUND

            Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that indicate binary (0/1, yes/no, true/false) values. The value may be directly stated or implied through context. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"      LLM returned no response for binary value extraction")
                return ""

            message = response.choices[0].message
            content = message.content
            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            if content is None:
                print(f"      LLM returned None for binary value extraction")
                return ""
            result = clean_llm_response(content)

            print(f"      LLM binary value extraction: '{result}'")

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND"]:
                return ""
            if result and result.lower() in sentence.lower():
                return result
            else:
                print(f"      LLM extracted binary text not found in sentence: '{result}'")
                return ""

        except Exception as e:
            print(f"      Exception in LLM binary value extraction: {e}")
            return ""

    def query_local_llm_for_misc_value_extraction(self, sentence: str, field_name: str, field_value: str) -> str:
        """
        Query a local LLM to extract text indicating a miscellaneous character value in the sentence.
        Handles special characters like -, /, *, #, @, etc. that may appear in various forms.
        Returns the exact text portion, or empty string if not found.
        """
        readable_chars = []
        for char in field_value:
            if char == '-':
                readable_chars.append('dash/hyphen')
            elif char == '/':
                readable_chars.append('forward slash')
            elif char == '\\':
                readable_chars.append('backslash')
            elif char == '*':
                readable_chars.append('asterisk')
            elif char == '#':
                readable_chars.append('hash/pound')
            elif char == '@':
                readable_chars.append('at symbol')
            elif char == '&':
                readable_chars.append('ampersand')
            elif char == '%':
                readable_chars.append('percent')
            elif char == '$':
                readable_chars.append('dollar sign')
            elif char == '!':
                readable_chars.append('exclamation mark')
            elif char == '?':
                readable_chars.append('question mark')
            elif char == '.':
                readable_chars.append('period/dot')
            elif char == ',':
                readable_chars.append('comma')
            elif char == ':':
                readable_chars.append('colon')
            elif char == ';':
                readable_chars.append('semicolon')
            elif char == '_':
                readable_chars.append('underscore')
            elif char == '+':
                readable_chars.append('plus sign')
            elif char == '=':
                readable_chars.append('equals sign')
            elif char == '~':
                readable_chars.append('tilde')
            elif char == '`':
                readable_chars.append('backtick')
            elif char == '^':
                readable_chars.append('caret')
            elif char in '()':
                readable_chars.append('parenthesis')
            elif char in '[]':
                readable_chars.append('bracket')
            elif char in '{}':
                readable_chars.append('brace/curly bracket')
            elif char == '|':
                readable_chars.append('pipe/vertical bar')
            elif char == '<':
                readable_chars.append('less than')
            elif char == '>':
                readable_chars.append('greater than')
            elif char in '\'"':
                readable_chars.append('quote')
            elif char == '\u2013' or char == '\u2014':
                readable_chars.append('en-dash/em-dash')
            elif char == '\u2026':
                readable_chars.append('ellipsis')
            elif char == '\u2022':
                readable_chars.append('bullet point')
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
            - Described in words: "dash", "hyphen", "slash", "asterisk", "hash", "at symbol", etc.
            - As a placeholder indicator: "marked with", "indicated by", "represented by", "denoted by"
            - As a separator description: "separated by", "delimited by"
            - As absence/placeholder: "placeholder", "marker", "symbol", "separator"
            2. CRITICAL: Misc characters like "-" or "+" are OFTEN MISREPRESENTED as null/missing values in generated text:
            - "-" is commonly written as: "not specified", "unspecified", "not available", "unavailable", "unknown", "missing", "not provided", "not recorded", "none", "absent", "no data", "not applicable", "N/A"
            - "+" is commonly written as: "positive", "plus", "added", "additional", "and more", "extra"
            - Other misc chars may also appear as null-like text if the original generator treated them as missing data
            3. Common descriptions for misc characters:
            - "-" → "dash", "hyphen", "minus sign", "en-dash", "em-dash", OR any null-like phrase (see above)
            - "+" → "plus", "plus sign", "positive", "addition", "added"
            - "/" → "slash", "forward slash", "solidus"
            - "*" → "asterisk", "star"
            - "#" → "hash", "pound sign", "number sign", "hashtag"
            - "@" → "at", "at symbol", "at sign"
            - "&" → "ampersand", "and sign"
            - "..." or "\u2026" → "ellipsis", "dots"
            4. Find the EXACT text span in the sentence that represents this character value (including null-like phrases for "-" and "+")
            5. Return ONLY the matched text from the sentence - no explanations, no markdown, no quotes
            6. If the miscellaneous character value is NOT present in the sentence, return "NOT_FOUND"
            7. Do NOT add any text that's not in the original sentence

            Example 1:
            Sentence: "The separator used is a forward slash between the values."
            Field name: "Delimiter"
            Field value: "/"
            Response: forward slash

            Example 2:
            Sentence: "The entry is marked with a dash to indicate missing data."
            Field name: "Status"
            Field value: "-"
            Response: dash

            Example 3:
            Sentence: "The race of the superhero is not specified."
            Field name: "race"
            Field value: "-"
            Response: not specified

            Example 4:
            Sentence: "This field contains the asterisk symbol for special items."
            Field name: "Marker"
            Field value: "*"
            Response: asterisk symbol

            Example 5:
            Sentence: "The test result is positive for this patient."
            Field name: "Result"
            Field value: "+"
            Response: positive

            Example 6:
            Sentence: "The Admission indicates that the patient was admitted for inpatient care, which is crucial for understanding their treatment needs."
            Field name: "Admission"
            Field value: "+"
            Response: admitted for inpatient care

            Example 7:
            Sentence: "The patient has been hospitalized and is receiving treatment."
            Field name: "Status"
            Field value: "+"
            Response: hospitalized

            Example 8:
            Sentence: "The enrollment is confirmed for this program."
            Field name: "Enrollment"
            Field value: "+"
            Response: confirmed

            Example 9:
            Sentence: "The record shows standard processing."
            Field name: "Status"
            Field value: "-"
            Response: NOT_FOUND

            Your response (only the matched text or NOT_FOUND):"""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. You identify exact text spans that represent or describe miscellaneous character values like dashes, slashes, asterisks, and other special symbols. You return ONLY the matched text with no additional formatting or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"      LLM returned no response for misc value extraction")
                return ""

            message = response.choices[0].message
            content = message.content
            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            if content is None:
                print(f"      LLM returned None for misc value extraction")
                return ""
            result = clean_llm_response(content)

            print(f"      LLM misc value extraction: '{result}'")

            if result.upper() in ["NOT_FOUND", "NOT FOUND", "NOTFOUND"]:
                return ""
            if result and result.lower() in sentence.lower():
                return result
            else:
                print(f"      LLM extracted misc text not found in sentence: '{result}'")
                return ""

        except Exception as e:
            print(f"      Exception in LLM misc value extraction: {e}")
            return ""

    def mold_sentence_into_narrative(self, old_sentence: str, new_sentence: str, narrative_text: str) -> str:
        """Replace a specific sentence in the narrative with a corrected version, context-aware."""
        old_hash = TemplateGenerator.extract_hash(old_sentence)
        if old_hash and f'(Hash: {old_hash})' in narrative_text:
            delimited_block = self._find_delimited_block_by_hash(narrative_text, old_hash)
            if delimited_block:
                new_block = f"| {new_sentence} |"
                return narrative_text.replace(delimited_block, new_block)
            old_in_narrative = self._find_sentence_by_hash(narrative_text, old_hash)
            if old_in_narrative:
                return narrative_text.replace(old_in_narrative, new_sentence)

        prompt = f"""You are given a narrative document and must replace ONE specific sentence with a corrected version. Preserve the rest of the document exactly.

ORIGINAL SENTENCE TO REMOVE:
"{old_sentence}"

CORRECTED REPLACEMENT SENTENCE:
"{new_sentence}"

NARRATIVE DOCUMENT:
{narrative_text}

YOUR TASK:
1. Find the original sentence (or its closest match) in the narrative
2. Replace ONLY that sentence with the corrected sentence — do not modify any other content
3. Preserve ALL (Hash: XXXXXXXX) tags exactly as written — do not remove, modify, or relocate any hash
4. Preserve ALL pipe delimiters (|) exactly as written — sentences may be wrapped as | sentence | or | sentence. (Hash: xxx) |; leave all pipes unchanged
5. Return the FULL updated narrative

Return ONLY the updated narrative text."""

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": "You perform targeted sentence replacements. Replace only the specified sentence. Preserve all (Hash: ...) tags and | pipe delimiters exactly. Do not modify any other content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            track_openai_response(response, get_local_llm_model())
            if response and response.choices:
                content = response.choices[0].message.content
                if content and len(content) > len(narrative_text) * 0.5:
                    return content.strip()
        except Exception as e:
            print(f"      Exception in mold_sentence_into_narrative: {e}")

        if old_hash and f'(Hash: {old_hash})' in narrative_text:
            delimited_block = self._find_delimited_block_by_hash(narrative_text, old_hash)
            if delimited_block:
                new_block = f"| {new_sentence} |"
                return narrative_text.replace(delimited_block, new_block)
            old_in_narrative = self._find_sentence_by_hash(narrative_text, old_hash)
            if old_in_narrative:
                return narrative_text.replace(old_in_narrative, new_sentence)
        return narrative_text

    def _find_sentence_by_hash(self, narrative_text: str, hash_value: str) -> Optional[str]:
        """Extract the sentence containing a specific hash. Looks LEFT of the hash until the nearest |."""
        hash_tag = f'(Hash: {hash_value})'
        idx = narrative_text.find(hash_tag)
        if idx == -1:
            return None
        start = narrative_text.rfind('|', 0, idx)
        if start == -1:
            start = narrative_text.rfind('.', 0, idx)
            start = start + 1 if start != -1 else 0
        else:
            start = start + 1
        end = idx + len(hash_tag)
        return narrative_text[start:end].strip()

    def _find_delimited_block_by_hash(self, narrative_text: str, hash_value: str) -> Optional[str]:
        """Extract the full delimited block | sentence. (Hash: xxx) | for replacement."""
        hash_tag = f'(Hash: {hash_value})'
        idx = narrative_text.find(hash_tag)
        if idx == -1:
            return None
        start = narrative_text.rfind('|', 0, idx)
        if start == -1:
            return None
        end = narrative_text.find('|', idx + len(hash_tag))
        if end == -1:
            end = len(narrative_text)
        else:
            end = end + 1
        return narrative_text[start:end]

    def rewrite_sentence_for_detectability(
        self,
        sentence: str,
        column_name: str,
        column_value: str,
        field_type: str,
        descriptor: str,
        database: str = "",
        table: str = "",
        preserve_hash: str = None,
    ) -> str:
        """
        Rewrite a sentence to make the column name and value more easily detectable,
        while preserving the original generation mode (explicit/implicit for null/binary).
        Used when LLM fallback returns NO twice - correction instead of fresh regeneration.

        Mirrors the rules, warnings, and boundaries from generate_sentence_for_column in
        generate_all_templates.py.
        """
        field_value_str = str(column_value).strip()

        if field_type == "NULL":
            if self.null_mode == "explicit":
                mode_context = (
                    "Generation mode: EXPLICIT NULL. The value must be conveyed using the literal word NULL. "
                    "Do NOT use phrases like 'not specified' or 'not available' - use NULL exactly."
                )
            else:
                mode_context = (
                    "Generation mode: IMPLICIT NULL. The value must be conveyed using natural language "
                    "(e.g. 'not specified', 'not available', 'not provided', 'unknown'). "
                    "Do NOT use the word NULL or None in the sentence."
                )
        elif field_type == "BINARY":
            is_true = field_value_str in ['1', 'true', 'True', 'TRUE']
            if self.binary_mode == "explicit":
                mode_context = (
                    f"Generation mode: EXPLICIT BINARY. The value is {field_value_str}. "
                    "Use the literal value 0 or 1 in the sentence. Do NOT convert to yes/no or true/false."
                )
            else:
                mode_context = (
                    f"Generation mode: IMPLICIT BINARY. The state is: {'positive/yes/true' if is_true else 'negative/no/false'}. "
                    "Use natural language (e.g. 'is a', 'operates as' for positive; 'is not', 'does not have' for negative). "
                    "Do NOT include the raw 0 or 1 value in the sentence."
                )
        elif field_type == "MISC":
            mode_context = (
                f"Generation mode: MISC/SYMBOL. The value is '{field_value_str}' (special character). "
                "It may appear as the character itself or described in words (e.g. 'dash', 'hyphen' for '-')."
            )
        else:
            mode_context = (
                f"Generation mode: STANDARD. The value is '{field_value_str}'. "
                "Mention the value and column name directly - use exactly, no substitutions or abbreviations."
            )

        common_rules = """
RULES (same as original generation - MUST be followed):
1. Do NOT include any natural language that suggests the data comes from a column, field, database, table, or row. The sentence must read as ordinary prose — not detectable as coming from structured data.
2. Return only the rewritten sentence with no markdown, explanations, or other text.
3. Preserve the same generation mode - do not switch between explicit and implicit representations."""

        prompt = f"""The following sentence was synthetically generated based on database row data. However, the sentence contains heavy implicit logic which makes it unclear whether the sentence represents the column value, making it difficult for parsers to detect.

ORIGINAL SENTENCE:
"{sentence}"

COLUMN TO CLARIFY:
- Column name: {column_name}
- Original value (database format): {field_value_str}
- Field type: {field_type}
- Context: {descriptor}
- {mode_context}

YOUR TASK: Rewrite this sentence to MAXIMIZE DETECTABILITY. Prioritize simplicity and convergence:
1. PREFER SHORTER, SIMPLER SENTENCES over long or convoluted ones. Simplicity improves parser detection.
2. INCLUDE THE COLUMN NAME (or its natural language equivalent) directly in the sentence. Use "{column_name}" or a clear human-readable form (e.g. "extension" for Ext, "phone extension" for Ext, "EIL code" for EILCode). The column name MUST appear in recognizable form.
3. STATE THE VALUE CLEARLY. Avoid vague references like "such as" or "for example" — name the value or its absence explicitly.
4. AVOID complex subordinate clauses, hedging language, or indirect phrasing. Prefer direct statements: "X is Y" over "It may be the case that X could be considered Y."
5. The sentence remains in the same generation mode (explicit vs implicit).
6. Preserve the core meaning but simplify aggressively for detectability.
{common_rules}

Rewrite the sentence (keep it simple and direct):"""

        system_message = (
            "You rewrite sentences to improve detectability. PRIORITIZE: short, simple sentences; direct inclusion of the column name in natural language; clear statement of the value. "
            "Avoid complexity, hedging, or indirect phrasing. Output must never mention columns, fields, databases, or tables. "
            "Return ONLY the rewritten sentence."
        )

        try:
            response = self.local_llm_client.chat.completions.create(
                model=get_local_llm_model(),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500,
            )

            track_openai_response(response, get_local_llm_model())
            if not response or not response.choices:
                print(f"      LLM returned no response for sentence rewrite")
                return sentence

            message = response.choices[0].message
            content = message.content
            if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
            if content is None:
                print(f"      LLM returned None for sentence rewrite")
                return sentence

            result = clean_llm_response(content)
            if result and len(result) > 10:
                result = TemplateGenerator.strip_hash(result)
                if preserve_hash:
                    result = TemplateGenerator.append_hash(result, preserve_hash)
                print(f"      Rewritten sentence: {result[:80]}...")
                return result
            else:
                print(f"      Rewrite produced invalid output, keeping original")
                return sentence

        except Exception as e:
            print(f"      Exception in sentence rewrite: {e}")
            return sentence
