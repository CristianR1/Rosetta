"""Detection and replacement methods for NarrativeParsingAnalyzer."""
import re
import string
from typing import Dict, List, Tuple
from .text_utils import clean_field_string, is_misc_value, is_date_value
from .config import MISC_CHARACTERS


class NarrativeDetectionMixin:
    """Mixin providing field detection and replacement.

    Expects the host class to provide:
        self.null_mode, self.binary_mode
        self.null_patterns, self.binary_true_patterns, self.binary_false_patterns
        self.query_local_llm_for_* methods (from NarrativeLLMMixin)
    """

    def check_name_in_sentence(self, sentence: str, field_name: str, field_value: str = None, use_llm_fallback: bool = False, data_fields: Dict[str, str] = None) -> Tuple[bool, str]:
        """
        Check if a field name or value appears in a sentence.
        Returns (found, match_type) where match_type indicates match strength:
        1. 'field_name_direct': Field name found via direct regex matching
        2. 'field_name_word_by_word': Field name found via word-by-word detection
        3. 'field_name_llm': Field name found via LLM discernment
        4. 'field_value': Only field value found, not name
        5. '': Not found

        For digit values, performs strict standalone matching and checks if the digit
        is part of another column name to avoid false attributions.
        """
        field_name_cleaned = clean_field_string(field_name)
        field_value_cleaned = clean_field_string(str(field_value)) if field_value else None
        if not field_name_cleaned or len(field_name_cleaned) < 2:
            field_name_cleaned = field_name
        field_name_found = False
        field_name_patterns_to_check = []
        if field_name != field_name_cleaned:
            field_name_patterns_to_check.append(field_name)

        if '(' in field_name and ')' in field_name:
            field_name_no_parens = re.sub(r'[()]', '', field_name).strip()
            if field_name_no_parens not in field_name_patterns_to_check:
                field_name_patterns_to_check.append(field_name_no_parens)

        if '(' in field_name_cleaned and ')' in field_name_cleaned:
            field_name_clean = re.sub(r'[()]', '', field_name_cleaned).strip()
            if field_name_clean not in field_name_patterns_to_check:
                field_name_patterns_to_check.append(field_name_clean)
        else:
            if field_name_cleaned not in field_name_patterns_to_check:
                field_name_patterns_to_check.append(field_name_cleaned)

        for pattern in field_name_patterns_to_check.copy():
            if pattern.endswith('y'):
                plural_pattern = pattern[:-1] + 'ies'
            elif pattern.endswith(('s', 'sh', 'ch', 'x', 'z')):
                plural_pattern = pattern + 'es'
            else:
                plural_pattern = pattern + 's'
            if plural_pattern not in field_name_patterns_to_check:
                field_name_patterns_to_check.append(plural_pattern)

        for pattern in field_name_patterns_to_check:
            simple_word_pattern = r'\b' + re.escape(pattern) + r"'?s?\b"
            if re.search(simple_word_pattern, sentence, re.IGNORECASE):
                field_name_found = True
                print(f"      Found field NAME via word boundary matching: '{pattern}'")
                return True, 'field_name_direct'

            try:
                punctuation_no_dash = string.punctuation.replace("-", "")
                escaped_punct = ''.join(['\\' + c if c in r'\[]^-' else c for c in punctuation_no_dash])
                field_name_pattern = r"(?:(?<=^)|(?<=[\s" + escaped_punct + r"]))" + re.escape(pattern) + r"'?s?(?:(?=$)|(?=[\s" + escaped_punct + r"]))"
                if re.search(field_name_pattern, sentence, re.IGNORECASE):
                    field_name_found = True
                    print(f"      Found field NAME via pattern matching: '{pattern}'")
                    return True, 'field_name_direct'
            except re.error:
                pass

        if not field_name_found:
            field_names_to_check = [field_name]
            if field_name_cleaned != field_name:
                field_names_to_check.append(field_name_cleaned)

            for name_to_check in field_names_to_check:
                if len(name_to_check.split()) > 1:
                    field_words = [word.strip('()') for word in name_to_check.split() if len(word.strip('()')) > 2]
                    all_words_found = True
                    for word in field_words:
                        word_pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
                        if not re.search(word_pattern, sentence, re.IGNORECASE):
                            all_words_found = False
                            break
                    if all_words_found:
                        print(f"      Found field NAME via word-by-word matching: '{name_to_check}' (all words present)")
                        field_name_found = True
                        return True, 'field_name_word_by_word'

        field_value_str = field_value_cleaned if field_value_cleaned else str(field_value) if field_value else ""
        if field_value_str:
            is_numeric = False
            numeric_value = None
            try:
                if field_value_str.replace('.', '', 1).isdigit():
                    is_numeric = True
                    numeric_value = float(field_value_str)
            except (ValueError, AttributeError):
                pass

            if is_numeric:
                punctuation_no_dash = string.punctuation.replace("-", "")

                strict_digit_patterns = []
                strict_digit_patterns.append(
                    (field_value_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))')
                )

                if numeric_value == int(numeric_value):
                    int_str = str(int(numeric_value))
                    if int_str != field_value_str:
                        strict_digit_patterns.append(
                            (int_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(int_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))')
                        )

                    if numeric_value >= 1000:
                        comma_formatted = f"{int(numeric_value):,}"
                        strict_digit_patterns.append(
                            (comma_formatted, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(comma_formatted) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))')
                        )

                if 0 < numeric_value < 1:
                    percentage = numeric_value * 100
                    perc_formats = [
                        (f"{percentage}%", r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(f"{percentage}") + r'%' + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'),
                        (f"{percentage:.2f}%", r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(f"{percentage:.2f}") + r'%' + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'),
                    ]
                    strict_digit_patterns.extend(perc_formats)

                for pattern_value, pattern in strict_digit_patterns:
                    all_matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
                    if all_matches:
                        safe_match_found = False
                        if data_fields:
                            column_context_patterns = []
                            for check_field_name in data_fields.keys():
                                pattern_value_lower = pattern_value.lower()
                                words = check_field_name.split()
                                digit_word_index = None
                                for i, word in enumerate(words):
                                    word_clean = re.sub(r'[^\w]', '', word)
                                    if word_clean == pattern_value or word == pattern_value:
                                        digit_word_index = i
                                        break
                                    if pattern_value in word or pattern_value_lower in word.lower():
                                        digit_word_index = i
                                        break

                                if digit_word_index is not None:
                                    left_word = words[digit_word_index-1] if digit_word_index > 0 else None
                                    right_word = words[digit_word_index+1] if digit_word_index < len(words) - 1 else None
                                    digit_word = words[digit_word_index]

                                    if left_word and right_word:
                                        column_context_patterns.append((
                                            check_field_name,
                                            r'(?<!\w)' + re.escape(left_word) + r'\s+' + re.escape(pattern_value) + r'\s+' + re.escape(right_word) + r'(?!\w)'
                                        ))
                                    if left_word:
                                        column_context_patterns.append((
                                            check_field_name,
                                            r'(?<!\w)' + re.escape(left_word) + r'\s+' + re.escape(pattern_value) + r'(?!\w)'
                                        ))
                                    if right_word:
                                        column_context_patterns.append((
                                            check_field_name,
                                            r'(?<!\w)' + re.escape(pattern_value) + r'\s+' + re.escape(right_word) + r'(?!\w)'
                                        ))

                                    digit_word_escaped = re.escape(digit_word)
                                    column_context_patterns.append((
                                        check_field_name,
                                        r'(?<!\w)' + digit_word_escaped + r'(?!\w)'
                                    ))

                            for match in all_matches:
                                match_start = match.start()
                                match_end = match.end()
                                is_part_of_column_name = False

                                for ctx_field_name, ctx_pattern in column_context_patterns:
                                    window_start = max(0, match_start - 50)
                                    window_end = min(len(sentence), match_end + 50)
                                    window_text = sentence[window_start:window_end]
                                    if re.search(ctx_pattern, window_text, re.IGNORECASE):
                                        is_part_of_column_name = True

                                        if ctx_field_name == field_name:
                                            field_name_words = ctx_field_name.split()
                                            field_name_found_in_sentence = False

                                            for word in field_name_words:
                                                if len(word) > 2:
                                                    word_pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
                                                    word_matches = list(re.finditer(word_pattern, sentence, re.IGNORECASE))
                                                    for wm in word_matches:
                                                        if abs(wm.start() - match_start) < 20 or abs(wm.end() - match_end) < 20:
                                                            field_name_found_in_sentence = True
                                                            if (match_start >= wm.start() - 5 and match_end <= wm.end() + 5):
                                                                is_part_of_column_name = True
                                                                break
                                                            else:
                                                                is_part_of_column_name = False
                                                    if is_part_of_column_name:
                                                        break

                                            if field_name_found_in_sentence and not is_part_of_column_name:
                                                pass

                                        if is_part_of_column_name and ctx_field_name != field_name:
                                            print(f"      Digit '{pattern_value}' at position {match_start}-{match_end} appears to be part of column name '{ctx_field_name}' (pattern: {ctx_pattern}).")
                                            break
                                        elif is_part_of_column_name and ctx_field_name == field_name:
                                            print(f"      Digit '{pattern_value}' at position {match_start}-{match_end} appears to be part of current field's name '{ctx_field_name}' in sentence.")
                                            break

                                if not is_part_of_column_name:
                                    safe_match_found = True
                                    print(f"      Found safe instance of field value (digit) '{pattern_value}' at position {match_start}-{match_end}")
                                    break

                            if safe_match_found:
                                return True, 'field_value'
                            else:
                                print(f"      All instances of digit '{pattern_value}' appear to be part of column name contexts. Rejecting match for '{field_name}'.")
                                continue
                        else:
                            print(f"      Found field value (digit) in sentence as last resort")
                            return True, 'field_value'
            else:
                punctuation_no_dash = string.punctuation.replace("-", "")
                value_patterns = []

                field_value_clean = field_value_str
                if '(' in field_value_str and ')' in field_value_str:
                    field_value_clean = re.sub(r'[()]', '', field_value_str).strip()

                value_patterns.append(r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))')
                if field_value_clean != field_value_str:
                    value_patterns.append(r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_clean) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))')

                for value_pattern in value_patterns:
                    value_match = re.search(value_pattern, sentence, re.IGNORECASE)
                    if value_match:
                        print(f"      Found field VALUE (string) in sentence as last resort, moving forward to replacement")
                        return True, 'field_value'

        if not field_name_found and use_llm_fallback:
            field_names_to_try = [field_name]
            if field_name_cleaned != field_name:
                field_names_to_try.append(field_name_cleaned)

            for name_to_try in field_names_to_try:
                print(f"      Querying local LLM for field '{name_to_try}' (with value: {field_value_str}) detection")
                other_field_names = list(data_fields.keys()) if data_fields else None
                if other_field_names and name_to_try in other_field_names:
                    other_field_names = [fn for fn in other_field_names if fn != name_to_try]
                llm_detected = self.query_local_llm_for_field_detection(sentence, name_to_try, field_value_str, other_field_names)

                if llm_detected:
                    print(f"      LLM confirmed field '{name_to_try}' is eligible for replacement")
                    return True, 'field_name_llm'
                else:
                    print(f"      LLM confirmed field '{name_to_try}' is NOT eligible for replacement (attempt 1)")
                    print(f"      Retrying LLM detection for field '{name_to_try}' (attempt 2)")
                    llm_detected = self.query_local_llm_for_field_detection(sentence, name_to_try, field_value_str, other_field_names)
                    if llm_detected:
                        print(f"      LLM confirmed field '{name_to_try}' is eligible for replacement (on retry)")
                        return True, 'field_name_llm'
                    print(f"      LLM confirmed field '{name_to_try}' is NOT eligible for replacement (attempt 2)")

            return False, ''

        return False, ''

    def check_field_in_sentence(self, sentence: str, field_name: str, field_value: str, field_metadata: Dict[str, str], type_override: str = None, use_llm_fallback: bool = False, data_fields: Dict[str, str] = None) -> Tuple[bool, str, str, str]:
        """
        Check if a field appears in a sentence, considering both original and natural names for Y/N/T/F fields.
        Returns (found, replacement_value, matched_field_name, match_type)
        where match_type indicates match strength for prioritization.
        """
        field_value_original = str(field_value)
        field_value_cleaned = clean_field_string(field_value_original)
        field_value_str = field_value_original

        present, match_type = self.check_name_in_sentence(sentence, field_name, field_value_str, use_llm_fallback=use_llm_fallback, data_fields=data_fields)

        if present:
            if type_override:
                field_type = type_override
            else:
                field_type = field_metadata.get(field_name, "STANDARD")

            if field_type == "BINARY" and field_value_str in ['0', '1']:
                field_name_clean = field_name.replace('(Y/N)', '').replace('(T/F)', '').strip().lower()
                sentence_lower = sentence.lower()

                if match_type and match_type.startswith('field_name'):
                    direct_binary_pattern = r'(?:(?<=^)|(?<=\s))' + re.escape(field_value_str) + r'(?:(?=$)|(?=\s|[,.:;!?]))'
                    direct_match = re.search(direct_binary_pattern, sentence)
                    if direct_match:
                        print(f"      Found direct binary value '{field_value_str}' in sentence")
                        return True, direct_match.group(), field_name, match_type

                    if field_value_str == '1':
                        direct_text_patterns = [r'\byes\b', r'\btrue\b', r'\bactive\b', r'\benabled\b']
                    else:
                        direct_text_patterns = [r'\bno\b', r'\bfalse\b', r'\binactive\b', r'\bdisabled\b']

                    for pattern in direct_text_patterns:
                        match = re.search(pattern, sentence_lower, re.IGNORECASE)
                        if match:
                            return True, match.group(), field_name, match_type

                    positive_patterns = [
                        r'(?:as\s+a|is\s+a|is\s+an)\s+' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                        r'operating\s+(?:as\s+)?(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                    ]
                    negative_patterns = [
                        r'(?:\w+\s+)*is\s+not\s+(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                        r'(?:\w+\s+)*not\s+(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                    ]
                    patterns_to_check = positive_patterns if field_value_str == '1' else negative_patterns
                    for pattern in patterns_to_check:
                        match = re.search(pattern, sentence_lower, re.IGNORECASE)
                        if match:
                            return True, match.group(), field_name, match_type

                    return True, field_value_str, field_name, match_type

                direct_binary_pattern = r'(?:(?<=^)|(?<=\s))' + re.escape(field_value_str) + r'(?:(?=$)|(?=\s|[,.:;!?]))'
                direct_match = re.search(direct_binary_pattern, sentence)
                if direct_match:
                    print(f"      Found direct binary value '{field_value_str}' in sentence")
                    return True, direct_match.group(), field_name, 'field_value'

                if field_value_str == '1':
                    direct_text_patterns = [
                        r'\byes\b', r'\btrue\b', r'\bactive\b', r'\benabled\b',
                        r'\baffirmative\b', r'\bpositive\b', r'\bvalid\b'
                    ]
                else:
                    direct_text_patterns = [
                        r'\bno\b', r'\bfalse\b', r'\binactive\b', r'\bdisabled\b',
                        r'\bnegative\b', r'\binvalid\b'
                    ]

                for pattern in direct_text_patterns:
                    match = re.search(pattern, sentence_lower, re.IGNORECASE)
                    if match:
                        print(f"      Found direct binary text '{match.group()}' for value '{field_value_str}'")
                        return True, match.group(), field_name, 'field_value'

                positive_patterns = [
                    r'(?:as\s+a|is\s+a|is\s+an)\s+' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                    r'operating\s+(?:as\s+)?(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                    r'functions\s+(?:as\s+)?(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                    r'operates\s+as\s+a\s+' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                    r'(?<!\w)a\s+' + re.escape(field_name_clean) + r'(?=\s|$|[^\w])',
                    r'(?:as\s+a|is\s+a|is\s+an)\s+' + re.escape(field_name_clean) + r'\s+status(?=\s|$|[^\w])',
                    r'(?<!\w)' + re.escape(field_name_clean) + r'\s+(?:classification|status)(?=\s|$|[^\w])',
                    r'(?<!\w)' + re.escape(field_name_clean) + r'.*?compliant(?=\s|$|[^\w])'
                ]

                negative_patterns = [
                    r'(?:\w+\s+)*is\s+not\s+(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                    r'(?:\w+\s+)*not\s+(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                    r'(?:\w+\s+)*not\s+operating\s+as\s+(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                    r'(?:\w+\s+)*does\s+not\s+function\s+as\s+(?:a\s+|an\s+)?' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                    r'(?:\w+\s+)*non\s+' + re.escape(field_name_clean) + r'(?:\s+\w+)*',
                    r'(?<!\w)' + re.escape(field_name_clean) + r'.*?(?:not\s+compliant|non\s+compliant|incompliant)(?=\s|$|[^\w])'
                ]

                patterns_to_check = positive_patterns if field_value_str == '1' else negative_patterns

                for pattern in patterns_to_check:
                    match = re.search(pattern, sentence_lower, re.IGNORECASE)
                    if match:
                        matched_text = match.group()
                        return True, matched_text, field_name, 'field_name_direct'

                if use_llm_fallback:
                    print(f"      Querying LLM to extract binary indicator")
                    llm_extracted_text = self.query_local_llm_for_binary_value_extraction(sentence, field_name, field_value_str)
                    if llm_extracted_text:
                        print(f"      LLM extracted binary indicator: '{llm_extracted_text}'")
                        return True, llm_extracted_text, field_name, 'field_name_llm'

                return False, field_value_str, field_name, ''

            elif field_type == "NULL" and (field_value_str.upper() in ["NULL", "NONE"] or field_value_str.strip() == ""):
                if match_type and match_type.startswith('field_name'):
                    sentence_lower = sentence.lower()
                    null_patterns = ['not specified', 'unspecified', 'is not specified', 'none', 'null']
                    for pattern in null_patterns:
                        if pattern in sentence_lower:
                            return True, pattern, field_name, match_type
                    return True, 'not specified', field_name, match_type

                sentence_lower = sentence.lower()
                null_patterns = [
                    'not specified', 'is unspecified', 'remains unspecified', 'is not detailed',
                    'unspecified', 'no specified', 'an unspecified', 'not detailed',
                    'lacks a specified', 'has not been specified', 'does not specify',
                    'remains not specified', 'is not specified', 'is absent', 'none', 'null',
                    'not available', 'unavailable', 'unknown', 'not provided', 'not given',
                    'missing', 'not recorded', 'not listed', 'not mentioned', 'not defined'
                ]

                for pattern in null_patterns:
                    if pattern in sentence_lower:
                        return True, pattern, field_name, 'field_value'

                if use_llm_fallback:
                    print(f"      Last resort for null value: Querying LLM to extract null indicator")
                    llm_extracted_text = self.query_local_llm_for_null_value_extraction(sentence, field_name, field_value_str)
                    if llm_extracted_text:
                        print(f"      LLM extracted null indicator: '{llm_extracted_text}'")
                        return True, llm_extracted_text, field_name, 'field_name_llm'

                return False, "[ERROR]", field_name, ''

            elif field_type == "MISC" and is_misc_value(field_value_str):
                if match_type and match_type.startswith('field_name'):
                    sentence_lower = sentence.lower()
                    escaped_value = re.escape(field_value_str)
                    direct_match = re.search(escaped_value, sentence)
                    if direct_match:
                        return True, direct_match.group(), field_name, match_type
                    misc_word_patterns = {
                        '-': ['not specified', 'unspecified', 'not available'],
                        '+': ['plus', 'positive', 'added', 'additional'],
                    }
                    if field_value_str in misc_word_patterns:
                        for word_pattern in misc_word_patterns[field_value_str]:
                            if word_pattern in sentence_lower:
                                return True, word_pattern, field_name, match_type
                    return True, field_value_str, field_name, match_type

                sentence_lower = sentence.lower()
                escaped_value = re.escape(field_value_str)
                direct_match = re.search(escaped_value, sentence)
                if direct_match:
                    return True, direct_match.group(), field_name, 'field_value'

                misc_word_patterns = {
                    '-': ['dash', 'hyphen', 'minus', 'en-dash', 'em-dash',
                          'not specified', 'unspecified', 'not available', 'unavailable',
                          'unknown', 'missing', 'not provided', 'not recorded', 'none',
                          'absent', 'no data', 'not applicable', 'n/a', 'not given',
                          'not listed', 'not mentioned', 'not defined', 'is absent'],
                    '--': ['double dash', 'double hyphen', 'em-dash'],
                    '---': ['triple dash', 'horizontal rule'],
                    '/': ['slash', 'forward slash', 'solidus'],
                    '\\': ['backslash', 'back slash'],
                    '*': ['asterisk', 'star'],
                    '**': ['double asterisk', 'bold marker'],
                    '#': ['hash', 'pound', 'number sign', 'hashtag'],
                    '@': ['at', 'at symbol', 'at sign'],
                    '&': ['ampersand', 'and symbol'],
                    '%': ['percent', 'percentage'],
                    '$': ['dollar', 'dollar sign'],
                    '!': ['exclamation', 'exclamation mark', 'bang'],
                    '?': ['question mark', 'question'],
                    '.': ['period', 'dot', 'full stop'],
                    '..': ['double dot', 'range'],
                    '...': ['ellipsis', 'dots', 'three dots'],
                    '\u2026': ['ellipsis', 'dots'],
                    ',': ['comma'],
                    ':': ['colon'],
                    ';': ['semicolon', 'semi-colon'],
                    '_': ['underscore', 'underline'],
                    '__': ['double underscore'],
                    '+': ['plus', 'plus sign', 'positive', 'added', 'additional',
                          'and more', 'extra', 'addition', 'admitted', 'inpatient',
                          'hospitalized', 'enrolled', 'registered', 'confirmed',
                          'approved', 'accepted', 'included', 'present', 'yes'],
                    '=': ['equals', 'equal sign', 'equals sign'],
                    '~': ['tilde', 'approximately'],
                    '`': ['backtick', 'grave accent'],
                    '^': ['caret', 'circumflex'],
                    '()': ['parentheses', 'brackets'],
                    '(': ['open parenthesis', 'left parenthesis', 'opening bracket'],
                    ')': ['close parenthesis', 'right parenthesis', 'closing bracket'],
                    '[]': ['square brackets', 'brackets'],
                    '[': ['open bracket', 'left bracket'],
                    ']': ['close bracket', 'right bracket'],
                    '{}': ['curly braces', 'braces'],
                    '{': ['open brace', 'left brace'],
                    '}': ['close brace', 'right brace'],
                    '|': ['pipe', 'vertical bar', 'bar'],
                    '<': ['less than', 'left angle bracket'],
                    '>': ['greater than', 'right angle bracket'],
                    '<>': ['angle brackets', 'diamond'],
                    "'": ['apostrophe', 'single quote'],
                    '"': ['quote', 'double quote', 'quotation mark'],
                    '\u2013': ['en-dash', 'dash', 'not specified', 'unspecified'],
                    '\u2014': ['em-dash', 'long dash', 'not specified', 'unspecified'],
                    '\u2022': ['bullet', 'bullet point'],
                    '\u00b7': ['middle dot', 'interpunct'],
                    '\u00b0': ['degree', 'degree symbol'],
                    '\u00b1': ['plus-minus', 'plus or minus'],
                    '\u00d7': ['times', 'multiplication'],
                    '\u00f7': ['division', 'divide'],
                    'N/A': ['not applicable', 'n/a', 'na'],
                    'n/a': ['not applicable', 'n/a', 'na'],
                    'TBD': ['to be determined', 'tbd'],
                    'TBA': ['to be announced', 'tba'],
                }

                if field_value_str in misc_word_patterns:
                    for word_pattern in misc_word_patterns[field_value_str]:
                        if word_pattern in sentence_lower:
                            return True, word_pattern, field_name, 'field_value'

                if use_llm_fallback:
                    print(f"      Querying LLM to extract misc indicator")
                    llm_extracted_text = self.query_local_llm_for_misc_value_extraction(sentence, field_name, field_value_str)
                    if llm_extracted_text:
                        print(f"      LLM extracted misc indicator: '{llm_extracted_text}'")
                        return True, llm_extracted_text, field_name, 'field_name_llm'

                return False, field_value_str, field_name, ''

            else:
                punctuation_no_dash = string.punctuation.replace("-", "")
                direct_value_patterns = []
                if len(field_value_str) == 1 and field_value_str.isdigit():
                    direct_value_patterns.append((field_value_str, r'(?:(?<=^)|(?<=\s))' + re.escape(field_value_str) + r'(?:(?=$)|(?=\s))'))
                elif field_value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                    try:
                        numeric_value = float(field_value_str)
                    except (ValueError, TypeError):
                        direct_value_patterns.append((field_value_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                    else:
                        direct_value_patterns.append((field_value_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                        if numeric_value == int(numeric_value):
                            int_str = str(int(numeric_value))
                            direct_value_patterns.append((int_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(int_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            int_val = int(numeric_value)
                            if int_val >= 0:
                                if 10 <= int_val % 100 <= 20:
                                    ordinal_suffix = "th"
                                else:
                                    last_digit = int_val % 10
                                    if last_digit == 1:
                                        ordinal_suffix = "st"
                                    elif last_digit == 2:
                                        ordinal_suffix = "nd"
                                    elif last_digit == 3:
                                        ordinal_suffix = "rd"
                                    else:
                                        ordinal_suffix = "th"

                                ordinal_str = f"{int_str}{ordinal_suffix}"
                                direct_value_patterns.append((ordinal_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(ordinal_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            if abs(numeric_value) >= 1000:
                                comma_formatted = f"{int(numeric_value):,}"
                                direct_value_patterns.append((comma_formatted, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(comma_formatted) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                        if '.' in field_value_str:
                            for decimal_places in range(1, 11):
                                rounded_value = round(numeric_value, decimal_places)
                                rounded_str = f"{rounded_value:.{decimal_places}f}"
                                direct_value_patterns.append((rounded_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(rounded_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                                rounded_str_no_trailing = str(rounded_value)
                                direct_value_patterns.append((rounded_str_no_trailing, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(rounded_str_no_trailing) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                        if 0 < abs(numeric_value) < 1:
                            percentage = numeric_value * 100
                            for decimal_places in range(0, 5):
                                if decimal_places == 0:
                                    perc_str = f"{percentage:.0f}%"
                                    perc_no_sign = f"{percentage:.0f}"
                                else:
                                    perc_str = f"{percentage:.{decimal_places}f}%"
                                    perc_no_sign = f"{percentage:.{decimal_places}f}"
                                direct_value_patterns.append((perc_str, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(perc_str) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                                direct_value_patterns.append((perc_no_sign, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(perc_no_sign) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                else:
                    direct_value_patterns.append((field_value_original, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_original) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                    if field_value_cleaned and field_value_cleaned != field_value_original:
                        direct_value_patterns.append((field_value_cleaned, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(field_value_cleaned) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                    for base_value in [field_value_original, field_value_cleaned]:
                        if base_value:
                            value_no_trailing_punct = base_value.rstrip(string.punctuation)
                            if value_no_trailing_punct != base_value and len(value_no_trailing_punct) > 2:
                                pattern_tuple = (value_no_trailing_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_no_trailing_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))')
                                if pattern_tuple not in direct_value_patterns:
                                    direct_value_patterns.append(pattern_tuple)

                    if ',' in field_value_str and not field_value_str.isdigit():
                        comma_count = field_value_str.count(',')
                        if comma_count == 1:
                            value_with_and = field_value_str.replace(',', ' and ')
                            value_with_and = re.sub(r'\s+', ' ', value_with_and).strip()
                            direct_value_patterns.append((value_with_and, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_and) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            value_with_or = field_value_str.replace(',', ' or ')
                            value_with_or = re.sub(r'\s+', ' ', value_with_or).strip()
                            direct_value_patterns.append((value_with_or, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_or) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            value_with_and_no_punct = value_with_and.rstrip(string.punctuation)
                            if value_with_and_no_punct != value_with_and and len(value_with_and_no_punct) > 2:
                                direct_value_patterns.append((value_with_and_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_and_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            value_with_or_no_punct = value_with_or.rstrip(string.punctuation)
                            if value_with_or_no_punct != value_with_or and len(value_with_or_no_punct) > 2:
                                direct_value_patterns.append((value_with_or_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_or_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))
                        else:
                            parts = field_value_str.split(',')

                            list_with_and = ', '.join(parts[:-1]) + ' and ' + parts[-1]
                            list_with_and = re.sub(r'\s+', ' ', list_with_and).strip()
                            direct_value_patterns.append((list_with_and, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_and) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_and_oxford = ', '.join(parts[:-1]) + ', and ' + parts[-1]
                            list_with_and_oxford = re.sub(r'\s+', ' ', list_with_and_oxford).strip()
                            direct_value_patterns.append((list_with_and_oxford, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_and_oxford) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_or = ', '.join(parts[:-1]) + ' or ' + parts[-1]
                            list_with_or = re.sub(r'\s+', ' ', list_with_or).strip()
                            direct_value_patterns.append((list_with_or, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_or) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_or_oxford = ', '.join(parts[:-1]) + ', or ' + parts[-1]
                            list_with_or_oxford = re.sub(r'\s+', ' ', list_with_or_oxford).strip()
                            direct_value_patterns.append((list_with_or_oxford, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_or_oxford) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            value_with_and_all = field_value_str.replace(',', ' and ')
                            value_with_and_all = re.sub(r'\s+', ' ', value_with_and_all).strip()
                            direct_value_patterns.append((value_with_and_all, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_and_all) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            value_with_or_all = field_value_str.replace(',', ' or ')
                            value_with_or_all = re.sub(r'\s+', ' ', value_with_or_all).strip()
                            direct_value_patterns.append((value_with_or_all, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_or_all) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_and_no_punct = list_with_and.rstrip(string.punctuation)
                            if list_with_and_no_punct != list_with_and and len(list_with_and_no_punct) > 2:
                                direct_value_patterns.append((list_with_and_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_and_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_and_oxford_no_punct = list_with_and_oxford.rstrip(string.punctuation)
                            if list_with_and_oxford_no_punct != list_with_and_oxford and len(list_with_and_oxford_no_punct) > 2:
                                direct_value_patterns.append((list_with_and_oxford_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_and_oxford_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_or_no_punct = list_with_or.rstrip(string.punctuation)
                            if list_with_or_no_punct != list_with_or and len(list_with_or_no_punct) > 2:
                                direct_value_patterns.append((list_with_or_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_or_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            list_with_or_oxford_no_punct = list_with_or_oxford.rstrip(string.punctuation)
                            if list_with_or_oxford_no_punct != list_with_or_oxford and len(list_with_or_oxford_no_punct) > 2:
                                direct_value_patterns.append((list_with_or_oxford_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(list_with_or_oxford_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                        value_with_comma_space = field_value_str.replace(',', ', ')
                        value_with_comma_space = re.sub(r'\s+', ' ', value_with_comma_space).strip()
                        if value_with_comma_space != field_value_str:
                            direct_value_patterns.append((value_with_comma_space, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_with_comma_space) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                    if '(' in field_value_str and ')' in field_value_str:
                        value_no_parens = re.sub(r'[()]', ' ', field_value_str)
                        value_no_parens = re.sub(r'\s+', ' ', value_no_parens).strip()
                        direct_value_patterns.append((value_no_parens, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_no_parens) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                        value_no_parens_no_punct = value_no_parens.rstrip(string.punctuation)
                        if value_no_parens_no_punct != value_no_parens and len(value_no_parens_no_punct) > 2:
                            direct_value_patterns.append((value_no_parens_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_no_parens_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                        value_without_paren_content = re.sub(r'\s*\([^)]*\)', '', field_value_str).strip()
                        if value_without_paren_content and len(value_without_paren_content) > 2:
                            direct_value_patterns.append((value_without_paren_content, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_without_paren_content) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                            value_without_paren_no_punct = value_without_paren_content.rstrip(string.punctuation)
                            if value_without_paren_no_punct != value_without_paren_content and len(value_without_paren_no_punct) > 2:
                                direct_value_patterns.append((value_without_paren_no_punct, r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(value_without_paren_no_punct) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'))

                for match_text, value_pattern in direct_value_patterns:
                    value_match = re.search(value_pattern, sentence, re.IGNORECASE)
                    if value_match:
                        matched_value = value_match.group()
                        result_match_type = match_type if (match_type and match_type.startswith('field_name')) else 'field_value'
                        print(f"      Found field VALUE in sentence: '{matched_value}' \u2192 will replace with placeholder")
                        return True, matched_value, field_name, result_match_type

                is_complex_value = (
                    '{' in field_value_str or '[' in field_value_str or
                    ("'" in field_value_str and ':' in field_value_str) or
                    'http' in field_value_str.lower() or
                    len(field_value_str) > 50 or
                    (field_value_str.count(',') >= 2 and ':' in field_value_str)
                )

                if is_complex_value and use_llm_fallback:
                    print(f"      Direct match failed for complex value - trying LLM extraction")
                    llm_extracted_text = self.query_local_llm_for_complex_list_extraction(sentence, field_name, field_value_str)
                    if llm_extracted_text:
                        print(f"      LLM extracted complex value in direct match: '{llm_extracted_text}'")
                        result_match_type = match_type if (match_type and match_type.startswith('field_name')) else 'field_name_llm'
                        return True, llm_extracted_text, field_name, result_match_type

                if is_date_value(field_value_str) and use_llm_fallback:
                    print(f"      Direct match failed for date value - trying LLM date extraction")
                    llm_extracted_date = self.query_local_llm_for_date_extraction(sentence, field_name, field_value_str)
                    if llm_extracted_date:
                        print(f"      LLM extracted date in direct match: '{llm_extracted_date}'")
                        result_match_type = match_type if (match_type and match_type.startswith('field_name')) else 'field_name_llm'
                        return True, llm_extracted_date, field_name, result_match_type

                if match_type and match_type.startswith('field_name'):
                    return True, field_value_str, field_name, match_type

                return False, field_value_str, field_name, ''
        else:
            return False, field_value_str, field_name, ''

    def attempt_replacement(self, sentence: str, field_name: str, matched_text: str, field_type: str) -> Tuple[bool, str]:
        """
        Attempt to replace the matched text in the sentence with a placeholder.
        Returns (success, replaced_sentence)
        """
        if not matched_text or matched_text in ["[ERROR]", "[FIELD_NAME_ONLY]"]:
            return False, sentence

        placeholder = f"[{field_name.upper().replace(' ', '_')}]"

        punctuation_no_dash = string.punctuation.replace("-", "")

        if field_type == "MISC":
            escaped_match = re.escape(matched_text)
            pattern = escaped_match
        elif field_type in ["BINARY", "NULL"]:
            pattern = r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(matched_text) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'
        else:
            pattern = r'(?:(?<=^)|(?<=[\s' + re.escape(punctuation_no_dash) + r']))' + re.escape(matched_text) + r'(?:(?=$)|(?=[\s' + re.escape(punctuation_no_dash) + r']))'

        try:
            replaced_sentence, num_replacements = re.subn(pattern, placeholder, sentence, count=1, flags=re.IGNORECASE)

            if num_replacements > 0:
                print(f"      Replacement successful: '{matched_text}' -> '{placeholder}'")
                return True, replaced_sentence
            else:
                simple_replaced = sentence.replace(matched_text, placeholder, 1)
                if simple_replaced != sentence:
                    print(f"      Replacement successful (simple): '{matched_text}' -> '{placeholder}'")
                    return True, simple_replaced

                lower_sentence = sentence.lower()
                lower_match = matched_text.lower()
                if lower_match in lower_sentence:
                    start_idx = lower_sentence.find(lower_match)
                    end_idx = start_idx + len(matched_text)
                    case_replaced = sentence[:start_idx] + placeholder + sentence[end_idx:]
                    print(f"      Replacement successful (case-insensitive): '{matched_text}' -> '{placeholder}'")
                    return True, case_replaced

                print(f"      Replacement failed: could not find '{matched_text}' in sentence")
                return False, sentence

        except re.error as e:
            print(f"      Regex error during replacement: {e}")
            simple_replaced = sentence.replace(matched_text, placeholder, 1)
            if simple_replaced != sentence:
                return True, simple_replaced
            return False, sentence
