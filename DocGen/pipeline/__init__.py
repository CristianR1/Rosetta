"""
DocGen pipeline package.

Provides backward-compatible re-exports so callers using old module names
continue to work. Import from here or from the specific sub-modules directly.
"""

from .config import (
    MISC_CHARACTERS,
    get_output_root,
    get_repo_root,
    load_repo_dotenv,
    normalize_openai_compat_base_url,
    is_local_llm_configured,
    get_local_llm_base_url,
    get_local_llm_api_key,
    get_configured_local_llm_model,
    get_cloud_parsing_fallback_model,
    get_parsing_llm_model,
    create_parsing_llm_openai_client,
    get_local_llm_model,
    create_local_llm_openai_client,
    get_ground_truth_data_path,
    get_ground_truth_table_sample_description_dir,
    get_ground_truth_column_descriptors_enhanced_path,
    get_ground_truth_column_descriptors_path,
    get_ground_truth_context_cache_path,
    table_sample_entry_path,
    get_mode_folder_name,
    get_noise_ratio_folder_name,
    get_sentence_template_path,
    get_narrative_template_path,
    get_sentence_templates_dir,
    get_narrative_templates_dir,
    variation_templates_exist,
    parse_data_noise_ratio_str,
    parse_ratio_folder_name,
    find_narrative_template_path_scan,
    resolve_backfill_noise_xy_and_narrative_path,
    compute_expected_transition_count,
)

from .models import SentenceTemplate, ColumnAnalysis, NarrativeAnalysis

from .data_loader import (
    load_column_descriptors,
    get_sample_entries,
    get_table_row_count,
    get_sample_data_from_database,
    is_complex_embedded_value,
    detect_complex_columns,
    detect_legacy_template_layout,
    migrate_legacy_templates,
)

from .text_utils import (
    clean_field_string,
    clean_llm_response,
    is_misc_value,
    is_date_value,
    count_placeholders,
    strip_hashes_from_text,
    identify_field_metadata,
    get_detection_patterns,
)

from .variation_generator import VariationBankGenerator, _ensure_nltk_wordnet
from .template_system import DocumentTemplateSystem
from .template_generator import TemplateGenerator
from .narrative_analyzer import NarrativeParsingAnalyzer
from .data_to_text_converter import DataToTextConverter
from .document_assembler import process_templates_standalone
