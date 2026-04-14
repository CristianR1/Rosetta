"""DocumentTemplateSystem - main coordinator class assembled from mixins."""

import os
from typing import Dict, List, Any

from .config import create_local_llm_openai_client
from .variation_generator import VariationBankGenerator
from .llm_extraction import LLMExtractionMixin
from .field_detection import FieldDetectionMixin
from .template_creation import TemplateCreationMixin
from .document_assembler import DocumentAssemblyMixin
from .models import SentenceTemplate
from .text_utils import clean_field_string, clean_llm_response, is_misc_value, is_date_value, count_placeholders


class DocumentTemplateSystem(
    LLMExtractionMixin,
    FieldDetectionMixin,
    TemplateCreationMixin,
    DocumentAssemblyMixin,
):
    """Orchestrates template creation, field detection, and document variation generation."""

    def __init__(
        self,
        null_mode: str = "implicit",
        binary_mode: str = "implicit",
        dataset_folder_name: str = "MINIDEV",
        docgen_base_dir: str = None,
        num_variations: int = 15,
    ):
        self._null_mode: str = null_mode
        self._binary_mode: str = binary_mode
        if docgen_base_dir is None:
            docgen_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._docgen_base_dir: str = docgen_base_dir
        self.dataset_folder_name: str = dataset_folder_name
        self.variation_generator = VariationBankGenerator(
            num_variations=num_variations,
            null_mode=self._null_mode,
            binary_mode=self._binary_mode,
        )
        self.templates: List[SentenceTemplate] = []
        self.consistency_mapping: Dict[str, str] = {}
        self.document_context: str = ""
        self.local_llm_client = create_local_llm_openai_client()

    @property
    def num_variations(self) -> int:
        return self.variation_generator.num_variations

    @num_variations.setter
    def num_variations(self, value: int):
        self.variation_generator.num_variations = int(value)

    @property
    def null_mode(self) -> str:
        return self._null_mode

    @null_mode.setter
    def null_mode(self, value: str):
        self._null_mode = value
        self.variation_generator.null_mode = value

    @property
    def binary_mode(self) -> str:
        return self._binary_mode

    @binary_mode.setter
    def binary_mode(self, value: str):
        self._binary_mode = value
        self.variation_generator.binary_mode = value

    def clean_field_string(self, text):
        return clean_field_string(text)

    def clean_llm_response(self, response):
        return clean_llm_response(response)

    def is_misc_value(self, value):
        return is_misc_value(value)

    def count_placeholders(self, text):
        return count_placeholders(text)

    def is_date_value(self, value):
        return is_date_value(value)
