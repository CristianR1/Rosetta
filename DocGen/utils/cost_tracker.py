#!/usr/bin/env python3

import json
import os
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMCostConfig:
    # API Cost Configuration
    gpt4o_input_cost_per_1m: float = 5.00
    gpt4o_output_cost_per_1m: float = 15.00
    gpt4o_mini_input_cost_per_1m: float = 0.15
    gpt4o_mini_output_cost_per_1m: float = 0.60
    local_llm_input_cost_per_1m: float = 0.00  # Local LLM is free
    local_llm_output_cost_per_1m: float = 0.00
    
    # Average token estimates for different operations
    avg_descriptor_input_tokens: int = 800
    avg_descriptor_output_tokens: int = 150
    avg_db_context_input_tokens: int = 1500
    avg_db_context_output_tokens: int = 200
    avg_table_context_input_tokens: int = 1200
    avg_table_context_output_tokens: int = 150
    avg_template_input_tokens: int = 2000
    avg_template_output_tokens: int = 1000
    avg_enhancement_input_tokens: int = 1500
    avg_enhancement_output_tokens: int = 500
    
    # Sentence variation tokens (now using local LLM)
    avg_sentence_variation_input_tokens: int = 1500
    avg_sentence_variation_output_tokens: int = 800
    
    # Local LLM operation token estimates
    avg_field_detection_input_tokens: int = 800
    avg_field_detection_output_tokens: int = 50
    avg_field_verification_input_tokens: int = 1200
    avg_field_verification_output_tokens: int = 50
    avg_value_replacement_input_tokens: int = 1000
    avg_value_replacement_output_tokens: int = 300
    avg_counter_refinement_input_tokens: int = 500
    avg_counter_refinement_output_tokens: int = 300
    avg_lexical_filter_input_tokens: int = 600
    avg_lexical_filter_output_tokens: int = 100
    
    # Structural variation generation tokens (GPT-4o for quality variations)
    avg_structural_variation_input_tokens: int = 1500
    avg_structural_variation_output_tokens: int = 1200
    
    # Estimation constants for full runs
    avg_sentences_per_table: int = 12
    avg_fields_per_sentence: int = 2
    avg_binary_null_fields_per_table: int = 3
    avg_lexical_words_per_sentence: int = 6


@dataclass
class LLMUsageStats:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0


@dataclass
class SkippedItemStats:
    skipped_descriptors: int = 0
    skipped_db_contexts: int = 0
    skipped_table_contexts: int = 0
    skipped_templates: int = 0
    skipped_sentence_variations: int = 0
    skipped_enhancements: int = 0
    reused_descriptors: int = 0


@dataclass
class LocalLLMOperationStats:
    field_detection_calls: int = 0
    field_verification_calls: int = 0
    value_replacement_calls: int = 0
    counter_refinement_calls: int = 0
    date_extraction_calls: int = 0
    null_extraction_calls: int = 0
    binary_extraction_calls: int = 0
    misc_extraction_calls: int = 0
    complex_list_extraction_calls: int = 0
    structural_variation_calls: int = 0
    lexical_filter_calls: int = 0
    sentence_generation_calls: int = 0
    common_language_field_calls: int = 0


class CostTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.config = LLMCostConfig()
        self.usage_stats: Dict[str, LLMUsageStats] = {}
        self.skipped_stats = SkippedItemStats()
        self.local_llm_ops = LocalLLMOperationStats()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        self._init_models()
    
    def _init_models(self):
        self.usage_stats = {
            'gpt-4o': LLMUsageStats(model='gpt-4o'),
            'gpt-4o-mini': LLMUsageStats(model='gpt-4o-mini'),
            'local-llm': LLMUsageStats(model='local-llm')
        }
    
    def reset(self):
        self._init_models()
        self.skipped_stats = SkippedItemStats()
        self.local_llm_ops = LocalLLMOperationStats()
        self.start_time = None
        self.end_time = None
    
    def start_tracking(self):
        self.reset()
        self.start_time = datetime.now()
    
    def stop_tracking(self):
        self.end_time = datetime.now()
    
    def track_call(self, model: str, input_tokens: int, output_tokens: int):
        model_key = self._normalize_model_name(model)
        
        if model_key not in self.usage_stats:
            self.usage_stats[model_key] = LLMUsageStats(model=model_key)
        
        self.usage_stats[model_key].input_tokens += input_tokens
        self.usage_stats[model_key].output_tokens += output_tokens
        self.usage_stats[model_key].call_count += 1
    
    def track_skipped_descriptors(self, count: int):
        self.skipped_stats.skipped_descriptors += count
    
    def track_reused_descriptors(self, count: int):
        self.skipped_stats.reused_descriptors += count
    
    def track_skipped_db_contexts(self, count: int):
        self.skipped_stats.skipped_db_contexts += count
    
    def track_skipped_table_contexts(self, count: int):
        self.skipped_stats.skipped_table_contexts += count
    
    def track_skipped_templates(self, count: int):
        self.skipped_stats.skipped_templates += count
    
    def track_skipped_sentence_variations(self, count: int):
        self.skipped_stats.skipped_sentence_variations += count
    
    def track_skipped_enhancements(self, count: int):
        self.skipped_stats.skipped_enhancements += count
    
    def track_local_llm_operation(self, operation_type: str, count: int = 1):
        op_mapping = {
            'field_detection': 'field_detection_calls',
            'field_verification': 'field_verification_calls',
            'value_replacement': 'value_replacement_calls',
            'counter_refinement': 'counter_refinement_calls',
            'date_extraction': 'date_extraction_calls',
            'null_extraction': 'null_extraction_calls',
            'binary_extraction': 'binary_extraction_calls',
            'misc_extraction': 'misc_extraction_calls',
            'complex_list_extraction': 'complex_list_extraction_calls',
            'structural_variation': 'structural_variation_calls',
            'lexical_filter': 'lexical_filter_calls',
            'sentence_generation': 'sentence_generation_calls',
            'common_language_field': 'common_language_field_calls',
        }
        
        if operation_type in op_mapping:
            attr = op_mapping[operation_type]
            setattr(self.local_llm_ops, attr, getattr(self.local_llm_ops, attr) + count)
    
    def track_gpt4o_operation(self, operation_type: str, count: int = 1):
        pass
    
    def _normalize_model_name(self, model: str) -> str:
        model_lower = model.lower()
        
        if 'gpt-4o-mini' in model_lower or 'gpt4o-mini' in model_lower:
            return 'gpt-4o-mini'
        elif 'gpt-4o' in model_lower or 'gpt4o' in model_lower:
            return 'gpt-4o'
        elif 'local' in model_lower or 'oss' in model_lower or 'bellatrix' in model_lower:
            return 'local-llm'
        else:
            return 'local-llm'
    
    def calculate_cost(self, model_key: str) -> float:
        if model_key not in self.usage_stats:
            return 0.0
        
        stats = self.usage_stats[model_key]
        
        if model_key == 'gpt-4o':
            input_cost = (stats.input_tokens / 1_000_000) * self.config.gpt4o_input_cost_per_1m
            output_cost = (stats.output_tokens / 1_000_000) * self.config.gpt4o_output_cost_per_1m
        elif model_key == 'gpt-4o-mini':
            input_cost = (stats.input_tokens / 1_000_000) * self.config.gpt4o_mini_input_cost_per_1m
            output_cost = (stats.output_tokens / 1_000_000) * self.config.gpt4o_mini_output_cost_per_1m
        else:
            input_cost = (stats.input_tokens / 1_000_000) * self.config.local_llm_input_cost_per_1m
            output_cost = (stats.output_tokens / 1_000_000) * self.config.local_llm_output_cost_per_1m
        
        return input_cost + output_cost
    
    def get_total_cost(self) -> float:
        total = 0.0
        for model_key in self.usage_stats:
            total += self.calculate_cost(model_key)
        return total
    
    def calculate_theoretical_cost(self) -> Dict[str, float]:
        theoretical = {
            'gpt-4o': 0.0,
            'gpt-4o-mini': 0.0,
            'local-llm': 0.0,
            'total': 0.0,
            'breakdown': {}
        }
        
        skipped = self.skipped_stats
        cfg = self.config
        
        descriptor_input = (skipped.skipped_descriptors + skipped.reused_descriptors) * cfg.avg_descriptor_input_tokens
        descriptor_output = (skipped.skipped_descriptors + skipped.reused_descriptors) * cfg.avg_descriptor_output_tokens
        descriptor_cost = (descriptor_input / 1_000_000) * cfg.gpt4o_mini_input_cost_per_1m + \
                         (descriptor_output / 1_000_000) * cfg.gpt4o_mini_output_cost_per_1m
        theoretical['gpt-4o-mini'] += descriptor_cost
        theoretical['breakdown']['skipped_descriptors'] = {
            'count': skipped.skipped_descriptors + skipped.reused_descriptors,
            'cost': descriptor_cost
        }
        
        db_context_input = skipped.skipped_db_contexts * cfg.avg_db_context_input_tokens
        db_context_output = skipped.skipped_db_contexts * cfg.avg_db_context_output_tokens
        db_context_cost = (db_context_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                         (db_context_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        theoretical['gpt-4o'] += db_context_cost
        theoretical['breakdown']['skipped_db_contexts'] = {
            'count': skipped.skipped_db_contexts,
            'cost': db_context_cost
        }
        
        table_context_input = skipped.skipped_table_contexts * cfg.avg_table_context_input_tokens
        table_context_output = skipped.skipped_table_contexts * cfg.avg_table_context_output_tokens
        table_context_cost = (table_context_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                            (table_context_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        theoretical['gpt-4o'] += table_context_cost
        theoretical['breakdown']['skipped_table_contexts'] = {
            'count': skipped.skipped_table_contexts,
            'cost': table_context_cost
        }
        
        template_input = skipped.skipped_templates * cfg.avg_template_input_tokens
        template_output = skipped.skipped_templates * cfg.avg_template_output_tokens
        template_cost = (template_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                       (template_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        theoretical['gpt-4o'] += template_cost
        theoretical['breakdown']['skipped_templates'] = {
            'count': skipped.skipped_templates,
            'cost': template_cost
        }
        
        # Sentence variations now use LOCAL LLM (free) for field operations
        # but still use GPT-4o for structural variation generation and lexical filtering
        # Estimate: ~12 sentences per table, each with structural variations (GPT-4o)
        # and field detection/verification (local LLM - free)
        variation_input = skipped.skipped_sentence_variations * cfg.avg_sentence_variation_input_tokens
        variation_output = skipped.skipped_sentence_variations * cfg.avg_sentence_variation_output_tokens
        # Local LLM operations are free
        variation_cost = (variation_input / 1_000_000) * cfg.local_llm_input_cost_per_1m + \
                        (variation_output / 1_000_000) * cfg.local_llm_output_cost_per_1m
        theoretical['local-llm'] += variation_cost  # This will be 0.0
        theoretical['breakdown']['skipped_sentence_variations'] = {
            'count': skipped.skipped_sentence_variations,
            'cost': variation_cost,
            'note': 'Local LLM (free)'
        }
        
        enhancement_input = skipped.skipped_enhancements * cfg.avg_enhancement_input_tokens
        enhancement_output = skipped.skipped_enhancements * cfg.avg_enhancement_output_tokens
        enhancement_cost = (enhancement_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                          (enhancement_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        theoretical['gpt-4o'] += enhancement_cost
        theoretical['breakdown']['skipped_enhancements'] = {
            'count': skipped.skipped_enhancements,
            'cost': enhancement_cost
        }
        
        theoretical['total'] = theoretical['gpt-4o'] + theoretical['gpt-4o-mini'] + theoretical['local-llm']
        
        return theoretical
    
    def estimate_full_run_cost(self, num_tables: int = 798, 
                                avg_columns_per_table: int = 15,
                                avg_sentences_per_table: int = None,
                                include_breakdown: bool = True) -> Dict:
        """
        Estimate the cost for a full run with all tables.
        
        Args:
            num_tables: Number of tables to process (default 798)
            avg_columns_per_table: Average columns per table
            avg_sentences_per_table: Average sentences per generated narrative
            include_breakdown: Include detailed breakdown by operation type
            
        Returns:
            Dict with cost estimates and breakdowns
        """
        cfg = self.config
        
        if avg_sentences_per_table is None:
            avg_sentences_per_table = cfg.avg_sentences_per_table
        
        total_columns = num_tables * avg_columns_per_table
        total_sentences = num_tables * avg_sentences_per_table
        
        # Count different field types per table (binary/null fields get counter variations)
        binary_null_sentences = num_tables * cfg.avg_binary_null_fields_per_table
        standard_sentences = total_sentences - binary_null_sentences
        
        estimate = {
            'parameters': {
                'num_tables': num_tables,
                'avg_columns_per_table': avg_columns_per_table,
                'total_columns': total_columns,
                'avg_sentences_per_table': avg_sentences_per_table,
                'total_sentences': total_sentences,
            },
            'gpt_4o_cost': 0.0,
            'gpt_4o_mini_cost': 0.0,
            'local_llm_cost': 0.0,
            'total_cost': 0.0,
            'breakdown': {}
        }
        
        # 1. Column Descriptors (GPT-4o-mini) - one per column
        descriptor_input = total_columns * cfg.avg_descriptor_input_tokens
        descriptor_output = total_columns * cfg.avg_descriptor_output_tokens
        descriptor_cost = (descriptor_input / 1_000_000) * cfg.gpt4o_mini_input_cost_per_1m + \
                         (descriptor_output / 1_000_000) * cfg.gpt4o_mini_output_cost_per_1m
        estimate['gpt_4o_mini_cost'] += descriptor_cost
        estimate['breakdown']['column_descriptors'] = {
            'count': total_columns,
            'model': 'gpt-4o-mini',
            'input_tokens': descriptor_input,
            'output_tokens': descriptor_output,
            'cost': descriptor_cost
        }
        
        # 2. Database Contexts (GPT-4o) - one per database (assume ~50 unique databases)
        num_databases = min(num_tables // 10, 50)  # Rough estimate
        db_context_input = num_databases * cfg.avg_db_context_input_tokens
        db_context_output = num_databases * cfg.avg_db_context_output_tokens
        db_context_cost = (db_context_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                         (db_context_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        estimate['gpt_4o_cost'] += db_context_cost
        estimate['breakdown']['database_contexts'] = {
            'count': num_databases,
            'model': 'gpt-4o',
            'input_tokens': db_context_input,
            'output_tokens': db_context_output,
            'cost': db_context_cost
        }
        
        # 3. Table Contexts (GPT-4o) - one per table
        table_context_input = num_tables * cfg.avg_table_context_input_tokens
        table_context_output = num_tables * cfg.avg_table_context_output_tokens
        table_context_cost = (table_context_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                            (table_context_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        estimate['gpt_4o_cost'] += table_context_cost
        estimate['breakdown']['table_contexts'] = {
            'count': num_tables,
            'model': 'gpt-4o',
            'input_tokens': table_context_input,
            'output_tokens': table_context_output,
            'cost': table_context_cost
        }
        
        # 4. Initial Template/Narrative Generation (GPT-4o) - one per table
        template_input = num_tables * cfg.avg_template_input_tokens
        template_output = num_tables * cfg.avg_template_output_tokens
        template_cost = (template_input / 1_000_000) * cfg.gpt4o_input_cost_per_1m + \
                       (template_output / 1_000_000) * cfg.gpt4o_output_cost_per_1m
        estimate['gpt_4o_cost'] += template_cost
        estimate['breakdown']['narrative_templates'] = {
            'count': num_tables,
            'model': 'gpt-4o',
            'input_tokens': template_input,
            'output_tokens': template_output,
            'cost': template_cost
        }
        
        # 5. Structural Variations (Local LLM - FREE) - 15 variations per sentence
        # Standard sentences: 1 call for 15 variations
        # Binary/Null sentences: 2 calls (variations + counter_variations)
        structural_calls_standard = standard_sentences
        structural_calls_binary_null = binary_null_sentences * 2
        total_structural_calls = structural_calls_standard + structural_calls_binary_null
        
        structural_input = total_structural_calls * cfg.avg_structural_variation_input_tokens
        structural_output = total_structural_calls * cfg.avg_structural_variation_output_tokens
        structural_cost = 0.0
        estimate['local_llm_cost'] += structural_cost
        estimate['breakdown']['structural_variations'] = {
            'count': total_structural_calls,
            'model': 'local-llm',
            'input_tokens': structural_input,
            'output_tokens': structural_output,
            'cost': structural_cost,
            'note': f'Local LLM (FREE) - Standard: {structural_calls_standard}, Binary/Null: {structural_calls_binary_null}'
        }
        
        # 6. Lexical Filtering (Local LLM - FREE) - ~6 words per sentence
        lexical_calls = total_sentences * cfg.avg_lexical_words_per_sentence
        lexical_input = lexical_calls * cfg.avg_lexical_filter_input_tokens
        lexical_output = lexical_calls * cfg.avg_lexical_filter_output_tokens
        lexical_cost = 0.0
        estimate['local_llm_cost'] += lexical_cost
        estimate['breakdown']['lexical_filtering'] = {
            'count': lexical_calls,
            'model': 'local-llm',
            'input_tokens': lexical_input,
            'output_tokens': lexical_output,
            'cost': lexical_cost,
            'note': 'Local LLM (FREE)'
        }
        
        # 7. Local LLM Operations (FREE)
        # Field detection/verification: ~2 fields per sentence * 15 variations
        field_ops_per_sentence = cfg.avg_fields_per_sentence * 15  # Per variation
        total_field_detection = total_sentences * field_ops_per_sentence
        
        # Value replacement fallback: ~20% of variations need LLM fallback
        value_replacement_calls = int(total_sentences * 15 * 0.2)
        
        # Counter refinement: 15 per binary/null sentence
        counter_refinement_calls = binary_null_sentences * 15
        
        local_llm_input = (
            total_field_detection * cfg.avg_field_detection_input_tokens +
            value_replacement_calls * cfg.avg_value_replacement_input_tokens +
            counter_refinement_calls * cfg.avg_counter_refinement_input_tokens
        )
        local_llm_output = (
            total_field_detection * cfg.avg_field_detection_output_tokens +
            value_replacement_calls * cfg.avg_value_replacement_output_tokens +
            counter_refinement_calls * cfg.avg_counter_refinement_output_tokens
        )
        # Local LLM is free
        local_llm_cost = 0.0
        estimate['local_llm_cost'] = local_llm_cost
        estimate['breakdown']['local_llm_operations'] = {
            'field_detection_calls': total_field_detection,
            'value_replacement_calls': value_replacement_calls,
            'counter_refinement_calls': counter_refinement_calls,
            'total_calls': total_field_detection + value_replacement_calls + counter_refinement_calls,
            'model': 'local-llm',
            'input_tokens': local_llm_input,
            'output_tokens': local_llm_output,
            'cost': local_llm_cost,
            'note': 'Local LLM is free'
        }
        
        # Calculate totals
        estimate['total_cost'] = estimate['gpt_4o_cost'] + estimate['gpt_4o_mini_cost'] + estimate['local_llm_cost']
        
        estimate['summary'] = {
            'total_gpt4o_tokens': (
                estimate['breakdown']['database_contexts']['input_tokens'] +
                estimate['breakdown']['database_contexts']['output_tokens'] +
                estimate['breakdown']['table_contexts']['input_tokens'] +
                estimate['breakdown']['table_contexts']['output_tokens'] +
                estimate['breakdown']['narrative_templates']['input_tokens'] +
                estimate['breakdown']['narrative_templates']['output_tokens']
            ),
            'total_gpt4o_mini_tokens': (
                estimate['breakdown']['column_descriptors']['input_tokens'] +
                estimate['breakdown']['column_descriptors']['output_tokens']
            ),
            'total_local_llm_tokens': (
                local_llm_input + local_llm_output +
                estimate['breakdown']['structural_variations']['input_tokens'] +
                estimate['breakdown']['structural_variations']['output_tokens'] +
                estimate['breakdown']['lexical_filtering']['input_tokens'] +
                estimate['breakdown']['lexical_filtering']['output_tokens']
            ),
            'estimated_api_calls': {
                'gpt-4o': (num_databases + num_tables + num_tables),
                'gpt-4o-mini': total_columns,
                'local-llm': (total_field_detection + value_replacement_calls + counter_refinement_calls + 
                             total_structural_calls + lexical_calls)
            }
        }
        
        return estimate
    
    def print_full_run_estimate(self, num_tables: int = 798, avg_columns_per_table: int = 15):
        """Print a formatted estimate for a full run"""
        estimate = self.estimate_full_run_cost(num_tables, avg_columns_per_table)
        
        print()
        print("=" * 80)
        print("  FULL RUN COST ESTIMATION")
        print("=" * 80)
        print()
        print(f"  Parameters:")
        print(f"    Tables to process:        {estimate['parameters']['num_tables']:>10,}")
        print(f"    Avg columns per table:    {estimate['parameters']['avg_columns_per_table']:>10}")
        print(f"    Total columns:            {estimate['parameters']['total_columns']:>10,}")
        print(f"    Avg sentences per table:  {estimate['parameters']['avg_sentences_per_table']:>10}")
        print(f"    Total sentences:          {estimate['parameters']['total_sentences']:>10,}")
        print()
        print("-" * 80)
        print("  COST BREAKDOWN BY OPERATION")
        print("-" * 80)
        print()
        
        breakdown = estimate['breakdown']
        
        print(f"  {'Operation':<35} {'Model':<15} {'Calls':>10} {'Cost':>12}")
        print(f"  {'-'*35} {'-'*15} {'-'*10} {'-'*12}")
        
        for op_name, op_data in breakdown.items():
            model = op_data.get('model', 'unknown')
            count = op_data.get('count', op_data.get('total_calls', 0))
            cost = op_data.get('cost', 0)
            print(f"  {op_name.replace('_', ' ').title():<35} {model:<15} {count:>10,} ${cost:>10.4f}")
        
        print()
        print("-" * 80)
        print("  COST SUMMARY BY MODEL")
        print("-" * 80)
        print(f"    GPT-4o:                   ${estimate['gpt_4o_cost']:>12.4f}")
        print(f"    GPT-4o-mini:              ${estimate['gpt_4o_mini_cost']:>12.4f}")
        print(f"    Local LLM:                ${estimate['local_llm_cost']:>12.4f}  (FREE)")
        print("-" * 80)
        print(f"    TOTAL ESTIMATED COST:     ${estimate['total_cost']:>12.4f}")
        print("=" * 80)
        print()
        
        # Print API call summary
        api_calls = estimate['summary']['estimated_api_calls']
        print("  ESTIMATED API CALLS:")
        print(f"    GPT-4o calls:             {api_calls['gpt-4o']:>12,}")
        print(f"    GPT-4o-mini calls:        {api_calls['gpt-4o-mini']:>12,}")
        print(f"    Local LLM calls:          {api_calls['local-llm']:>12,}  (FREE)")
        print()
        
        return estimate
    
    def get_summary(self) -> Dict:
        summary = {
            'models': {},
            'actual_cost': 0.0,
            'total_calls': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'duration_seconds': None,
            'skipped_items': {},
            'local_llm_operations': {},
            'theoretical_savings': {},
            'total_theoretical_cost': 0.0
        }
        
        for model_key, stats in self.usage_stats.items():
            if stats.call_count > 0:
                model_cost = self.calculate_cost(model_key)
                summary['models'][model_key] = {
                    'call_count': stats.call_count,
                    'input_tokens': stats.input_tokens,
                    'output_tokens': stats.output_tokens,
                    'cost': model_cost
                }
                summary['actual_cost'] += model_cost
                summary['total_calls'] += stats.call_count
                summary['total_input_tokens'] += stats.input_tokens
                summary['total_output_tokens'] += stats.output_tokens
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            summary['duration_seconds'] = duration
        
        summary['skipped_items'] = {
            'descriptors': self.skipped_stats.skipped_descriptors,
            'reused_descriptors': self.skipped_stats.reused_descriptors,
            'db_contexts': self.skipped_stats.skipped_db_contexts,
            'table_contexts': self.skipped_stats.skipped_table_contexts,
            'templates': self.skipped_stats.skipped_templates,
            'sentence_variations': self.skipped_stats.skipped_sentence_variations,
            'enhancements': self.skipped_stats.skipped_enhancements
        }
        
        summary['local_llm_operations'] = {
            'field_detection': self.local_llm_ops.field_detection_calls,
            'field_verification': self.local_llm_ops.field_verification_calls,
            'value_replacement': self.local_llm_ops.value_replacement_calls,
            'counter_refinement': self.local_llm_ops.counter_refinement_calls,
            'date_extraction': self.local_llm_ops.date_extraction_calls,
            'null_extraction': self.local_llm_ops.null_extraction_calls,
            'binary_extraction': self.local_llm_ops.binary_extraction_calls,
            'misc_extraction': self.local_llm_ops.misc_extraction_calls,
            'complex_list_extraction': self.local_llm_ops.complex_list_extraction_calls,
            'structural_variations': self.local_llm_ops.structural_variation_calls,
            'lexical_filter': self.local_llm_ops.lexical_filter_calls,
            'sentence_generation': self.local_llm_ops.sentence_generation_calls,
            'common_language_field': self.local_llm_ops.common_language_field_calls,
        }
        
        theoretical = self.calculate_theoretical_cost()
        summary['theoretical_savings'] = theoretical
        summary['total_theoretical_cost'] = summary['actual_cost'] + theoretical['total']
        
        return summary
    
    def print_summary(self):
        summary = self.get_summary()
        
        print()
        print("=" * 70)
        print("  LLM USAGE AND COST SUMMARY")
        print("=" * 70)
        print()
        
        if summary['duration_seconds']:
            minutes = int(summary['duration_seconds'] // 60)
            seconds = int(summary['duration_seconds'] % 60)
            print(f"  Total Duration: {minutes}m {seconds}s")
            print()
        
        print("  ACTUAL USAGE (This Run):")
        print("-" * 70)
        print(f"  {'Model':<20} {'Calls':>10} {'Input Tokens':>15} {'Output Tokens':>15} {'Cost':>10}")
        print("-" * 70)
        
        for model_key, model_stats in summary['models'].items():
            cost_display = f"${model_stats['cost']:>8.4f}" if model_key != 'local-llm' else f"${model_stats['cost']:>8.4f} (FREE)"
            print(f"  {model_key:<20} {model_stats['call_count']:>10,} {model_stats['input_tokens']:>15,} {model_stats['output_tokens']:>15,} {cost_display}")
        
        print("-" * 70)
        print(f"  {'ACTUAL TOTAL':<20} {summary['total_calls']:>10,} {summary['total_input_tokens']:>15,} {summary['total_output_tokens']:>15,} ${summary['actual_cost']:>8.4f}")
        print("=" * 70)
        print()
        
        # Local LLM operation breakdown
        local_ops = summary.get('local_llm_operations', {})
        total_local_ops = sum(v for k, v in local_ops.items() if not k.endswith('_gpt4o'))
        total_gpt4o_ops = sum(v for k, v in local_ops.items() if k.endswith('_gpt4o'))
        
        if total_local_ops > 0 or total_gpt4o_ops > 0:
            print("  LOCAL LLM OPERATION BREAKDOWN (Sentence Variation Pipeline):")
            print("-" * 70)
            
            # Local LLM operations (free)
            if total_local_ops > 0:
                print("  Free Operations (Local LLM):")
                for op_name, count in local_ops.items():
                    if count > 0 and not op_name.endswith('_gpt4o'):
                        display_name = op_name.replace('_', ' ').title()
                        print(f"    {display_name:<35} {count:>10,}")
                print(f"    {'Total Local LLM Calls':<35} {total_local_ops:>10,} (FREE)")
            
            # GPT-4o operations within pipeline (paid)
            if total_gpt4o_ops > 0:
                print()
                print("  Paid Operations (GPT-4o within pipeline):")
                for op_name, count in local_ops.items():
                    if count > 0 and op_name.endswith('_gpt4o'):
                        display_name = op_name.replace('_gpt4o', '').replace('_', ' ').title()
                        print(f"    {display_name:<35} {count:>10,}")
            
            print("-" * 70)
            print()
        
        skipped = summary['skipped_items']
        total_skipped = sum(skipped.values())
        
        if total_skipped > 0:
            print("  SKIPPED/CACHED ITEMS (Pre-existing Data):")
            print("-" * 70)
            
            if skipped['descriptors'] > 0 or skipped['reused_descriptors'] > 0:
                print(f"    Column descriptors skipped:     {skipped['descriptors']:>6}")
                print(f"    Column descriptors reused:      {skipped['reused_descriptors']:>6}")
            if skipped['db_contexts'] > 0:
                print(f"    Database contexts cached:       {skipped['db_contexts']:>6}")
            if skipped['table_contexts'] > 0:
                print(f"    Table contexts cached:          {skipped['table_contexts']:>6}")
            if skipped['templates'] > 0:
                print(f"    Templates skipped:              {skipped['templates']:>6}")
            if skipped['sentence_variations'] > 0:
                print(f"    Sentence variations skipped:    {skipped['sentence_variations']:>6} (Local LLM - FREE)")
            if skipped['enhancements'] > 0:
                print(f"    Enhancements skipped:           {skipped['enhancements']:>6}")
            
            print("-" * 70)
            print()
            
            theoretical = summary['theoretical_savings']
            print("  THEORETICAL COST (If Generated From Scratch):")
            print("-" * 70)
            
            breakdown = theoretical.get('breakdown', {})
            for item_type, item_data in breakdown.items():
                if item_data['count'] > 0:
                    display_name = item_type.replace('skipped_', '').replace('_', ' ').title()
                    note = item_data.get('note', '')
                    note_str = f" ({note})" if note else ""
                    print(f"    {display_name:<30} {item_data['count']:>6} items  ${item_data['cost']:>8.4f}{note_str}")
            
            print("-" * 70)
            print(f"    {'Estimated savings from cache:':<38}     ${theoretical['total']:>8.4f}")
            print("=" * 70)
            print()
        
        print("  COST SUMMARY:")
        print("-" * 70)
        print(f"    Actual cost (this run):       ${summary['actual_cost']:>10.4f}")
        
        if total_skipped > 0:
            print(f"    Estimated savings (cached):   ${summary['theoretical_savings']['total']:>10.4f}")
            print(f"    Theoretical full cost:        ${summary['total_theoretical_cost']:>10.4f}")
            
            if summary['total_theoretical_cost'] > 0:
                savings_pct = (summary['theoretical_savings']['total'] / summary['total_theoretical_cost']) * 100
                print(f"    Savings percentage:           {savings_pct:>10.1f}%")
        
        print("-" * 70)
        print(f"    TOTAL PAID THIS RUN:          ${summary['actual_cost']:>10.4f}")
        print("=" * 70)
        print()
    
    def save_report(self, output_path: str):
        summary = self.get_summary()
        summary['timestamp'] = datetime.now().isoformat()
        summary['config'] = {
            'gpt4o_input_per_1m': self.config.gpt4o_input_cost_per_1m,
            'gpt4o_output_per_1m': self.config.gpt4o_output_cost_per_1m,
            'gpt4o_mini_input_per_1m': self.config.gpt4o_mini_input_cost_per_1m,
            'gpt4o_mini_output_per_1m': self.config.gpt4o_mini_output_cost_per_1m,
            'local_llm_input_per_1m': self.config.local_llm_input_cost_per_1m,
            'local_llm_output_per_1m': self.config.local_llm_output_cost_per_1m,
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)


def get_cost_tracker() -> CostTracker:
    return CostTracker()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // 4


def track_openai_response(response, model: str = None):
    """Track OpenAI API response tokens and costs"""
    tracker = get_cost_tracker()
    
    if hasattr(response, 'usage') and response.usage:
        input_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
        
        if model is None and hasattr(response, 'model'):
            model = response.model
        
        if model:
            tracker.track_call(model, input_tokens, output_tokens)
    elif model:
        tracker.track_call(model, 0, 0)


def track_local_llm_operation(operation_type: str, count: int = 1):
    """Convenience function to track local LLM operations"""
    tracker = get_cost_tracker()
    tracker.track_local_llm_operation(operation_type, count)


def track_gpt4o_operation(operation_type: str, count: int = 1):
    """Convenience function to track GPT-4o operations in the pipeline"""
    tracker = get_cost_tracker()
    tracker.track_gpt4o_operation(operation_type, count)


def print_full_run_estimate(num_tables: int = 798, avg_columns_per_table: int = 15):
    """
    Print a cost estimate for a full run with all tables.
    
    This function estimates the cost of processing all 798 tables with nothing
    pre-generated, accounting for:
    - Column descriptors (GPT-4o-mini)
    - Database and table contexts (GPT-4o)
    - Narrative templates (GPT-4o)
    - Structural variations (GPT-4o)
    - Lexical filtering (GPT-4o)
    - Field detection/verification (Local LLM - FREE)
    - Value replacement (Local LLM - FREE)
    - Counter refinement (Local LLM - FREE)
    """
    tracker = get_cost_tracker()
    return tracker.print_full_run_estimate(num_tables, avg_columns_per_table)


def get_full_run_estimate(num_tables: int = 798, avg_columns_per_table: int = 15) -> Dict:
    """Get cost estimate as a dictionary for programmatic use"""
    tracker = get_cost_tracker()
    return tracker.estimate_full_run_cost(num_tables, avg_columns_per_table)


# Run estimation when module is executed directly
if __name__ == "__main__":
    print("\n" + "="*80)
    print("  DGS COST ESTIMATION TOOL")
    print("="*80)
    print()
    print("  This tool estimates the cost of running the Document Generation System")
    print("  for all tables in the MINIDEV database.")
    print()
    print("  Current configuration:")
    print("    - Sentence variations: Local LLM (FREE)")
    print("    - Field detection/verification: Local LLM (FREE)")
    print("    - Structural variations: GPT-4o (paid)")
    print("    - Lexical filtering: GPT-4o (paid)")
    print("    - Column descriptors: GPT-4o-mini (paid)")
    print()
    
    # Run estimation for 798 tables
    print_full_run_estimate(num_tables=798, avg_columns_per_table=15)
