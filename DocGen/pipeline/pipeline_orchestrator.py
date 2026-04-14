#!/usr/bin/env python3

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cost_tracker import get_cost_tracker, CostTracker
from pipeline.config import get_ground_truth_data_path


class PipelineConfig:
    def __init__(self):
        self.data_folder_name = "MINIDEV"
        self.sentence_variations = 15
        self.null_mode = "implicit"
        self.binary_mode = "implicit"
        self.include_validation = True
        self.skip_phase1_validation = False
        self.selected_tables = []
        self.documents_per_table = 10
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_noise_ratio_x = 1
        self.data_noise_ratio_y = 0

    def get_data_path(self) -> str:
        return get_ground_truth_data_path(self.base_dir, self.data_folder_name)

    def get_dev_databases_path(self) -> str:
        return os.path.join(self.get_data_path(), "dev_databases")


class DocumentGenerationPipeline:
    def __init__(self):
        self.config = PipelineConfig()
        self.available_tables = []
        self.cost_tracker = get_cost_tracker()

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        print()

    def prompt_data_folder(self):
        self.print_header("DATA FOLDER CONFIGURATION")
        print("This pipeline uses a dataset folder under the repository GroundTruth directory.")
        print(f"Expected path pattern: <repo>/GroundTruth/<folder_name>/ (default: MINIDEV)")
        print()
        print(f"Default folder: {self.config.data_folder_name}")
        print()

        user_input = input("Enter folder name (or press Enter for default): ").strip()

        if user_input:
            self.config.data_folder_name = user_input

        data_path = self.config.get_data_path()
        if not os.path.exists(data_path):
            print(f"\nError: Folder not found at {data_path}")
            print("Please ensure the folder exists and try again.")
            sys.exit(1)

        print(f"\nUsing data folder: {self.config.data_folder_name}")
        print()

    def prompt_sentence_variations(self):
        self.print_header("SENTENCE VARIATION CONFIGURATION")
        print("This is a template-based document generation system which generates")
        print("sentence variations for each sentence to increase diversity in the output.")
        print()
        print("More variations = higher diversity but longer generation time.")
        print("Recommended: 10-20 variations per sentence.")
        print()

        while True:
            user_input = input("Enter number of sentence variations per sentence (default: 15): ").strip()

            if not user_input:
                self.config.sentence_variations = 15
                break

            try:
                num = int(user_input)
                if 1 <= num <= 50:
                    self.config.sentence_variations = num
                    break
                else:
                    print("Please enter a number between 1 and 50.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        print(f"\nUsing {self.config.sentence_variations} sentence variations per sentence.")
        print()

    def prompt_null_mode(self):
        self.print_header("NULL VALUE GENERATION MODE")
        print("Choose how NULL values should appear in generated text:")
        print()
        print("1. EXPLICIT - Sentences will contain raw NULL values like 'None' or 'NULL'")
        print("   Example: 'The customer address is NULL.'")
        print()
        print("2. IMPLICIT - Natural language embeddings of null values")
        print("   Example: 'The customer address is not specified.'")
        print()

        while True:
            user_input = input("Enter choice (1 for explicit, 2 for implicit, default: 2): ").strip()

            if not user_input or user_input == "2":
                self.config.null_mode = "implicit"
                break
            elif user_input == "1":
                self.config.null_mode = "explicit"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

        print(f"\nUsing NULL {self.config.null_mode} mode.")
        print()

    def prompt_binary_mode(self):
        self.print_header("BINARY VALUE GENERATION MODE")
        print("Choose how binary values (0/1, true/false) should appear in generated text:")
        print()
        print("1. EXPLICIT - Sentences will contain raw binary values like 'true', 'false', '1', '0'")
        print("   Example: 'The active status is 1.'")
        print()
        print("2. IMPLICIT - Natural language that specifies these states")
        print("   Example: 'The account is currently active.'")
        print()

        while True:
            user_input = input("Enter choice (1 for explicit, 2 for implicit, default: 2): ").strip()

            if not user_input or user_input == "2":
                self.config.binary_mode = "implicit"
                break
            elif user_input == "1":
                self.config.binary_mode = "explicit"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

        print(f"\nUsing binary {self.config.binary_mode} mode.")
        print()

    def prompt_data_noise_ratio(self):
        self.print_header("DATA TO NOISE RATIO CONFIGURATION")
        print("Configure the ratio of data sentences to transition (noise) sentences.")
        print()
        print("The ratio X:Y means: for every X data sentences, include Y transition sentences.")
        print("Transition sentences add narrative flow without containing actual data.")
        print()
        print("Examples:")
        print("  5:1 - For every 5 data sentences, add 1 transition sentence")
        print("  3:2 - For every 3 data sentences, add 2 transition sentences")
        print("  1:0 - No transition sentences (data only)")
        print()
        print("Notes:")
        print("  - X must be >= 1 (data is required)")
        print("  - Y can be 0 (no transitions) or >= 1")
        print("  - Ratios like 0:Y are invalid (data is always required)")
        print("  - When Y=0, sentences are dumped to text without LLM narrative generation")
        print()

        while True:
            user_input = input("Enter ratio X:Y (default: 1:0 for no transitions): ").strip()

            if not user_input:
                self.config.data_noise_ratio_x = 1
                self.config.data_noise_ratio_y = 0
                break

            if ':' not in user_input:
                print("Invalid format. Please use X:Y format (e.g., 5:1)")
                continue

            parts = user_input.split(':')
            if len(parts) != 2:
                print("Invalid format. Please use X:Y format (e.g., 5:1)")
                continue

            try:
                x = int(parts[0].strip())
                y = int(parts[1].strip())

                if x < 1:
                    print("X must be >= 1 (data sentences are required)")
                    continue
                if y < 0:
                    print("Y must be >= 0")
                    continue

                self.config.data_noise_ratio_x = x
                self.config.data_noise_ratio_y = y
                break

            except ValueError:
                print("Invalid numbers. Please enter integers for X and Y.")

        ratio_str = f"{self.config.data_noise_ratio_x}:{self.config.data_noise_ratio_y}"
        if self.config.data_noise_ratio_y == 0:
            print(f"\nUsing ratio {ratio_str} - No transition sentences (raw data dump).")
        else:
            print(f"\nUsing ratio {ratio_str} - Transition sentences will be added proportionally.")
        print()

    def prompt_validation(self):
        self.print_header("VALIDATION AND QUALITY ASSURANCE")
        print("Validation analyzes generated narratives for data integrity issues.")
        print()
        print("The validation process includes:")
        print("  - Narrative parsing analysis (detects missing columns)")
        print("  - Enhancement of narratives with missing sentences")
        print("  - Verification of 100% column detection")
        print()
        print("-" * 70)
        print("  WARNING: SKIPPING VALIDATION MAY RESULT IN:")
        print("-" * 70)
        print("  - DATA LOSS: Some column values may not appear in generated text")
        print("  - CONTEXT BLEEDING: Incorrect values may be substituted")
        print("  - MISSING INFORMATION: Important data fields may be omitted")
        print("  - LOWER QUALITY: Generated documents may be incomplete")
        print("-" * 70)
        print()
        print("1. INCLUDE VALIDATION (Recommended) - Ensures data integrity")
        print("2. SKIP VALIDATION (Faster but risky) - May cause errors")
        print()

        while True:
            user_input = input("Enter choice (1 for include, 2 for skip, default: 1): ").strip()

            if not user_input or user_input == "1":
                self.config.include_validation = True
                print("\nValidation ENABLED - Ensuring data integrity.")
                break
            elif user_input == "2":
                print()
                print("!" * 70)
                print("  YOU HAVE CHOSEN TO SKIP VALIDATION")
                print("  This may result in data loss and errors in generated documents.")
                print("!" * 70)
                confirm = input("\nAre you sure you want to skip validation? (yes/no): ").strip().lower()
                if confirm == "yes":
                    self.config.include_validation = False
                    print("\nValidation DISABLED - Proceeding without quality checks.")
                    break
                else:
                    print("\nReturning to validation selection...")
                    continue
            else:
                print("Invalid choice. Please enter 1 or 2.")

        print()

    def prompt_skip_phase1(self):
        """Ask whether to skip Phase 1 (generated sentence detection) when validation is enabled."""
        if not self.config.include_validation:
            self.config.skip_phase1_validation = False
            return

        self.print_header("PHASE 1 VALIDATION (GENERATED SENTENCE DETECTION)")
        print("Phase 1 checks that each generated sentence in templates contains detectable column data.")
        print("It remediates any sentences that fail detection (rewrites and molds into narrative).")
        print()
        print("Skip Phase 1 if you already have valid templates (e.g., from a previous run)")
        print("and only need Phase 2 (hash-based narrative validation).")
        print()
        print("1. RUN PHASE 1 (Recommended) - Full validation with remediation")
        print("2. SKIP PHASE 1 - Faster; use only when templates are known valid")
        print()

        while True:
            user_input = input("Enter choice (1 for run, 2 for skip, default: 1): ").strip()

            if not user_input or user_input == "1":
                self.config.skip_phase1_validation = False
                print("\nPhase 1 ENABLED - Full validation will run.")
                break
            elif user_input == "2":
                self.config.skip_phase1_validation = True
                print("\nPhase 1 SKIPPED - Proceeding to Phase 2 only.")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

        print()

    def load_available_tables(self) -> List[Tuple[str, str]]:
        dev_db_path = self.config.get_dev_databases_path()
        tables = []

        if not os.path.exists(dev_db_path):
            print(f"Error: dev_databases directory not found at {dev_db_path}")
            return tables

        databases = [d for d in os.listdir(dev_db_path)
                    if os.path.isdir(os.path.join(dev_db_path, d)) and not d.startswith('.')]

        import sqlite3

        for db_name in sorted(databases):
            sqlite_file = os.path.join(dev_db_path, db_name, f"{db_name}.sqlite")

            if not os.path.exists(sqlite_file):
                continue

            try:
                conn = sqlite3.connect(sqlite_file)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'sqlite_sequence';")
                db_tables = [row[0] for row in cursor.fetchall()]
                conn.close()

                for table_name in sorted(db_tables):
                    tables.append((db_name, table_name))
            except Exception as e:
                print(f"Warning: Could not read tables from {db_name}: {e}")

        self.available_tables = tables
        return tables

    def prompt_table_selection(self):
        self.print_header("DATABASE AND TABLE SELECTION")
        print("Choose which databases and tables to process:")
        print()
        print("1. ALL - Process all databases and tables")
        print("2. SELECT - Choose specific tables from a list")
        print()

        while True:
            user_input = input("Enter choice (1 for all, 2 for select, default: 1): ").strip()

            if not user_input or user_input == "1":
                self.load_available_tables()
                self.config.selected_tables = self.available_tables.copy()
                print(f"\nProcessing all {len(self.config.selected_tables)} tables.")
                return
            elif user_input == "2":
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

        self.load_available_tables()

        if not self.available_tables:
            print("No tables found in the database directory.")
            sys.exit(1)

        print()
        print("Available databases and tables:")
        print("-" * 50)

        for idx, (db_name, table_name) in enumerate(self.available_tables, 1):
            print(f"  {idx:3d}. {db_name}.{table_name}")

        print("-" * 50)
        print()
        print("Enter table numbers to process.")
        print("You can enter:")
        print("  - A range: 1-5")
        print("  - A comma-separated list: 1,3,5,7")
        print("  - A combination: 1-3,5,7-10")
        print()

        while True:
            user_input = input("Enter selection: ").strip()

            if not user_input:
                print("Please enter at least one table number.")
                continue

            selected_indices = self.parse_selection(user_input, len(self.available_tables))

            if selected_indices:
                self.config.selected_tables = [self.available_tables[i-1] for i in selected_indices]
                break
            else:
                print("Invalid selection. Please try again.")

        print()
        print(f"Selected {len(self.config.selected_tables)} tables:")
        for db_name, table_name in self.config.selected_tables:
            print(f"  - {db_name}.{table_name}")
        print()

    def prompt_documents_per_table(self):
        self.print_header("DOCUMENTS PER TABLE CONFIGURATION")
        print("Specify how many documents to generate per table.")
        print("This controls how many database rows are converted into text documents.")
        print()
        print("Options:")
        print("  - Enter a number (e.g., 10, 100, 1000) to limit documents per table")
        print("  - Enter 'all' to generate documents for all rows in each table")
        print()
        print("Recommendations:")
        print("  - 10: Quick testing and preview (default)")
        print("  - 100-1000: Moderate dataset generation")
        print("  - all: Full dataset (may take significant time for large tables)")
        print()

        while True:
            user_input = input("Enter number of documents per table (or 'all' for all rows, default: 10): ").strip().lower()

            if not user_input:
                self.config.documents_per_table = 10
                break
            elif user_input == "all":
                self.config.documents_per_table = None
                break
            else:
                try:
                    num = int(user_input)
                    if num >= 1:
                        self.config.documents_per_table = num
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'all'.")

        if self.config.documents_per_table is None:
            print("\nGenerating ALL documents for each table.")
        else:
            print(f"\nGenerating up to {self.config.documents_per_table} documents per table.")
        print()

    def parse_selection(self, selection: str, max_val: int) -> List[int]:
        indices = set()

        try:
            parts = selection.replace(" ", "").split(",")

            for part in parts:
                if "-" in part:
                    range_parts = part.split("-")
                    if len(range_parts) == 2:
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        if 1 <= start <= max_val and 1 <= end <= max_val:
                            indices.update(range(start, end + 1))
                else:
                    num = int(part)
                    if 1 <= num <= max_val:
                        indices.add(num)

            return sorted(list(indices))
        except ValueError:
            return []

    def confirm_and_start(self):
        self.print_header("CONFIGURATION SUMMARY")
        print(f"Data folder:          {self.config.data_folder_name}")
        print(f"Sentence variations:  {self.config.sentence_variations}")
        print(f"NULL mode:            {self.config.null_mode}")
        print(f"Binary mode:          {self.config.binary_mode}")
        ratio_str = f"{self.config.data_noise_ratio_x}:{self.config.data_noise_ratio_y}"
        if self.config.data_noise_ratio_y == 0:
            print(f"Data:Noise ratio:     {ratio_str} (no narrative weaving)")
        else:
            print(f"Data:Noise ratio:     {ratio_str}")
        print(f"Documents per table:  {'All' if self.config.documents_per_table is None else self.config.documents_per_table}")
        print(f"Include validation:   {'Yes' if self.config.include_validation else 'No (WARNING: May cause errors)'}")
        if self.config.include_validation:
            print(f"Skip Phase 1:         {'Yes' if self.config.skip_phase1_validation else 'No'}")
        print(f"Tables to process:    {len(self.config.selected_tables)}")
        print()
        print("-" * 70)
        print()
        print("GENERATION IS BEGINNING")
        print()
        print("This process may take a while depending on:")
        print("  - Number of tables selected")
        print("  - Number of sentence variations")
        print("  - Generation mode (sample vs full)")
        if self.config.include_validation:
            print("  - Validation steps")
        print()
        print("LLM costs will be tracked and displayed at the end.")
        print()
        print("Please do not interrupt the process.")
        print()

        time.sleep(4)

    def run_pipeline(self):
        from setup.generate_table_sample_text_files import generate_table_sample_text_for_database
        from setup.generate_column_descriptors import generate_descriptors_for_tables
        from pipeline.template_generator import TemplateGenerator
        from pipeline.document_assembler import process_templates_standalone
        from pipeline.data_to_text_converter import DataToTextConverter
        from pipeline.config import get_output_root

        self.cost_tracker.start_tracking()

        self.print_header("STEP 1: GENERATING TABLE SAMPLE TEXT FILES")

        for db_name, table_name in self.config.selected_tables:
            print(f"Processing {db_name}.{table_name}...")
            generate_table_sample_text_for_database(
                self.config.get_dev_databases_path(),
                db_name,
                table_name,
                self.config.base_dir,
                self.config.documents_per_table,
                dataset_folder_name=self.config.data_folder_name,
            )

        print()
        self.print_header("STEP 2: GENERATING COLUMN DESCRIPTORS")

        tables_by_db = {}
        for db_name, table_name in self.config.selected_tables:
            if db_name not in tables_by_db:
                tables_by_db[db_name] = []
            tables_by_db[db_name].append(table_name)

        generate_descriptors_for_tables(
            self.config.get_dev_databases_path(),
            tables_by_db,
            self.config.base_dir,
            dataset_folder_name=self.config.data_folder_name,
        )

        print()
        self.print_header("STEP 3: GENERATING NARRATIVE TEMPLATES")

        template_generator = TemplateGenerator(
            self.config.base_dir,
            null_mode=self.config.null_mode,
            binary_mode=self.config.binary_mode,
            data_noise_x=self.config.data_noise_ratio_x,
            data_noise_y=self.config.data_noise_ratio_y,
            dataset_folder_name=self.config.data_folder_name,
        )
        template_generator.generate_templates_for_tables(
            self.config.selected_tables,
            data_noise_x=self.config.data_noise_ratio_x,
            data_noise_y=self.config.data_noise_ratio_y
        )

        if self.config.include_validation:
            print()
            self.print_header("STEP 4: VALIDATION - ANALYZING NARRATIVE PARSING")
            self.run_validation()

            print()
            self.print_header("STEP 5: GENERATING SENTENCE VARIATIONS")
        else:
            print()
            self.print_header("STEP 4: GENERATING SENTENCE VARIATIONS (Validation Skipped)")

        process_templates_standalone(
            self.config.base_dir,
            self.config.selected_tables,
            self.config.sentence_variations,
            self.config.null_mode,
            self.config.binary_mode,
            data_noise_x=self.config.data_noise_ratio_x,
            data_noise_y=self.config.data_noise_ratio_y,
            dataset_folder_name=self.config.data_folder_name,
        )

        step_num = 6 if self.config.include_validation else 5
        print()
        self.print_header(f"STEP {step_num}: GENERATING FINAL DOCUMENTS")

        converter = DataToTextConverter(
            self.config.base_dir,
            null_mode=self.config.null_mode,
            binary_mode=self.config.binary_mode,
            data_noise_x=self.config.data_noise_ratio_x,
            data_noise_y=self.config.data_noise_ratio_y
        )
        converter.process_tables(
            self.config.selected_tables,
            self.config.documents_per_table
        )

        self.cost_tracker.stop_tracking()

        print()
        self.print_header("GENERATION COMPLETE")
        print("All documents have been generated successfully.")
        print()
        print("Output locations (under DocSets/):")
        print(f"  - Multi-column CSV:  documents/.../Multi Column/")
        print(f"  - Single-column CSV: documents/.../Single Column/")
        print(f"  - Text documents:    documents/.../Text/")
        print()

        self.cost_tracker.print_summary()

        cost_report_path = os.path.join(get_output_root(self.config.base_dir), "cost_report.json")
        self.cost_tracker.save_report(cost_report_path)
        print(f"Cost report saved to: {cost_report_path}")
        print()

    def run_validation(self):
        from pipeline.narrative_analyzer import NarrativeParsingAnalyzer
        from pipeline.config import get_mode_folder_name, get_output_root

        print("Analyzing narrative parsing for column detection...")
        print(f"Using NULL mode: {self.config.null_mode}, Binary mode: {self.config.binary_mode}")
        print(f"Data:Noise ratio: {self.config.data_noise_ratio_x}:{self.config.data_noise_ratio_y}")
        print(f"Validating {len(self.config.selected_tables)} selected tables...")

        mode_folder = get_mode_folder_name(self.config.null_mode, self.config.binary_mode)
        templates_dir = os.path.join(get_output_root(self.config.base_dir), "templates", mode_folder, "sentence_templates")

        analyzer = NarrativeParsingAnalyzer(
            templates_dir=templates_dir,
            null_mode=self.config.null_mode,
            binary_mode=self.config.binary_mode,
            selected_tables=self.config.selected_tables,
            data_noise_x=self.config.data_noise_ratio_x,
            data_noise_y=self.config.data_noise_ratio_y,
            dataset_folder_name=self.config.data_folder_name,
        )

        report = analyzer.analyze_all_templates(skip_phase1=self.config.skip_phase1_validation)

        if 'error' in report:
            print(f"Analysis failed: {report['error']}")
            return

        summary = report.get('summary', {})
        detection_rate = summary.get('overall_detection_rate', 0)

        print(f"Overall detection rate: {detection_rate:.1f}%")
        print(f"Tables analyzed: {summary.get('total_tables', 0)}")
        print(f"Columns detected: {summary.get('total_detected', 0)}/{summary.get('total_columns', 0)}")

    def run(self):
        self.clear_screen()
        self.print_header("TEMPLATE-BASED DOCUMENT GENERATION SYSTEM")
        print("This system converts structured database data into natural language documents.")
        print("Follow the prompts to configure the generation pipeline.")
        print()
        input("Press Enter to continue...")
        print()

        self.prompt_data_folder()
        self.prompt_sentence_variations()
        self.prompt_null_mode()
        self.prompt_binary_mode()
        self.prompt_data_noise_ratio()
        self.prompt_validation()
        self.prompt_skip_phase1()
        self.prompt_table_selection()
        self.prompt_documents_per_table()
        self.confirm_and_start()
        self.run_pipeline()


def main():
    pipeline = DocumentGenerationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
