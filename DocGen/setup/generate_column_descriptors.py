#!/usr/bin/env python3

import os
import sys
import sqlite3
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cost_tracker import get_cost_tracker, track_openai_response
from pipeline.config import (
    get_ground_truth_data_path,
    get_ground_truth_table_sample_description_dir,
    get_ground_truth_column_descriptors_enhanced_path,
    get_ground_truth_column_descriptors_path,
    get_ground_truth_context_cache_path,
    load_repo_dotenv,
)

_DOCGEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_repo_dotenv(_DOCGEN_DIR)


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def get_column_mapping(csv_file_path: str) -> Dict[str, Any]:
    column_info = {}
    if not os.path.exists(csv_file_path):
        return column_info
    
    try:
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(csv_file_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        original_name = row.get('original_column_name', '').strip()
                        column_name = row.get('column_name', '').strip()
                        description = row.get('column_description', '').strip()
                        value_description = row.get('value_description', '').strip()
                        data_format = row.get('data_format', '').strip()
                        
                        if original_name:
                            original_name = original_name.encode('utf-8').decode('utf-8-sig').strip()
                        if column_name:
                            column_name = column_name.encode('utf-8').decode('utf-8-sig').strip()
                        
                        if original_name:
                            column_info[original_name] = {
                                'original_name': original_name,
                                'description': description,
                                'value_description': value_description,
                                'data_format': data_format
                            }
                            
                            if column_name and column_name != original_name:
                                column_info[column_name] = {
                                    'original_name': original_name,
                                    'description': description,
                                    'value_description': value_description,
                                    'data_format': data_format
                                }
                break
            except UnicodeDecodeError:
                continue
                    
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
    
    return column_info


def get_full_csv_content(csv_file_path: str) -> str:
    if not os.path.exists(csv_file_path):
        return "No CSV structure available."
    
    try:
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(csv_file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    if len(content) > 2000:
                        lines = content.split('\n')
                        truncated_content = '\n'.join(lines[:min(10, len(lines))])
                        truncated_content += f"\n... (showing first 10 rows of {len(lines)} total rows)"
                        return truncated_content
                    return content
            except UnicodeDecodeError:
                continue
                
    except Exception as e:
        print(f"Error reading full CSV file {csv_file_path}: {e}")
    
    return "CSV structure could not be read."


def get_database_tables(db_path: str) -> List[str]:
    tables = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'sqlite_sequence';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error reading database {db_path}: {e}")
    
    return tables


def get_table_columns_with_types(db_path: str, table_name: str) -> List[Dict[str, Any]]:
    columns = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        for row in cursor.fetchall():
            columns.append({
                'name': row[1],  
                'type': row[2],  
                'not_null': bool(row[3]), 
                'primary_key': bool(row[5]) 
            })
        conn.close()
    except Exception as e:
        print(f"Error getting columns for table {table_name}: {e}")
    
    return columns


def get_sample_entries_from_table_data(
    base_dir: str,
    db_name: str,
    table_name: str,
    num_samples: int = 3,
    dataset_folder_name: str = "MINIDEV",
) -> List[Dict[str, str]]:
    sample_entries = []
    table_data_dir = os.path.join(
        get_ground_truth_table_sample_description_dir(base_dir, dataset_folder_name),
        db_name,
        table_name,
    )
    
    if not os.path.exists(table_data_dir):
        return sample_entries
    
    try:
        files = [f for f in os.listdir(table_data_dir) if f.endswith('.txt')]
        files.sort()
        
        sample_files = files[:min(num_samples, len(files))]
        
        for sample_file in sample_files:
            file_path = os.path.join(table_data_dir, sample_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    entry = {}
                    for line in content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            entry[key.strip()] = value.strip()
                    if entry:
                        sample_entries.append(entry)
            except Exception:
                continue
        
    except Exception:
        pass
    
    return sample_entries


def get_few_shot_examples_for_database_context() -> str:
    examples = """
EXAMPLE 1:
Database Name: california_schools
Sample Data Preview:
- Table 'frpm': CDSCode: 01100170109835, Academic Year: 2021-2022, County Name: Alameda, District Name: Alameda Unified, School Name: Lincoln Elementary, Free Meal Count: 245, Enrollment: 512, Percent Eligible Free: 0.478
- Table 'satscores': cds: 01100170000000, sname: Alameda Community Learning Center, NumTstTakr: 34, AvgScrRead: 482, AvgScrWrite: 478
- Table 'schools': CDSCode: 01100170109835, StatusType: Active, County: Alameda, District: Alameda Unified, School: Lincoln Elementary, Street: 1234 Main St

Database Context: Contains data about California public schools including enrollment statistics, free/reduced meal program eligibility data, SAT test performance scores, and administrative details about school locations and status.

---

EXAMPLE 2:
Database Name: formula_1
Sample Data Preview:
- Table 'drivers': driverId: 1, driverRef: hamilton, number: 44, code: HAM, forename: Lewis, surname: Hamilton, nationality: British
- Table 'races': raceId: 1, year: 2009, round: 1, circuitId: 1, name: Australian Grand Prix, date: 2009-03-29
- Table 'results': resultId: 1, raceId: 1, driverId: 1, position: 1, points: 10, laps: 58, time: 1:34:15.757

Database Context: Contains Formula 1 racing data including driver biographical information, race schedules and details, championship results, constructor/team information, circuit specifications, and detailed race performance statistics.

---

EXAMPLE 3:
Database Name: financial
Sample Data Preview:
- Table 'account': account_id: 1, district_id: 18, frequency: POPLATEK MESICNE, date: 1995-03-24
- Table 'client': client_id: 1, gender: F, birth_date: 1970-12-13, district_id: 18
- Table 'loan': loan_id: 4959, account_id: 2, date: 1994-01-05, amount: 80952, duration: 24, status: A

Database Context: Contains banking and financial services data including customer account information, client demographics, loan agreements with amounts and durations, transaction histories, credit card details, and geographic district information for a Czech banking institution.
"""
    return examples


def get_few_shot_examples_for_table_context() -> str:
    examples = """
EXAMPLE 1:
Database: california_schools
Table Name: frpm
Sample Entry: {CDSCode: 01100170109835, Academic Year: 2021-2022, County Name: Alameda, Free Meal Count K-12: 245, Enrollment K-12: 512, Percent Eligible Free K-12: 0.478}
Column Names: CDSCode, Academic Year, County Code, District Code, School Code, Charter School Y/N, Free Meal Count K-12, Enrollment K-12, Percent Eligible Free K-12

Table Context: Free and Reduced Price Meal (FRPM) eligibility data for California schools, tracking student enrollment counts and the percentage of students qualifying for free or reduced-price meals under federal nutrition programs.

---

EXAMPLE 2:
Database: european_football_2
Table Name: Player_Attributes
Sample Entry: {id: 1, player_fifa_api_id: 505942, player_api_id: 505942, date: 2016-02-18, overall_rating: 67, potential: 71, preferred_foot: right}
Column Names: id, player_fifa_api_id, player_api_id, date, overall_rating, potential, preferred_foot, attacking_work_rate, defensive_work_rate

Table Context: Player skill ratings and performance attributes from FIFA video game data, including overall ratings, potential scores, preferred foot, and various attacking/defensive work rate classifications for European football players.

---

EXAMPLE 3:
Database: toxicology
Table Name: molecule
Sample Entry: {molecule_id: TR000, label: +}
Column Names: molecule_id, label

Table Context: Chemical compound molecular structure data for toxicology studies, with molecule identifiers and toxicity labels indicating whether compounds are carcinogenic (+) or non-carcinogenic (-).
"""
    return examples


def infer_database_context_with_llm(client: OpenAI, db_name: str, tables: List[str], 
                                     sample_data: Dict[str, List[Dict[str, str]]]) -> str:
    sample_preview = []
    for table_name, entries in sample_data.items():
        if entries:
            entry = entries[0]
            entry_preview = {k: v for i, (k, v) in enumerate(entry.items()) if i < 5}
            sample_preview.append(f"- Table '{table_name}': {entry_preview}")
    
    sample_text = '\n'.join(sample_preview) if sample_preview else "No sample data available."
    
    few_shot_examples = get_few_shot_examples_for_database_context()
    
    prompt = f"""Analyze the following database and its sample data to describe what kind of data this database contains.

{few_shot_examples}

---

NOW ANALYZE THIS DATABASE:

Database Name: {db_name}
Tables: {', '.join(tables)}

Sample Data Preview:
{sample_text}

Based on the database name, table names, and sample data values, write a ONE-SENTENCE description of what this database contains. Focus on:
1. The domain/subject area (e.g., education, sports, finance, healthcare)
2. The main types of data stored (e.g., player statistics, transaction records, medical tests)
3. Any notable characteristics of the data

Database Context:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert database analyst. Given sample data from a database, you can accurately describe what the database contains and its purpose. Write concise, informative descriptions that capture the essence of the data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        track_openai_response(response, "gpt-4o")
        
        context = response.choices[0].message.content.strip()
        if context.startswith('"') and context.endswith('"'):
            context = context[1:-1]
        
        return context
        
    except Exception as e:
        print(f"Error inferring database context: {e}")
        return f"Database containing {db_name.replace('_', ' ')} related data"


def infer_table_context_with_llm(client: OpenAI, db_name: str, table_name: str, 
                                  columns: List[Dict[str, Any]], 
                                  sample_entries: List[Dict[str, str]],
                                  database_context: str) -> str:
    column_names = [col['name'] for col in columns]
    
    sample_text = "No sample data available."
    if sample_entries:
        entry = sample_entries[0]
        entry_preview = {k: v for i, (k, v) in enumerate(entry.items()) if i < 8}
        sample_text = str(entry_preview)
    
    few_shot_examples = get_few_shot_examples_for_table_context()
    
    prompt = f"""Analyze the following table and its sample data to describe what kind of data this table contains.

{few_shot_examples}

---

NOW ANALYZE THIS TABLE:

Database: {db_name}
Database Context: {database_context}
Table Name: {table_name}
Column Names: {', '.join(column_names[:15])}{'...' if len(column_names) > 15 else ''}
Sample Entry: {sample_text}

Based on the table name, column names, and sample data, write a ONE-SENTENCE description of what this table contains and its purpose within the database. Be specific about the type of records stored.

Table Context:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert database analyst. Given sample data from a table, you can accurately describe what the table contains. Write concise, informative descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        track_openai_response(response, "gpt-4o")
        
        context = response.choices[0].message.content.strip()
        if context.startswith('"') and context.endswith('"'):
            context = context[1:-1]
        
        return context
        
    except Exception as e:
        print(f"Error inferring table context: {e}")
        return f"Data table containing {table_name.replace('_', ' ')} information"


def generate_descriptor_with_llm(client: OpenAI, database_name: str, table_name: str, 
                                  column_info: Dict[str, Any], database_context: str, 
                                  table_context: str, csv_description: str = "", 
                                  value_description: str = "") -> str:
    column_name = column_info['name']
    column_type = column_info['type']
    
    csv_context = ""
    if csv_description and csv_description.lower() not in ['n/a', '', 'null', 'none']:
        csv_context = f"\nCSV Description: {csv_description}"
        
        excluded_values = ['n/a', '', 'null', 'none', 'unuseful', 'not useful', 'unusable']
        if (value_description and value_description.lower().strip() not in excluded_values):
            csv_context += f"\nValue Context: {value_description}"
    
    prompt = f"""You are creating natural language descriptors for database columns. Your goal is to create ONE clear, comprehensive sentence that explains what the column actually measures or represents.

REQUIREMENTS:
1. Expand ALL abbreviations and acronyms (e.g., NSLP = National School Lunch Program)
2. Focus on what the column actually measures/describes, not whether it's legacy or useful
3. Include context about how this relates to the broader table/database purpose
4. Write exactly ONE rich, informative sentence
5. Make it clear and understandable to a general audience
6. Use the CSV description when available to ensure accuracy
7. If Value Context is provided, incorporate it to add clarity and specificity

DATABASE CONTEXT: {database_name} - {database_context}
TABLE CONTEXT: {table_name} - {table_context}

COLUMN DETAILS:
- Column Name: {column_name}
- Data Type: {column_type}{csv_context}

EXAMPLES:
- "CDSCode" -> "Unique 14-digit California Department of Education identifier that combines county, district, and school codes to precisely locate any educational institution within the state's administrative hierarchy."
- "NSLP Provision Status" -> "Indicates the National School Lunch Program provision status that determines how the school participates in federal meal assistance programs."
- "Charter School (Y/N)" -> "Binary indicator showing whether the institution operates as a charter school under special state authorization rather than traditional district governance."
- "enroll12" with CSV "enrollment (1st-12nd grade)" -> "Total number of students enrolled in grades 1 through 12, representing the complete K-12 student population at the school."
- "NumTstTakr" with CSV "Number of Test Takers in this school" + Value Context "number of test takers in each school" -> "Number of students who participated in standardized testing at each individual school, providing insight into testing engagement levels."

Generate a rich, informative descriptor for the column "{column_name}":"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at creating clear, educational descriptions of database fields. Always expand abbreviations and focus on what the data actually represents. Create rich, comprehensive descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        track_openai_response(response, "gpt-4o-mini")
        
        descriptor = response.choices[0].message.content.strip()
        if descriptor.startswith('"') and descriptor.endswith('"'):
            descriptor = descriptor[1:-1]
        
        return descriptor
        
    except Exception as e:
        print(f"Error generating descriptor for {column_name}: {e}")
        return f"Data field representing {column_name.lower().replace('_', ' ')} information."


class ContextCache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"databases": {}, "tables": {}}
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Could not save context cache: {e}")
    
    def get_database_context(self, db_name: str) -> Optional[str]:
        return self.cache.get("databases", {}).get(db_name)
    
    def set_database_context(self, db_name: str, context: str):
        if "databases" not in self.cache:
            self.cache["databases"] = {}
        self.cache["databases"][db_name] = context
    
    def get_table_context(self, db_name: str, table_name: str) -> Optional[str]:
        key = f"{db_name}.{table_name}"
        return self.cache.get("tables", {}).get(key)
    
    def set_table_context(self, db_name: str, table_name: str, context: str):
        if "tables" not in self.cache:
            self.cache["tables"] = {}
        key = f"{db_name}.{table_name}"
        self.cache["tables"][key] = context


def load_existing_descriptors(output_file: str) -> Dict[str, Any]:
    """Load existing descriptors from the output file."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_base_descriptors(base_dir: str) -> Dict[str, Any]:
    """
    Load existing descriptors from column_descriptors.json to check what's already processed.
    This prevents re-processing columns that already have descriptors.
    """
    base_file = get_ground_truth_column_descriptors_path(base_dir)
    
    if os.path.exists(base_file):
        try:
            with open(base_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  Loaded existing descriptors from column_descriptors.json")
                return data
        except Exception as e:
            print(f"  Warning: Could not load column_descriptors.json: {e}")
    
    return {}


def is_column_already_processed(base_descriptors: Dict[str, Any], db_name: str, 
                                 table_name: str, column_name: str) -> bool:
    """
    Check if a column has already been processed in the base descriptors file.
    Returns True if the column exists and has a valid descriptor.
    """
    if db_name not in base_descriptors:
        return False
    
    if table_name not in base_descriptors[db_name]:
        return False
    
    table_data = base_descriptors[db_name][table_name]
    
    if column_name not in table_data:
        return False
    
    # Check if descriptor exists and is not empty
    column_data = table_data[column_name]
    
    if isinstance(column_data, dict):
        descriptor = column_data.get('descriptor', '')
        return bool(descriptor and descriptor.strip())
    elif isinstance(column_data, str):
        # Old format where value is directly the descriptor string
        return bool(column_data and column_data.strip())
    
    return False


def get_existing_descriptor(base_descriptors: Dict[str, Any], db_name: str,
                            table_name: str, column_name: str) -> Optional[Dict[str, Any]]:
    """
    Get existing descriptor data for a column from base descriptors.
    Returns None if not found.
    """
    if db_name not in base_descriptors:
        return None
    
    if table_name not in base_descriptors[db_name]:
        return None
    
    table_data = base_descriptors[db_name][table_name]
    
    if column_name not in table_data:
        return None
    
    column_data = table_data[column_name]
    
    if isinstance(column_data, dict):
        return column_data
    elif isinstance(column_data, str):
        # Convert old format to new format
        return {'descriptor': column_data}
    
    return None


def generate_descriptors_for_tables(
    dev_databases_path: str,
    tables_by_db: Dict[str, List[str]],
    base_dir: str,
    dataset_folder_name: str = "MINIDEV",
):
    client = get_openai_client()
    cost_tracker = get_cost_tracker()
    
    output_file = get_ground_truth_column_descriptors_enhanced_path(base_dir)
    cache_file = get_ground_truth_context_cache_path(base_dir, dataset_folder_name)
    
    context_cache = ContextCache(cache_file)
    all_descriptors = load_existing_descriptors(output_file)
    
    base_descriptors = load_base_descriptors(base_dir)
    
    total_skipped = 0
    total_processed = 0
    total_reused = 0
    cached_db_contexts = 0
    cached_table_contexts = 0
    
    for db_name, table_names in tables_by_db.items():
        print(f"\nProcessing database: {db_name}")
        
        db_dir = os.path.join(dev_databases_path, db_name)
        sqlite_file = os.path.join(db_dir, f"{db_name}.sqlite")
        description_dir = os.path.join(db_dir, "database_description")
        
        if not os.path.exists(sqlite_file):
            print(f"  Warning: SQLite file not found: {sqlite_file}")
            continue
        
        # Initialize database entry and get context
        all_tables = get_database_tables(sqlite_file)
        
        if db_name not in all_descriptors:
            all_descriptors[db_name] = {}

        # One sample row per table (up to 5 tables) for LLM database-level context.
        # Must run whenever we infer context — not only when db_name is new in all_descriptors
        # (otherwise mid-migration / resumed runs hit NameError on db_sample_data).
        db_sample_data: Dict[str, List[Dict[str, str]]] = {}
        for tbl in all_tables[:5]:
            sample_entries = get_sample_entries_from_table_data(
                base_dir, db_name, tbl, num_samples=1, dataset_folder_name=dataset_folder_name
            )
            if sample_entries:
                db_sample_data[tbl] = sample_entries

        database_context = context_cache.get_database_context(db_name)
        if not database_context:
            print(f"  Inferring database context...")
            database_context = infer_database_context_with_llm(client, db_name, all_tables, db_sample_data)
            context_cache.set_database_context(db_name, database_context)
            context_cache.save_cache()
        else:
            cached_db_contexts += 1
        
        print(f"  Database Context: {database_context[:100]}...")
        
        for table_name in table_names:
            print(f"\n  Processing table: {table_name}")
            
            if table_name not in all_descriptors[db_name]:
                all_descriptors[db_name][table_name] = {}
            
            columns = get_table_columns_with_types(sqlite_file, table_name)
            
            columns_to_process = []
            columns_from_base = []
            columns_already_enhanced = []
            
            for col in columns:
                col_name = col['name']
                
                if col_name in all_descriptors[db_name][table_name]:
                    columns_already_enhanced.append(col)
                elif is_column_already_processed(base_descriptors, db_name, table_name, col_name):
                    columns_from_base.append(col)
                else:
                    columns_to_process.append(col)
            
            print(f"    Columns: {len(columns)} total, {len(columns_already_enhanced)} already enhanced, "
                  f"{len(columns_from_base)} in base file, {len(columns_to_process)} new")
            
            if columns_from_base:
                print(f"    Reusing {len(columns_from_base)} descriptors from column_descriptors.json")
                for col in columns_from_base:
                    col_name = col['name']
                    existing_data = get_existing_descriptor(base_descriptors, db_name, table_name, col_name)
                    
                    if existing_data:
                        all_descriptors[db_name][table_name][col_name] = {
                            'descriptor': existing_data.get('descriptor', ''),
                            'data_type': existing_data.get('data_type', col['type']),
                            'original_name': existing_data.get('original_name', col_name),
                            'is_primary_key': existing_data.get('is_primary_key', col['primary_key'])
                        }
                        total_reused += 1
            
            total_skipped += len(columns_already_enhanced)
            
            if not columns_to_process:
                if columns_already_enhanced or columns_from_base:
                    print(f"    All columns accounted for - no new processing needed")
                continue
            
            print(f"    Processing {len(columns_to_process)} new columns...")
            
            sample_entries = get_sample_entries_from_table_data(
                base_dir, db_name, table_name, num_samples=3, dataset_folder_name=dataset_folder_name
            )
            
            table_context = context_cache.get_table_context(db_name, table_name)
            if not table_context:
                print(f"    Inferring table context...")
                table_context = infer_table_context_with_llm(
                    client, db_name, table_name, columns, sample_entries, database_context
                )
                context_cache.set_table_context(db_name, table_name, table_context)
                context_cache.save_cache()
            else:
                cached_table_contexts += 1
            
            print(f"    Table Context: {table_context[:80]}...")
            
            csv_file = os.path.join(description_dir, f"{table_name}.csv")
            column_mappings = get_column_mapping(csv_file)
            
            for column in columns_to_process:
                column_name = column['name']
                print(f"      Generating descriptor for: {column_name}")
                
                csv_info = column_mappings.get(column_name, {})
                csv_description = csv_info.get('description', '')
                value_description = csv_info.get('value_description', '')
                original_name = csv_info.get('original_name', column_name)
                
                descriptor = generate_descriptor_with_llm(
                    client, db_name, table_name, column, database_context, table_context,
                    csv_description, value_description
                )
                
                all_descriptors[db_name][table_name][column_name] = {
                    'descriptor': descriptor,
                    'data_type': column['type'],
                    'original_name': original_name,
                    'is_primary_key': column['primary_key']
                }
                
                total_processed += 1
                print(f"        Generated: {descriptor[:60]}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_descriptors, f, ensure_ascii=False, indent=2)
    
    cost_tracker.track_skipped_descriptors(total_skipped)
    cost_tracker.track_reused_descriptors(total_reused)
    cost_tracker.track_skipped_db_contexts(cached_db_contexts)
    cost_tracker.track_skipped_table_contexts(cached_table_contexts)
    
    print(f"\n" + "="*60)
    print(f"DESCRIPTOR GENERATION SUMMARY")
    print(f"="*60)
    print(f"  Columns reused from column_descriptors.json: {total_reused}")
    print(f"  Columns already in enhanced file (skipped): {total_skipped}")
    print(f"  Database contexts from cache: {cached_db_contexts}")
    print(f"  Table contexts from cache: {cached_table_contexts}")
    print(f"  New columns processed with LLM: {total_processed}")
    print(f"  Total columns in output: {total_reused + total_skipped + total_processed}")
    print(f"\nColumn descriptors saved to: {output_file}")
    context_cache.save_cache()
    print(f"Context cache saved to: {cache_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    minidev_path = get_ground_truth_data_path(base_dir, "MINIDEV")
    dev_databases_path = os.path.join(minidev_path, "dev_databases")
    base_descriptors_file = get_ground_truth_column_descriptors_path(base_dir)
    output_file = get_ground_truth_column_descriptors_enhanced_path(base_dir)
    
    print("="*60)
    print("COLUMN DESCRIPTOR GENERATOR")
    print("="*60)
    
    if not os.path.exists(minidev_path):
        print(f"Error: MINIDEV directory not found at {minidev_path}")
        return
    
    if not os.path.exists(dev_databases_path):
        print(f"Error: dev_databases directory not found at {dev_databases_path}")
        return
    
    # Check for existing base descriptors
    if os.path.exists(base_descriptors_file):
        base_descriptors = load_base_descriptors(base_dir)
        total_existing = sum(
            len(tables) for db, tables in base_descriptors.items()
            for table, cols in (tables.items() if isinstance(tables, dict) else [])
        )
        print(f"Found existing column_descriptors.json with data for {len(base_descriptors)} databases")
    else:
        print(f"No existing column_descriptors.json found - will generate all descriptors")
    
    print(f"Output will be saved to: {output_file}")
    print()
    
    databases = [d for d in os.listdir(dev_databases_path) 
                if os.path.isdir(os.path.join(dev_databases_path, d)) and not d.startswith('.')]
    
    tables_by_db = {}
    for db_name in databases:
        sqlite_file = os.path.join(dev_databases_path, db_name, f"{db_name}.sqlite")
        if os.path.exists(sqlite_file):
            tables_by_db[db_name] = get_database_tables(sqlite_file)
    
    print(f"Found {len(databases)} databases with {sum(len(t) for t in tables_by_db.values())} total tables")
    print()
    
    generate_descriptors_for_tables(dev_databases_path, tables_by_db, base_dir, dataset_folder_name="MINIDEV")


if __name__ == "__main__":
    main()
