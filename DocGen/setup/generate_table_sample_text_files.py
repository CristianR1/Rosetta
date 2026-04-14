#!/usr/bin/env python3

import os
import sys
import sqlite3
import csv
from pathlib import Path
from typing import Dict, List, Optional

_DOCGEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DOCGEN_DIR not in sys.path:
    sys.path.insert(0, _DOCGEN_DIR)
from pipeline.config import get_ground_truth_data_path, get_ground_truth_table_sample_description_dir


def ensure_directory_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_column_mapping(csv_file_path: str) -> Dict[str, str]:
    column_mapping = {}
    try:
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(csv_file_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        original_name = row.get('original_column_name', '').strip()
                        column_name = row.get('column_name', '').strip()
                        
                        if original_name:
                            original_name = original_name.encode('utf-8').decode('utf-8-sig').strip()
                        if column_name:
                            column_name = column_name.encode('utf-8').decode('utf-8-sig').strip()
                        
                        if not original_name:
                            continue
                        
                        if column_name:
                            column_mapping[column_name] = original_name
                        else:
                            column_mapping[original_name] = original_name
                break
            except UnicodeDecodeError:
                continue
                    
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
    
    return column_mapping


def find_csv_file(description_dir: str, table_name: str) -> Optional[str]:
    exact_path = os.path.join(description_dir, f"{table_name}.csv")
    if os.path.exists(exact_path):
        return exact_path
    
    if os.path.exists(description_dir):
        for file in os.listdir(description_dir):
            if file.lower() == f"{table_name.lower()}.csv":
                return os.path.join(description_dir, file)
    
    return None


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


def get_table_columns(db_path: str, table_name: str) -> List[str]:
    columns = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error getting columns for table {table_name}: {e}")
    
    return columns


def write_row_sample_text_files_for_table(db_path: str, table_name: str, column_mapping: Dict[str, str], output_dir: str, limit: Optional[int] = None):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
        total_rows = cursor.fetchone()[0]
        
        if limit is not None:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit};")
        else:
            cursor.execute(f"SELECT * FROM `{table_name}`;")
        
        columns = [description[0] for description in cursor.description]
        
        table_dir = os.path.join(output_dir, table_name)
        ensure_directory_exists(table_dir)
        
        record_count = 0
        for row in cursor.fetchall():
            record_count += 1
            
            text_file_path = os.path.join(table_dir, f"{table_name}{record_count}.txt")
            
            with open(text_file_path, 'w', encoding='utf-8') as f:
                for i, value in enumerate(row):
                    column_name = columns[i]
                    
                    original_name = column_mapping.get(column_name, column_name)
                    
                    if value is None:
                        value = "NULL"
                    
                    f.write(f"{original_name}: {value}\n")
        
        if limit is not None and total_rows > limit:
            print(f"Generated {record_count} of {total_rows} text files for table '{table_name}'")
        else:
            print(f"Generated {record_count} text files for table '{table_name}'")
        conn.close()
        
    except Exception as e:
        print(f"Error processing table {table_name}: {e}")


def generate_table_sample_text_for_database(
    dev_databases_path: str,
    db_name: str,
    table_name: str,
    base_dir: str,
    limit: Optional[int] = None,
    dataset_folder_name: str = "MINIDEV",
):
    output_base_dir = get_ground_truth_table_sample_description_dir(base_dir, dataset_folder_name)
    ensure_directory_exists(output_base_dir)
    
    db_dir = os.path.join(dev_databases_path, db_name)
    sqlite_file = os.path.join(db_dir, f"{db_name}.sqlite")
    description_dir = os.path.join(db_dir, "database_description")
    
    db_output_dir = os.path.join(output_base_dir, db_name)
    ensure_directory_exists(db_output_dir)
    
    if not os.path.exists(sqlite_file):
        print(f"Warning: SQLite file not found: {sqlite_file}")
        return
    
    csv_file = find_csv_file(description_dir, table_name)
    
    if csv_file:
        column_mapping = get_column_mapping(csv_file)
        print(f"Found column mapping CSV: {len(column_mapping)} columns")
    else:
        print(f"Warning: CSV file not found for {table_name}, using database column names")
        db_columns = get_table_columns(sqlite_file, table_name)
        column_mapping = {col: col for col in db_columns}
    
    write_row_sample_text_files_for_table(sqlite_file, table_name, column_mapping, db_output_dir, limit)


def generate_all_table_sample_text_files(
    dev_databases_path: str,
    base_dir: str,
    limit: int = 1000,
    dataset_folder_name: str = "MINIDEV",
):
    output_base_dir = get_ground_truth_table_sample_description_dir(base_dir, dataset_folder_name)
    ensure_directory_exists(output_base_dir)
    
    if not os.path.exists(dev_databases_path):
        print(f"Error: dev_databases directory not found at {dev_databases_path}")
        return
    
    databases = [d for d in os.listdir(dev_databases_path) 
                if os.path.isdir(os.path.join(dev_databases_path, d)) and not d.startswith('.')]
    
    total_tables = 0
    processed_tables = 0
    
    for db_name in databases:
        print(f"\nProcessing database: {db_name}")
        
        db_dir = os.path.join(dev_databases_path, db_name)
        sqlite_file = os.path.join(db_dir, f"{db_name}.sqlite")
        description_dir = os.path.join(db_dir, "database_description")
        
        db_output_dir = os.path.join(output_base_dir, db_name)
        ensure_directory_exists(db_output_dir)
        
        if not os.path.exists(sqlite_file):
            print(f"  Warning: SQLite file not found: {sqlite_file}")
            continue
        
        tables = get_database_tables(sqlite_file)
        total_tables += len(tables)
        
        for table_name in tables:
            print(f"  Processing table: {table_name}")
            
            csv_file = find_csv_file(description_dir, table_name)
            
            if csv_file:
                column_mapping = get_column_mapping(csv_file)
                print(f"    Found column mapping CSV: {len(column_mapping)} columns")
            else:
                print(f"    Warning: CSV file not found for {table_name}")
                db_columns = get_table_columns(sqlite_file, table_name)
                column_mapping = {col: col for col in db_columns}
            
            write_row_sample_text_files_for_table(sqlite_file, table_name, column_mapping, db_output_dir, limit)
            processed_tables += 1
    
    print(f"\nGeneration completed!")
    print(f"  Databases processed: {len(databases)}")
    print(f"  Total tables: {total_tables}")
    print(f"  Files generated: {processed_tables}")
    print(f"  Files saved to: {output_base_dir}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    minidev_path = get_ground_truth_data_path(base_dir, "MINIDEV")
    dev_databases_path = os.path.join(minidev_path, "dev_databases")
    
    if not os.path.exists(dev_databases_path):
        print(f"Error: dev_databases directory not found at {dev_databases_path}")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        print("Starting LIMITED file generation (up to 1000 files per table)...")
        generate_all_table_sample_text_files(dev_databases_path, base_dir, limit=1000)
    else:
        print("Starting FULL file generation (all records)...")
        print("Tip: Use --single flag to generate up to 1000 files per table")
        generate_all_table_sample_text_files(dev_databases_path, base_dir, limit=None)


if __name__ == "__main__":
    main()
