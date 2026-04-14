# Rosetta

**Rosetta** is a benchmarking framework that converts structured Text-to-SQL resources into document-centric benchmarks for evaluating Semantic Data Processing Systems (SDPSs). It bridges the gap between structured relational benchmarks and unstructured document system evaluation by providing two coordinated pipelines: one that generates diverse, controllable document corpora from relational data, and one that compiles SQL queries into system-agnostic semantic execution pipelines that can be run against LOTUS, DocETL, and Palimpzest.

Existing Text-to-SQL datasets (like BIRD MINIDEV) already provide structured data, natural language queries, gold SQL, and ground-truth answers. Rosetta reuses all of this to generate rigorous SDPS benchmarks without manual curation. SQL-grounded answers ensure evaluation correctness, while systematic control over document representation, noise injection, and corpus size enables fair, reproducible comparison across systems.

---

## Repository Layout

```
Rosetta/
├── requirements.txt                # Shared Python deps (use a dedicated venv; see Setup)
├── Config/
│   └── .env                        # ← API keys
│
├── GroundTruth/                     # Input datasets and shared descriptors
│   ├── column_descriptors_enhanced.json   # LLM-generated column descriptors
│   └── MINIDEV/                           # Place your BIRD MINIDEV dataset here
│       ├── dev_databases/                 #   one folder per database with .sqlite files
│       ├── dev_tables.json
│       ├── mini_dev_sqlite_gold.sql
│       ├── mini_dev_postgresql_gold.sql
│       ├── mini_dev_sqlite.json
│       ├── mini_dev_postgresql.json
│       └── table_sample_data/
│           └── description/               # Generated per-row text files (Step 1 output)
│
├── DocGen/                          # Pipeline 1 — Document Generation
│   ├── pipeline/                    # Core pipeline modules
│   │   ├── pipeline_orchestrator.py #   Interactive entry point
│   │   ├── template_generator.py    #   Narrative sentence template generation (OpenAI)
│   │   ├── variation_generator.py   #   Structural variation bank (local LLM)
│   │   ├── data_to_text_converter.py#   Row data → final text documents
│   │   ├── narrative_analyzer.py    #   Hash-based column detection & validation
│   │   ├── variation_validator.py   #   Post-generation placeholder validation
│   │   ├── document_assembler.py    #   Sentence variation assembly
│   │   ├── template_system.py       #   DocumentTemplateSystem coordinator
│   │   ├── field_detection.py       #   Binary/null/misc field classification
│   │   ├── config.py                #   All shared path helpers
│   │   └── data_loader.py           #   SQLite + descriptor I/O
│   ├── setup/
│   │   ├── generate_table_sample_text_files.py  # Dump SQLite rows to table_sample_data/description .txt
│   │   └── generate_column_descriptors.py      # LLM descriptor generation (OpenAI)
│   └── utils/
│       ├── cost_tracker.py          # Token / cost accounting
│       └── analyze_results.py       # Results analysis helpers
│
├── DocSets/                         # Pipeline 1 output — generated document corpora
│   ├── templates/
│   │   └── {null_mode}/
│   │       ├── sentence_templates/  # One JSON per table, hash-tagged sentences
│   │       └── narrative_templates/
│   │           └── {noise_ratio}/   # Narrative wrappers (data + noise sentences)
│   ├── variations/
│   │   └── {null_mode}/{noise_ratio}/  # Sentence variation banks per table
│   ├── documents/
│   │   └── {null_mode}/{noise_ratio}/  # Final generated text documents
│   │       ├── Text/                   #   Full narrative .txt files
│   │       ├── Multi Column/           #   CSV (one column per field)
│   │       └── Single Column/          #   CSV (single contents column)
│   ├── tbd_sentence_cache.json      # LLM cache for complex-value sentences
│   └── cost_report.json             # Running token / cost summary
│
├── SemPipelineBuilder/              # Pipeline 2 — Semantic Pipeline Extraction
│   ├── explain_queries.py           # SQLite EXPLAIN QUERY PLAN → execution trees
│   ├── postgres_explain_pipeline.py # PostgreSQL EXPLAIN (FORMAT JSON) → operator DAGs
│   ├── explain_parser.py            # Parse physical plans into semantic pipeline dicts
│   ├── docker-compose.yml           # Spins up postgres:15 for plan collection
│   └── pipelines/                   # Output: explain_trees*.json + summary .txt
│
├── SemPipelineExecutor/             # Pipeline 2 — SDPS Evaluation
│   ├── rosetta_env.py               # Shared env loader + results path helper
│   ├── build_pipelines.py           # Merge questions + explain trees → pipelines.json
│   ├── pipeline_sources.py          # Canonical table → database name mapping
│   ├── semantic_executor.py         # System-agnostic semantic plan executor
│   ├── lotus_pipeline.py            # LOTUS backend
│   ├── docetl_pipeline.py           # DocETL backend
│   ├── palimpzest_pipeline.py       # Palimpzest backend
│   ├── run_evaluation.py            # Orchestrator — runs all three systems
│   └── sem-pipelines/
│       └── pipelines.json           # Pre-built pipeline cache
│
└── Results/                         # All evaluation results (created at runtime)
    └── {system}/{null_mode}/{noise_ratio}/{N}_documents/
        ├── {system}_metrics.csv
        └── {system}_metrics_summary.csv
```

---

## How It Works

### Pipeline 1 — Document Generation (`DocGen/`)

Rosetta converts rows from relational BIRD databases into readable text documents. Each document represents one database row expressed as a coherent natural-language narrative.

```
SQLite tables
      │
      ▼
[Step 1] generate_table_sample_text_files.py
         Exports every row to a plain-text sample file under
         GroundTruth/MINIDEV/table_sample_data/description/<db>/<table>/
      │
      ▼
[Step 2] generate_column_descriptors.py
         Calls OpenAI (GPT-4o) to produce semantic descriptors for every column,
         stored in GroundTruth/column_descriptors_enhanced.json
      │
      ▼
[Step 3] TemplateGenerator (template_generator.py)
         Generates one hash-tagged sentence per column using OpenAI.
         Sentences are grouped into a sentence template JSON per table.
         If noise ratio > 0, a second narrative pass wraps them in a
         flowing narrative with filler (noise) sentences.
      │
      ▼
[Step 4] NarrativeParsingAnalyzer (narrative_analyzer.py)
         Validates that each column's value can be round-tripped back
         from its sentence via hash-based detection. Failed sentences
         are regenerated automatically.
      │
      ▼
[Step 5] process_templates_standalone (document_assembler.py)
         VariationBankGenerator calls the local LLM (Bellatrix) to expand
         each sentence into N structural variations, plus counter-variations
         for binary/null fields.  Results are saved as variation bank JSONs.
      │
      ▼
[Step 6] DataToTextConverter (data_to_text_converter.py)
         For every row in the database, picks one variation per field,
         replaces [COLUMN_NAME] placeholders with actual values, assembles
         a full document, and writes Text / CSV outputs under DocSets/.
```

**Controllable dimensions:**
| Dimension | DocSets path segment | CLI flag |
|-----------|---------------------|----------|
| Null/binary exposure representation | `{null_mode}` (e.g. `null_binary_explicit`) | `--null-mode`, `--binary-mode` |
| Data-to-noise sentence ratio | `{noise_ratio}` (e.g. `1_data_1_noise`) | `--data-noise-ratio` |
| Document count per table | Chosen at runtime | `documents_per_table` in config |

### Pipeline 2 — Semantic Pipeline Extraction & Evaluation

Rosetta compiles each gold SQL query into a system-agnostic semantic execution plan (a sequence of `sem_filter`, `sem_join`, `sem_agg`, `sem_extract`, `sem_topk`,`sem_clusterby` operations with natural-language instructions), then runs that plan against the generated document corpus on each target SDPS.

```
BIRD gold SQL
      │
      ▼
[Step A] postgres_explain_pipeline.py / explain_queries.py (SemPipelineBuilder/)
         Runs EXPLAIN (FORMAT JSON) on every gold query against PostgreSQL
         (or SQLite EXPLAIN QUERY PLAN) and parses physical operator trees
         → SemPipelineBuilder/pipelines/explain_trees*.json
      │
      ▼
[Step B] explain_parser.py
         Translates physical operators (SeqScan, Hash Join, Aggregate, Sort…)
         into high-level semantic operations and identifies subquery pipelines.
      │
      ▼
[Step C] build_pipelines.py (SemPipelineExecutor/)
         Merges BIRD questions with explain-derived pipelines and calls OpenAI
         to generate predicate-focused natural-language instructions
         (semantic_nl) for each step → sem-pipelines/pipelines.json
      │
      ▼
[Step D] run_evaluation.py
         Runs lotus_pipeline.py, docetl_pipeline.py, palimpzest_pipeline.py
         in sequence with the same parameters, feeding each the generated
         document corpus from DocSets/ and the compiled pipelines.json.
         Results (per-question metrics + summary) are written to:
         Results/{system}/{null_mode}/{noise_ratio}/{N}_documents/
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for PostgreSQL explain plan collection)
- An OpenAI API key (required for all cloud paths, including **gpt-4o-mini** fallback when no local server is configured)
- Optionally, an OpenAI-compatible **local** HTTP API for parsing / variation work (see section 1b)

#### Python environment and dependencies

Use a **separate virtual environment** for Rosetta so versions of heavy ML / SDPS libraries do not fight with other projects (and so you can spin up **different envs per backend** if LOTUS, DocETL, and Palimpzest each need incompatible stacks).

From the repository root:

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers **common** needs: DocGen (OpenAI, NLTK, dotenv), SQLite explain tooling, shared helpers, PostgreSQL explain collection (`psycopg2-binary`), and pandas/PyYAML used by the executor utilities and DocETL-oriented scripts.

**Do not** install every SDPS backend in one go unless you know they resolve cleanly. After the base install, **manually** add only what you will run, following each project’s docs for compatible versions, for example:

```bash
# LOTUS-backed evaluation (typical extras; check lotus-ai docs for your platform)
pip install lotus-ai faiss-cpu sentence-transformers

# DocETL-backed evaluation
pip install docetl

# Palimpzest-backed evaluation
pip install palimpzest
```

If two backends conflict, use **two separate venvs** (e.g. `.venv-lotus` and `.venv-docetl`) and activate the one matching the pipeline you are testing.

---

### 1. Configure your API key

Create (or edit) `Config/.env` at the repository root:

```
Config/
└── .env   ← create this file
```

Add your OpenAI key:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

Every script in `DocGen/` and `SemPipelineExecutor/` loads this file automatically via `load_rosetta_env()` / `load_repo_dotenv()`. No other configuration is needed for the key.

---

### 1b. Local OpenAI-compatible LLM vs OpenAI fallback (optional)

Several DocGen stages use a shared **parsing / variation** LLM: sentence-variation generation, template-field extraction, narrative analysis, and similar paths.

- **If `ROSETTA_LOCAL_LLM_BASE_URL` is not set** (or empty), those calls go to the **OpenAI API** using **`gpt-4o-mini`** and your `OPENAI_API_KEY`. Override the model with `ROSETTA_CLOUD_PARSING_MODEL` if needed.
- **If `ROSETTA_LOCAL_LLM_BASE_URL` is set**, those calls use your **OpenAI-compatible** server (`/v1/chat/completions` wire format) with the model and key below.

| Variable | Purpose |
|----------|---------|
| `ROSETTA_LOCAL_LLM_BASE_URL` | When set, use this server for parsing/variation LLM calls. If you omit the trailing `/v1`, it is appended (e.g. `http://127.0.0.1:11434` → `http://127.0.0.1:11434/v1`). |
| `ROSETTA_LOCAL_LLM_API_KEY` | Sent with requests to the local server; many stacks ignore it. Default: `EMPTY`. |
| `ROSETTA_LOCAL_LLM_MODEL` | Model id your server expects (e.g. Ollama: `llama3`). Default when URL is set: `openai/gpt-oss-120b`. |
| `ROSETTA_CLOUD_PARSING_MODEL` | OpenAI model when **no** local URL is set. Default: `gpt-4o-mini`. |

**Examples** (`Config/.env` or your shell):

```env
OPENAI_API_KEY=sk-...

# --- Option A: no local server (omit ROSETTA_LOCAL_LLM_BASE_URL) ---
# Parsing/variation steps use OpenAI gpt-4o-mini.
# ROSETTA_CLOUD_PARSING_MODEL=gpt-4o-mini

# --- Option B: Ollama (OpenAI compatibility enabled) ---
ROSETTA_LOCAL_LLM_BASE_URL=http://127.0.0.1:11434
ROSETTA_LOCAL_LLM_MODEL=llama3

# --- Option C: LM Studio ---
ROSETTA_LOCAL_LLM_BASE_URL=http://127.0.0.1:1234
ROSETTA_LOCAL_LLM_MODEL=your-loaded-model-name
ROSETTA_LOCAL_LLM_API_KEY=lm-studio
```

Other cloud steps (column descriptors, primary sentence templates with GPT-4o, etc.) still use `OPENAI_API_KEY` as before.

---

### 2. Install the BIRD MINIDEV dataset

Rosetta is designed to work with the [BIRD MINIDEV](https://bird-bench.github.io/) benchmark dataset. Download it and place it under `GroundTruth/`:

```
GroundTruth/
└── MINIDEV/
    ├── dev_databases/
    │   ├── california_schools/
    │   │   ├── california_schools.sqlite
    │   │   └── database_description/         ← optional CSV column descriptions
    │   ├── card_games/
    │   │   └── ...
    │   └── ...                               ← one folder per database
    ├── dev_tables.json
    ├── mini_dev_sqlite_gold.sql
    ├── mini_dev_postgresql_gold.sql
    ├── mini_dev_sqlite.json
    └── mini_dev_postgresql.json
```

> **Using a custom dataset?** Create a new folder under `GroundTruth/` (e.g. `GroundTruth/MYDATA/dev_databases/...`) with the same layout. When you run the orchestrator it will prompt you for the folder name.

For BIRD MINIDEV, download from [https://bird-bench.github.io/](https://bird-bench.github.io/) and extract the `dev_databases`, gold SQL files, and JSON files into the structure above.

#### PostgreSQL dump for explain plans (`BIRD_dev.sql`)

The full BIRD development PostgreSQL dump **`BIRD_dev.sql`** is about **1 GB** and **is not included in this repository** (GitHub enforces a 100 MB file limit). You must download it from the [BIRD benchmark](https://bird-bench.github.io/) release or dataset package that contains the PostgreSQL `dev` SQL dump, then place it here:

```
GroundTruth/
└── MINIDEV_postgresql/
    └── BIRD_dev.sql
```

This file is only needed if you run `SemPipelineBuilder/postgres_explain_pipeline.py` against a local PostgreSQL instance (see the SemPipelineBuilder section). SQLite-only explain collection does not require it.

---

### 3. Generate Documents — `DocGen/`

Run the interactive orchestrator from the `DocGen/` directory:

```bash
cd DocGen
python pipeline/pipeline_orchestrator.py
```

The orchestrator will prompt for:
- **Dataset folder** (default: `MINIDEV`)
- **Tables to process** (select from available tables)
- **Null/binary representation mode** (`implicit` or `explicit`)
- **Data-to-noise ratio** (e.g. `1:0` for pure data, `1:1` for equal noise)
- **Sentence variations per sentence** (default: 15)
- **Documents per table** (default: 10)

Generated documents will be written to `DocSets/documents/{null_mode}/{noise_ratio}/`.

To run individual setup steps manually:

```bash
# Step 1: Export row samples from SQLite
python setup/generate_table_sample_text_files.py

# Step 2: Generate LLM column descriptors (requires OpenAI key)
python setup/generate_column_descriptors.py
```

---

### 4. Extract Semantic Pipelines — `SemPipelineBuilder/`

**SQLite explain trees (no Docker needed):**

```bash
cd SemPipelineBuilder
python explain_queries.py
# Output: SemPipelineBuilder/pipelines/explain_trees.json
```

**PostgreSQL explain trees (requires Docker):**

```bash
cd SemPipelineBuilder

# Start Postgres, load dump, run EXPLAIN on all gold queries, stop Postgres
python postgres_explain_pipeline.py --start-docker --stop-docker

# Or migrate from SQLite instead of using a dump
python postgres_explain_pipeline.py --sqlite-source --start-docker --stop-docker

# Output: SemPipelineBuilder/pipelines/explain_trees_postgres.json
```

---

### 5. Build the Execution Pipeline Cache — `SemPipelineExecutor/`

```bash
cd SemPipelineExecutor
python build_pipelines.py
# Output: sem-pipelines/pipelines.json  (cached; subsequent runs reuse it)

# Force a full rebuild
python build_pipelines.py --force-rebuild

# Dry-run (preview without calling OpenAI)
python build_pipelines.py --dry-run --limit 5
```

---

### 6. Run Evaluations — `SemPipelineExecutor/`

**Run all three systems with one command:**

```bash
cd SemPipelineExecutor
python run_evaluation.py \
  --null-param null_binary_implicit \
  --noise-param 1_data_0_noise \
  --num_documents 10
```

**Run a single system:**

```bash
python lotus_pipeline.py \
  --semantic \
  --null-param null_binary_explicit \
  --noise-param 1_data_1_noise \
  --num_documents 100
```

**Filter to specific semantic operation types:**

```bash
python run_evaluation.py \
  --null-param null_binary_implicit \
  --noise-param 1_data_0_noise \
  --ops filter join \
  --limit 50
```

**Key CLI parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--null-param` | Null/binary representation mode folder | `null_binary_explicit` |
| `--noise-param` | Data-to-noise ratio folder | `1_data_1_noise` |
| `--num_documents` | Documents per table (must match corpus size) | `10`, `100` |
| `--ops` | Only run entries with these op types | `filter join agg` |
| `--limit` | Max questions to evaluate (0 = all) | `50` |
| `--systems` | Subset of systems to run | `lotus docetl` |

**Available null/binary modes:**

| Folder name | Null encoding | Binary encoding |
|-------------|--------------|-----------------|
| `null_binary_implicit` | Natural language ("not specified") | Natural language ("is a charter school") |
| `null_binary_explicit` | Literal `NULL` | Literal `0` / `1` |
| `null_explicit_binary_implicit` | Literal `NULL` | Natural language |
| `null_implicit_binary_explicit` | Natural language | Literal `0` / `1` |

---

### 7. Results

All evaluation outputs are written to:

```
Results/{system}/{null_mode}/{noise_ratio}/{N}_documents/
```

For example:

```
Results/
└── lotus/
│   └── null_binary_implicit/
│       └── 1_data_0_noise/
│           └── 10_documents/
│               ├── lotus_metrics.csv         ← per-question accuracy, time, tokens, cost
│               └── lotus_metrics_summary.csv ← aggregate accuracy, total cost
└── docetl/
│   └── ...
└── palimpzest/
    └── ...
```

---

## Architecture at a Glance

```
BIRD MINIDEV (SQLite + Gold SQL + NL Questions)
               │
   ┌───────────┴──────────────────────────────────┐
   │                                              │
   ▼                                              ▼
DocGen/                                   SemPipelineBuilder/
  [1] Row sample extraction                 [A] Explain plan collection
  [2] LLM column descriptor generation      [B] Physical → semantic plan
  [3] LLM sentence template generation          translation
  [4] Narrative wrapping (noise injection)
  [5] Local LLM variation bank generation
  [6] Row-by-row document assembly
   │                                              │
   ▼                                              ▼
DocSets/documents/                        SemPipelineExecutor/
  {null_mode}/{noise_ratio}/                [C] build_pipelines.py
    Text/*.txt                                    (NL instruction generation)
    Multi Column/*.csv                      [D] run_evaluation.py →
    Single Column/*.csv                          lotus / docetl / palimpzest
                                                  │
                                                  ▼
                                           Results/{system}/{...}/
                                             metrics.csv
                                             metrics_summary.csv
```

---

## Key Design Properties

| Property | How Rosetta achieves it |
|----------|------------------------|
| **Optimality** | Ground-truth answers come directly from executing the gold SQL against the original SQLite databases — no manual labeling. |
| **Controllability** | Null encoding, binary encoding, data-to-noise ratio, and corpus size are all independent, parameterized dimensions. |
| **Scalability** | Templates and variation banks are generated once and reused; document corpora can be scaled by adjusting `documents_per_table`; the pipeline cache (`pipelines.json`) avoids redundant LLM calls. |
| **Reproducibility** | Every generated artifact (templates, variations, documents, pipeline cache) is deterministically stored on disk and versioned by parameter path. |
