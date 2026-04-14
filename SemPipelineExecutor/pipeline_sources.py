"""
Shared table-name → database mapping and SQL table extraction.

Single source of truth for the BIRD MINIDEV schema used across
build_pipelines, executor scripts, and pipeline_builder tools.
"""

import re

DATABASE_TABLES: dict[str, str] = {
    "frpm": "california_schools",
    "satscores": "california_schools",
    "schools": "california_schools",

    "cards": "card_games",
    "foreign_data": "card_games",
    "legalities": "card_games",
    "sets": "card_games",
    "set_translations": "card_games",
    "ruling": "card_games",

    "customers": "debit_card_specializing",
    "gasstations": "debit_card_specializing",
    "products": "debit_card_specializing",
    "transactions_1k": "debit_card_specializing",
    "yearmonth": "debit_card_specializing",

    "account": "financial",
    "card": "financial",
    "client": "financial",
    "disp": "financial",
    "district": "financial",
    "loan": "financial",
    "order": "financial",
    "trans": "financial",

    "circuits": "formula_1",
    "constructors": "formula_1",
    "drivers": "formula_1",
    "seasons": "formula_1",
    "races": "formula_1",
    "constructorresults": "formula_1",
    "constructorstandings": "formula_1",
    "driverstandings": "formula_1",
    "laptimes": "formula_1",
    "pitstops": "formula_1",
    "qualifying": "formula_1",
    "status": "formula_1",
    "results": "formula_1",

    "player_attributes": "european_football_2",
    "player": "european_football_2",
    "league": "european_football_2",
    "country": "european_football_2",
    "team": "european_football_2",
    "team_attributes": "european_football_2",
    "match": "european_football_2",

    "examination": "thrombosis_prediction",
    "patient": "thrombosis_prediction",
    "laboratory": "thrombosis_prediction",

    "atom": "toxicology",
    "bond": "toxicology",
    "connected": "toxicology",
    "molecule": "toxicology",

    "event": "student_club",
    "major": "student_club",
    "zip_code": "student_club",
    "attendance": "student_club",
    "budget": "student_club",
    "expense": "student_club",
    "income": "student_club",
    "member": "student_club",

    "alignment": "superhero",
    "attribute": "superhero",
    "colour": "superhero",
    "gender": "superhero",
    "publisher": "superhero",
    "race": "superhero",
    "superhero": "superhero",
    "hero_attribute": "superhero",
    "superpower": "superhero",
    "hero_power": "superhero",

    "badges": "codebase_community",
    "comments": "codebase_community",
    "posthistory": "codebase_community",
    "postlinks": "codebase_community",
    "posts": "codebase_community",
    "tags": "codebase_community",
    "users": "codebase_community",
    "votes": "codebase_community",
}

_TABLE_RE = re.compile(
    r"(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN)\s+"
    r"([a-zA-Z0-9_]+)(?:\s+AS\s+\w+)?",
    re.IGNORECASE,
)


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract known table names from SQL, deduplicated in first-seen order."""
    if not sql or not isinstance(sql, str):
        return []
    sql_norm = sql.replace("\n", " ")
    seen: set[str] = set()
    tables: list[str] = []
    for m in _TABLE_RE.findall(sql_norm):
        key = m.lower()
        if key not in seen and key in DATABASE_TABLES:
            seen.add(key)
            tables.append(key)
    return tables
