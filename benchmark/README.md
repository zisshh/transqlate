# Benchmarking transqlate on SPIDER

This directory contains the benchmarking setup for evaluating transqlate on the SPIDER benchmark for NL2SQL.

## Structure

- `data/`: Contains benchmark data files
  - `test.json`: Test set with NL questions and SQL queries
  - `test_tables.json`: Database schema information
  - `test_database/`: Directory containing SQLite database files for all test databases

- `scripts/`: Benchmarking scripts
  - `run_inference.py`: Run model inference on test data
  - `process_sql.py`: Process and extract SQL from model outputs
  - `evaluation.py`: Run evaluation metrics on predictions

- `results/`: Output files
  - `predictions.json`: Model predictions in JSON format
  - `predictions.sql`: SQL predictions for evaluation
  - `test_gold.sql`: Gold standard SQL queries
  - `eval_results.txt`: Evaluation results

## Quick Start

1. **Run inference**:
   ```bash
   python scripts/run_inference.py
   ```

2. **Run evaluation**:
   ```bash
   python scripts/evaluation.py --gold results/test_gold.sql --pred results/predictions.sql --etype all --db data/test_database --table data/test_tables.json
   ```

### Evaluation Arguments:
- `--gold`: Path to gold SQL file (format: `SQL \t db_id`)
- `--pred`: Path to predicted SQL file
- `--etype`: Evaluation type ("match", "exec", or "all")
- `--db`: Directory containing database subdirectories
- `--table`: Path to table.json file with schema information

## Database Structure

The `test_database/` directory contains subdirectories for each database, with each subdirectory containing the corresponding SQLite database file. The database names correspond to the `db_id` values in the test data.

### Data Content and Format

#### Question, SQL, and Parsed SQL

Each file in `test.json` contains the following fields:
- `question`: the natural language question
- `question_toks`: the natural language question tokens
- `db_id`: the database id to which this question is addressed.
- `query`: the SQL query corresponding to the question. 
- `query_toks`: the SQL query tokens corresponding to the question. 
- `sql`: parsed results of this SQL query using `process_sql.py`. 

```json
 {
        "db_id": "world_1",
        "query": "SELECT avg(LifeExpectancy) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  \"English\" AND T2.IsOfficial  =  \"T\")",
        "query_toks": ["SELECT", "avg", "(", "LifeExpectancy", ")", "FROM", ...],
        "question": "What is average life expectancy in the countries where English is not the official language?",
        "question_toks": ["What", "is", "average", "life", ...],
        "sql": {
            "except": null,
            "from": {
                "conds": [],
                "table_units": [
                    ...
            },
            "groupBy": [],
            "having": [],
            "intersect": null,
            "limit": null,
            "orderBy": [],
            "select": [
                ...
            ],
            "union": null,
            "where": [
                [
                    true,
                    ...
                    {
                        "except": null,
                        "from": {
                            "conds": [
                                [
                                    false,
                                    2,
                                    [
                                    ...
                        },
                        "groupBy": [],
                        "having": [],
                        "intersect": null,
                        "limit": null,
                        "orderBy": [],
                        "select": [
                            false,
                            ...
                        "union": null,
                        "where": [
                            [
                                false,
                                2,
                                [
                                    0,
                                   ...
        }
    },

```

#### Tables

`test_tables.json` contains the following information for each database:
- `db_id`: database id
- `table_names_original`: original table names stored in the database.
- `table_names`: cleaned and normalized table names. We make sure the table names are meaningful. [to be changed]
- `column_names_original`: original column names stored in the database. Each column looks like: `[0, "id"]`. `0` is the index of table names in `table_names`, which is `city` in this case. `"id"` is the column name. 
- `column_names`: cleaned and normalized column names. We make sure the column names are meaningful. [to be changed]
- `column_types`: data type of each column
- `foreign_keys`: foreign keys in the database. `[3, 8]` means column indices in the `column_names`. These two columns are foreign keys of two different tables.
- `primary_keys`: primary keys in the database. Each number is the index of `column_names`.


```json
{
    "column_names": [
      [
        0,
        "id"
      ],
      [
        0,
        "name"
      ],
      [
        0,
        "country code"
      ],
      [
        0,
        "district"
      ],
      .
      .
      .
    ],
    "column_names_original": [
      [
        0,
        "ID"
      ],
      [
        0,
        "Name"
      ],
      [
        0,
        "CountryCode"
      ],
      [
        0,
        "District"
      ],
      .
      .
      .
    ],
    "column_types": [
      "number",
      "text",
      "text",
      "text",
         .
         .
         .
    ],
    "db_id": "world_1",
    "foreign_keys": [
      [
        3,
        8
      ],
      [
        23,
        8
      ]
    ],
    "primary_keys": [
      1,
      8,
      23
    ],
    "table_names": [
      "city",
      "sqlite sequence",
      "country",
      "country language"
    ],
    "table_names_original": [
      "city",
      "sqlite_sequence",
      "country",
      "countrylanguage"
    ]
  }
```


#### Databases

All table contents are contained in corresponding SQLite3 database files.
