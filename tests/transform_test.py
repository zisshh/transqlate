import importlib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
transform = importlib.import_module("transqlate.sql_transform").transform


def test_sqlite_passthrough():
    sql = "SELECT 1;"
    assert transform("sqlite", sql) == "SELECT 1;"


def test_postgres_quotes_and_bool():
    sql = "SELECT `name` FROM users WHERE active=0;"
    expected = 'SELECT "name" FROM users WHERE active= FALSE;'
    assert transform("postgres", sql) == expected


def test_mysql_quotes_and_autoinc():
    sql = 'CREATE TABLE t ("id" INTEGER PRIMARY KEY AUTOINCREMENT);'
    expected = 'CREATE TABLE t (`id` INTEGER PRIMARY KEY AUTO_INCREMENT);'
    assert transform("mysql", sql) == expected


def test_mssql_limit():
    sql = 'SELECT "name" FROM users LIMIT 5;'
    expected = 'SELECT TOP 5 [name] FROM users;'
    assert transform("mssql", sql) == expected


def test_oracle_limit_and_now():
    sql = 'SELECT `id` FROM t LIMIT 2;'
    expected = 'SELECT "id" FROM t FETCH FIRST 2 ROWS ONLY;'
    assert transform("oracle", sql) == expected

