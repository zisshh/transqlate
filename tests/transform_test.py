import importlib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sql_mod = importlib.import_module("transqlate.sql_transform")
transform = sql_mod.transform
qualify_mssql = sql_mod.schema_qualify_mssql
qualify_pg = sql_mod.schema_qualify_postgres
qualify_oracle = sql_mod.schema_qualify_oracle


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


def test_mssql_schema_qualification():
    sql = 'SELECT TOP 5 Product FROM Products ORDER BY Price DESC;'
    mapping = {"Products": "Sales"}
    expected = 'SELECT TOP 5 Product FROM Sales.Products ORDER BY Price DESC;'
    assert qualify_mssql(sql, mapping) == expected


def test_postgres_schema_qualification():
    sql = 'SELECT * FROM items JOIN orders ON items.id=orders.item_id;'
    mapping = {"items": "shop", "orders": "sales"}
    expected = 'SELECT * FROM shop.items JOIN sales.orders ON items.id=orders.item_id;'
    assert qualify_pg(sql, mapping) == expected


def test_oracle_schema_qualification():
    sql = 'SELECT * FROM Products p JOIN Categories c ON p.cat=c.id'
    mapping = {"PRODUCTS": "SALES", "CATEGORIES": "HR"}
    expected = 'SELECT * FROM SALES.Products p JOIN HR.Categories c ON p.cat=c.id'
    assert qualify_oracle(sql, mapping, 'SCOTT') == expected

