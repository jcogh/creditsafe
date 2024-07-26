import sqlite3
import pandas as pd


def get_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [table[0] for table in cursor.fetchall()]


def print_table_contents(conn, table_name):
    print(f"\nContents of {table_name}:")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(df)
    except pd.io.sql.DatabaseError as e:
        print(f"Error reading table {table_name}: {e}")


def view_database(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            tables = get_tables(conn)
            print("\nTables in the database:")
            for table in tables:
                print(table)

            for table in tables:
                print_table_contents(conn, table)
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    view_database('credit_risk.db')
