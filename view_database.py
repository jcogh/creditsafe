import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('credit_risk.db')

# Get list of tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(table[0])

# Print contents of each table
for table in tables:
    print(f"\nContents of {table[0]}:")
    df = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)
    print(df)

# Close the connection
conn.close()
