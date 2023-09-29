import sqlite3
import csv

# Function to convert CSV to SQLite database
def csv_to_sqlite(csv_file, db_file, table_name):
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create a table with the same name as the CSV file (or a custom name)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT)")

    # Read the CSV file and insert data into the table
    with open(csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Assuming the first row is the header

        # Create columns based on the header
        for column_name in header:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN '{column_name}' TEXT")

        # Insert data from the CSV file
        for row in csv_reader:
            placeholders = ', '.join(['?'] * len(row))
            cursor.execute(f"INSERT INTO {table_name} VALUES (NULL, {placeholders})", row)

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    csv_file = "project 1\data\Sales Data.csv"  # Replace with your CSV file name
    db_file = "project 1\Sales_Data.db"        # Replace with your desired output database file name
    table_name = "sales"          # Replace with your desired table name

    csv_to_sqlite(csv_file, db_file, table_name)
    print(f"CSV file '{csv_file}' has been converted to SQLite database '{db_file}' in table '{table_name}'")
