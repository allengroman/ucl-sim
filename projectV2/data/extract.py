import sqlite3
import csv
import os

def export_sqlite_to_csv(db_path, output_dir='csv_exports'):
    """
    Export all tables from an SQLite database to individual CSV files.
    
    Args:
        db_path (str): Path to the SQLite database file
        output_dir (str): Directory to save CSV files (default: 'csv_exports')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            print("No tables found in the database.")
            return
        
        print(f"Found {len(tables)} tables. Exporting to CSV...")
        
        for table in tables:
            # Get table data
            cursor.execute(f"SELECT * FROM {table};")
            rows = cursor.fetchall()
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Create CSV file
            csv_file = os.path.join(output_dir, f"{table}.csv")
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)  # Write header
                writer.writerows(rows)     # Write data
            
            print(f"Exported {table} ({len(rows)} rows) to {csv_file}")
        
        print("All tables exported successfully!")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Get database path from user input
    db_path = input("Enter path to SQLite database file: ").strip()
    
    # Validate path
    if not os.path.isfile(db_path):
        print("Error: Database file not found.")
    else:
        export_sqlite_to_csv(db_path)
