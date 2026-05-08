import duckdb
import os

# --- PATH CONFIGURATION ---
# Use the exact path you provided
base_path = "/Users/divyanshjain/.cache/huggingface/hub/datasets--Zihan1004--FNSPID/snapshots/bf9189c41527198897d1af3e17b1a0095279fc45"

# FNSPID usually stores the main data in the 'Stock_news' subfolder
# We use a wildcard (*) to capture either 'nasdaq_external_data.csv' or 'All_external.csv'
input_csv = os.path.join(base_path, "Stock_news/*.csv")
output_file = "unique_stock_symbols.csv"

print(f"Scanning files in: {input_csv} ...")

# --- DUCKDB QUERY ---
# We use 'DISTINCT' to get unique values and 'COPY' to save directly to a file
con = duckdb.connect()

try:
    con.execute(f"""
        COPY (
            SELECT DISTINCT Stock_symbol 
            FROM read_csv_auto('{input_csv}', ignore_errors=True)
            WHERE Stock_symbol IS NOT NULL
            ORDER BY Stock_symbol
        ) TO '{output_file}' (HEADER, DELIMITER ',');
    """)
    
    print(f"SUCCESS! Unique symbols saved to: {os.path.abspath(output_file)}")
    
    # Quick count check
    count = con.execute(f"SELECT count(*) FROM read_csv_auto('{output_file}')").fetchone()[0]
    print(f"Total unique symbols found: {count}")

except Exception as e:
    print(f"ERROR: {e}")