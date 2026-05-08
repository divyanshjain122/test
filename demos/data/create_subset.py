import duckdb
import os

# --- CONFIGURATION ---
base_path = "/Users/divyanshjain/.cache/huggingface/hub/datasets--Zihan1004--FNSPID/snapshots/bf9189c41527198897d1af3e17b1a0095279fc45"
input_csv = os.path.join(base_path, "Stock_news/*.csv")
output_file = "filtered_stock_data.csv"

# Your specific list of tickers
tickers = [
    'AAPL', 'NVDA', 'AMZN', 'TSLA', 'BRK.B', 'MSFT', 'SHW', 
    'JPM', 'BAC', 'WFC', 'HD', 'V', 'GS', 'PG', 'XOM', 'DIS'
]

# --- DUCKDB FILTERING ---
con = duckdb.connect()

print(f"Filtering 30GB dataset for {len(tickers)} symbols after 2020...")

try:
    # 1. We use read_csv_auto with 'ignore_errors' for safety in large files
    # 2. We cast the string date to a TIMESTAMP to ensure accurate date filtering
    # 3. We use the IN clause for the tickers
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{input_csv}', ignore_errors=True)
            WHERE Stock_symbol IN {tuple(tickers)}
            AND CAST("Date" AS TIMESTAMP) >= '2019-01-01'
        ) TO '{output_file}' (HEADER, DELIMITER ',');
    """)

    # Verify the results
    row_count = con.execute(f"SELECT count(*) FROM read_csv_auto('{output_file}')").fetchone()[0]
    print(f"SUCCESS! Subset created with {row_count} rows.")
    print(f"File saved as: {os.path.abspath(output_file)}")

except Exception as e:
    # If "Date" column name is different in your specific version, 
    # check the headers and update the query.
    print(f"An error occurred: {e}")