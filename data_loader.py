# pulling and loading data

import duckdb
import os
import glob
import zipfile
import tempfile

# Loading raw ticks into DuckDB
data_path = r"MarketData"

def load_raw_ticks_to_duckdb(data_path):
    
    # searching the MarketData folder for anything terminating with "BTCUSDT-trades-*.zip" 
    # and printing out the number of files encountered
    zip_files = glob.glob(os.path.join(data_path, "*BTCUSDT-trades-*.zip"))
    print(f"Found {len(zip_files)} files")

    # connecting to the DuckDB 
    con = duckdb.connect()

    # creating the table the found files are going to be inserted in
    con.execute('''
        create table trades (
            id               bigint,
            price            double,
            qty              double,
            quote_qty        double,
            time             bigint,
            is_buyer_maker   boolean
        )
        ''')

    # sorting the files to be in chronological order. 
    for zip_file in sorted(zip_files):
        print(f"Loading {os.path.basename(zip_file)}..")

        # Opens each file in the folder, reads the data and stores the data
        with zipfile.ZipFile(zip_file, 'r') as z:
            csv_name = z.namelist()[0]

            # creates a temporary directory where all the file are stored together
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extract(csv_name, tmpdir)
                csv_path = os.path.join(tmpdir, csv_name).replace("\\", "/")

                con.execute(f'''
                    insert into trades
                    select * from read_csv(
                    '{csv_path}',
                    header = true,
                    columns = {{
                        'id':                'bigint',
                        'price':             'double',
                        'qty':               'double',
                        'quote_qty':         'double',
                        'time':              'bigint',
                        'is_buyer_maker':    'boolean'
                    }}
                    )
                ''')

        print(f"Done loading {os.path.basename(zip_file)}")

    print("\nAll files loaded into DuckDB")
    return con


if __name__ == "__main__":
    con = load_raw_ticks_to_duckdb(data_path)
