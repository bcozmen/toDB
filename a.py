import duckdb

PATH = "/home/baris/database_transformer/dataset/"
con = duckdb.connect(f"{PATH}/healthcare.duckdb", read_only=True)

table_name = "observations"
#get column names
query = f"PRAGMA table_info({table_name})"
df = con.execute(query).fetchdf()
print(df)

#get unique of column type
query = f"SELECT DISTINCT type FROM {table_name}"
df = con.execute(query).fetchdf()
print(df)

# get columns value and units where type is 'text'
query = f"SELECT value, units FROM {table_name} WHERE type='text' LIMIT 10"
df = con.execute(query).fetchdf()
print(df)

#get unique values of column 'value' where type is 'text'
query = f"SELECT category, code, description, units, value FROM {table_name} WHERE type='text' ORDER BY RANDOM() LIMIT 100"
df = con.execute(query).fetchdf()
print(df)

for row in df.itertuples():
    print(f"Category: {row.category}, Code: {row.code}, Description: {row.description}, Units: {row.units}, Value: {row.value}")
    print("--------------------------------------------------")