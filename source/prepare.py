import duckdb
import pathlib

data_path = pathlib.Path(__file__).parents[1] / "data"

query = f"""
    WITH t0 AS (
        SELECT 
            * EXCLUDE(seq),
            STRING_SPLIT(seq, ',') as seq_array
        FROM '{data_path / "train.parquet"}'
    )
    SELECT 
        *,
        seq_array[1] as seq_first,
        seq_array[-1] as seq_last,
        LENGTH(seq_array) as seq_length
    FROM t0
"""

target_path = data_path / "prepared/train.parquet"
target_path.parent.mkdir(parents=True, exist_ok=True)
duckdb.query(query).to_parquet(str(target_path))
