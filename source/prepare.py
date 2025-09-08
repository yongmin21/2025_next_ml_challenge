import duckdb
import pathlib

data_path = pathlib.Path(__file__).parents[1] / "data"

query = f"""
    WITH minority AS (
        SELECT * EXCLUDE(seq)
        FROM '{data_path / "train.parquet"}'
        WHERE clicked = 1
    ),

    majority AS (
        SELECT * EXCLUDE(seq)
        FROM '{data_path / "train.parquet"}'
        WHERE clicked = 0
        ORDER BY random()
        LIMIT 3 * (SELECT count(*) FROM minority)
    )

    SELECT *
    FROM minority

    UNION ALL

    SELECT *
    FROM majority
"""

target_path = data_path / "prepared/train_under_sampled.parquet"
target_path.parent.mkdir(parents=True, exist_ok=True)
duckdb.query(query).to_parquet(str(target_path))
