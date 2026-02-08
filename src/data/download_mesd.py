import os
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("somosnlp-hackathon-2022/MESD")
RAW_DATA_DIR = "data/raw/mesd"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

for split in dataset.keys():
    parquet_path = os.path.join(RAW_DATA_DIR, f"{split}.parquet")
    
    # Crear Parquet incremental
    first = True
    for i, item in enumerate(dataset[split]):
        df = pd.DataFrame([{
            "audio_array": item["audio_array"],
            "emotion": item["emotion"]
        }])
        # Si es el primer registro, escribir parquet con cabecera
        if first:
            df.to_parquet(parquet_path, index=False)
            first = False
        else:
            # Append: pandas no soporta append directo a Parquet, usamos pyarrow
            from pyarrow import Table
            import pyarrow.parquet as pq
            table = Table.from_pandas(df)
            with pq.ParquetWriter(parquet_path, table.schema, use_dictionary=True, compression="snappy") as writer:
                writer.write_table(table)

    print(f"Split '{split}' guardado en {parquet_path}")
