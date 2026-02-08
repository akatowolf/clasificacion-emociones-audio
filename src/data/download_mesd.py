import os
import pandas as pd
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

dataset = load_dataset("somosnlp-hackathon-2022/MESD")

RAW_DATA_DIR = "data/raw/mesd"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

for split in dataset.keys():
    parquet_path = os.path.join(RAW_DATA_DIR, f"{split}.parquet")
    
    # Crear ParquetWriter
    writer = None
    
    for i, item in enumerate(dataset[split]):
        df = pd.DataFrame([{
            "audio_array": item["audio_array"],
            "emotion": item["emotion"]
        }])
        table = pa.Table.from_pandas(df)
        
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
        
        writer.write_table(table)
    
    if writer:
        writer.close()
    
    print(f"Split '{split}' guardado en {parquet_path}")
