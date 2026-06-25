import os
import glob
import re
import pandas as pd

_DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")

def read_gold(base_dir, add_partition_date_as=None):
    parts = sorted(glob.glob(os.path.join(base_dir, "*.parquet")))
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        if add_partition_date_as:
            m = _DATE_RE.search(os.path.basename(p))
            if m:
                df[add_partition_date_as] = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
