import numpy as np
import pandas as pd


def int64_to_uint32_safe(series: pd.Series) -> pd.Series:
    # Clamp to valid range
    result = series.clip(lower=0, upper=np.iinfo(np.uint32).max)
    return result.astype(np.uint32)


def finalize_from_secagg(aggregated_series: pd.Series) -> pd.Series:
    return int64_to_uint32_safe(aggregated_series)
