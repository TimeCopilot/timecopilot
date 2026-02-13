from __future__ import annotations

from typing import Any, overload

import pandas as pd

import narwhals as nw

_UNSUPPORTED_DF_MESSAGE = (
    "Unsupported dataframe type. Install narwhals to enable support for "
    "polars and other dataframe libraries."
)


@overload
def to_pandas(df: pd.DataFrame) -> pd.DataFrame: ...


@overload
def to_pandas(df: Any, *, fallback: pd.DataFrame) -> pd.DataFrame: ...


@overload
def to_pandas(df: Any) -> pd.DataFrame: ...


def to_pandas(
    df: Any,
    *,
    fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    try:
        return nw.from_native(df).to_pandas()
    except TypeError as exc:
        if fallback is not None:
            return fallback
        raise TypeError(_UNSUPPORTED_DF_MESSAGE) from exc
