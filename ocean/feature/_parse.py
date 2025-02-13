from typing import Any

import pandas as pd

from ..abc import Mapper
from ..typing import Key
from ._feature import Feature

N_BINARY: int = 2

type Parsed = tuple[pd.DataFrame, Mapper[Feature]]


def _count_unique(series: "pd.Series[Any]") -> int:
    return series.nunique()


def _remove_na_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(axis=1)


def _remove_constant_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[:, data.apply(_count_unique) > 1]


def _parse(
    data: pd.DataFrame,
    *,
    discretes: tuple[Key, ...] = (),
    encodeds: tuple[Key, ...] = (),
    scale: bool = True,
) -> Parsed:
    discrete = set(discretes)
    encoded = set(encodeds)
    frames: dict[Key, pd.DataFrame | pd.Series[int] | pd.Series[float]] = {}
    mapping: dict[Key, Feature] = {}

    for column in data.columns:
        series: pd.Series[Any] = data[column].rename("")
        levels: tuple[float, ...] = ()
        codes: tuple[Key, ...] = ()
        is_binary = series.nunique() == N_BINARY
        is_numeric = pd.to_numeric(series, errors="coerce").notna().all()

        frame: pd.DataFrame | pd.Series[int] | pd.Series[float] = series

        if column in discrete:
            series = series.astype(float)
            levels = tuple(set(series.dropna()))
            feature = Feature(Feature.Type.DISCRETE, levels=levels)
        elif (column in encoded) or not (is_binary or is_numeric):
            frame = pd.get_dummies(series).astype(int)
            codes = tuple(set(series))
            feature = Feature(Feature.Type.ONE_HOT_ENCODED, codes=codes)
        elif is_binary:
            frame = (
                pd.get_dummies(series, drop_first=True)
                .iloc[:, 0]
                .rename("")
                .astype(int)
            )
            feature = Feature(Feature.Type.BINARY)
        else:
            x = series.astype(float)
            if scale:
                x = ((x - x.min()) / (x.max() - x.min()) - 0.5).astype(float)
            frame = x
            levels = (x.min() - 0.5, x.max() + 0.5)
            feature = Feature(Feature.Type.CONTINUOUS, levels=levels)

        frames[column] = frame
        mapping[column] = feature

    proc = pd.concat(frames, axis=1)

    if proc.columns.nlevels == 1:
        columns = pd.Index(proc.columns)
    else:
        columns = pd.MultiIndex.from_tuples(proc.columns)

    return proc, Mapper(mapping, columns=columns)


def parse_features(
    data: pd.DataFrame,
    *,
    discretes: tuple[Key, ...] = (),
    encoded: tuple[Key, ...] = (),
    drop_na: bool = True,
    drop_constant: bool = True,
    scale: bool = True,
) -> Parsed:
    """
    Preprocesses a DataFrame by validating, cleaning, and parsing features.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be processed.
    discretes : tuple[Key, ...], optional
        A tuple of column names that should be treated as discrete features.
        default is (). If None, no column is treated as discrete.
    encoded : tuple[Key, ...], optional
        A tuple of column names that should be treated as one-hot encoded
        features. default is ().
    drop_na : bool, optional
        Whether to drop columns with NaN values. default is True.
    drop_constant : bool, optional
        Whether to drop columns with constant values. default is True.
    scale : bool, optional
        Whether to scale continuous features between [0, 1].
        default is True.

    Returns
    -------
        Mapper[Feature]
            A mapper that maps the DataFrame columns to the features.
        pd.DataFrame
            The processed DataFrame.

    Raises
    ------
    ValueError
        If a column in `discretes` is not found in the DataFrame.

    """
    missing = [col for col in discretes if col not in data.columns]
    if missing:
        msg = f"Columns not found in the data: {missing}"
        raise ValueError(msg)

    if drop_na:
        data = _remove_na_columns(data)
    if drop_constant:
        data = _remove_constant_columns(data)

    return _parse(data, discretes=discretes, encodeds=encoded, scale=scale)
