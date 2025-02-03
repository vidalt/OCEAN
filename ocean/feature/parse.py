from collections.abc import Hashable
from typing import Any

import pandas as pd

from .feature import Feature
from .mapper import FeatureMapper

N_BINARY: int = 2


def _count_unique(series: "pd.Series[Any]") -> int:
    return series.nunique()


def _remove_na_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(axis=1)


def _remove_constant_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[:, data.apply(_count_unique) > 1]


def _parse(
    data: pd.DataFrame,
    *,
    discretes: tuple[Hashable, ...] = (),
    scale: bool = True,
) -> tuple[FeatureMapper, pd.DataFrame]:
    discrete = set(discretes)
    frames: list[pd.DataFrame | pd.Series[int] | pd.Series[float]] = []
    features: list[Feature] = []
    names: list[Hashable] = []

    for column in data.columns:
        series: pd.Series[Any] = data[column].rename("")
        levels: tuple[float, ...] = ()
        codes: tuple[Hashable, ...] = ()

        if column in discrete:
            series = series.astype(float)
            frames.append(series)
            ftype = Feature.Type.DISCRETE
            levels = tuple(set(series.dropna()))
        elif series.nunique() == N_BINARY:
            frames.append(
                pd.get_dummies(series, drop_first=True)
                .iloc[:, 0]
                .rename("")
                .astype(int)
            )
            ftype = Feature.Type.BINARY
        elif pd.to_numeric(series, errors="coerce").notna().all():
            x = series.astype(float)
            if scale:
                x = ((x - x.min()) / (x.max() - x.min()) - 0.5).astype(float)
            frames.append(x)
            ftype = Feature.Type.CONTINUOUS
            levels = (x.min() - 0.5, x.max() + 0.5)
        else:
            frames.append(pd.get_dummies(series).astype(int))
            ftype = Feature.Type.ONE_HOT_ENCODED
            codes = tuple(set(series))

        names.append(column)
        features.append(Feature(ftype=ftype, levels=levels, codes=codes))

    proc = pd.concat(frames, axis=1, keys=names)
    if proc.columns.nlevels == 1:
        columns = proc.columns
        mapper = FeatureMapper(names=names, features=features, columns=columns)
    else:
        columns = pd.MultiIndex.from_tuples(proc.columns)
        mapper = FeatureMapper(names=names, features=features, columns=columns)

    return mapper, proc


def parse_features(
    data: pd.DataFrame,
    *,
    discretes: tuple[Hashable, ...] = (),
    drop_na: bool = True,
    drop_constant: bool = True,
    scale: bool = True,
) -> tuple[FeatureMapper, pd.DataFrame]:
    """
    Preprocesses a DataFrame by validating, cleaning, and parsing features.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be processed.
    discretes : tuple[Hashable, ...], optional
        A tuple of column names that should be treated as discrete features.
        default is (). If None, no column is treated as discrete.
    drop_na : bool, optional
        Whether to drop columns with NaN values. default is True.
    drop_constant : bool, optional
        Whether to drop columns with constant values. default is True.
    scale : bool, optional
        Whether to scale continuous features between [0, 1].
        default is True.

    Returns
    -------
        FeatureMapper
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

    return _parse(data, discretes=discretes, scale=scale)
