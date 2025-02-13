from dataclasses import dataclass
from typing import Literal, overload

import pandas as pd

from ..abc import Mapper
from ..feature import Feature, parse_features

Dataset = tuple[pd.DataFrame, "pd.Series[int]"]
Loaded = tuple[Dataset, Mapper[Feature]]


@dataclass
class Loader:
    URL: str = "https://www.github.com/eminyous/ocean-datasets/blob/main"

    @overload
    def load(
        self,
        name: str,
        *,
        scale: bool = False,
        return_mapper: Literal[True] = True,
    ) -> Loaded: ...

    @overload
    def load(
        self,
        name: str,
        *,
        scale: bool = False,
        return_mapper: Literal[False],
    ) -> Dataset: ...

    def load(
        self,
        name: str,
        *,
        scale: bool = False,
        return_mapper: bool = True,
    ) -> Dataset | Loaded:
        path = f"{name}/{name}.csv"
        data = self.read(path)
        types: pd.Index[str] = data.columns.get_level_values(1)
        columns: pd.Index[str] = data.columns.get_level_values(0)
        data.columns = columns
        targets = data.columns[types == "T"].to_list()
        y = data[targets].iloc[:, 0].astype(int)
        discretes = tuple(data.columns[types == "D"].to_list())
        encoded = tuple(data.columns[types == "E"].to_list())
        data = data.drop(columns=targets)
        data, mapper = parse_features(
            data,
            discretes=discretes,
            encoded=encoded,
            scale=scale,
        )

        if return_mapper:
            return (data, y), mapper
        return data, y

    def read(self, path: str) -> pd.DataFrame:
        url = f"{self.URL}/{path}?raw=true"
        return pd.read_csv(url, header=[0, 1])
