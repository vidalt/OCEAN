from dataclasses import dataclass

import pandas as pd

from ..abc import Mapper
from ..feature import Feature, parse_features

Loaded = tuple[Mapper[Feature], tuple[pd.DataFrame, "pd.Series[int]"]]


@dataclass
class Loader:
    name: str
    URL: str = "https://www.github.com/eminyous/ocean-datasets/blob/main"
    path: str = ""

    def __post_init__(self) -> None:
        self.path = f"{self.name}/{self.name}.csv"

    def load(self) -> Loaded:
        data = self.read(self.path)
        types: pd.Index[str] = data.columns.get_level_values(1)
        columns: pd.Index[str] = data.columns.get_level_values(0)
        data.columns = columns
        targets = data.columns[types == "T"].to_list()
        y = data[targets].iloc[:, 0].astype(int)
        discretes = tuple(data.columns[types == "D"].to_list())
        encoded = tuple(data.columns[types == "E"].to_list())
        data = data.drop(columns=targets)
        mapper, data = parse_features(
            data,
            discretes=discretes,
            encoded=encoded,
            scale=False,
        )
        return mapper, (data, y)

    def read(self, path: str) -> pd.DataFrame:
        url = f"{self.URL}/{path}?raw=true"
        return pd.read_csv(url, header=[0, 1])
