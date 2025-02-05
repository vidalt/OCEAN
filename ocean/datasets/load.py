from dataclasses import dataclass

import pandas as pd

from ..feature import Feature, FeatureMapper

Loaded = tuple[FeatureMapper, tuple[pd.DataFrame, "pd.Series[int]"]]


@dataclass
class Loader:
    name: str
    target: str
    features: list[Feature]
    names: list[str]
    columns: "pd.Index[str] | pd.MultiIndex"

    URL: str = "https://www.github.com/eminyous/ocean-datasets/blob/main"
    path: str = ""

    def __post_init__(self) -> None:
        self.path = f"{self.name}/{self.name}.csv"

    def load(self) -> Loaded:
        mapper = FeatureMapper(self.names, self.features, self.columns)
        data = self.read(self.path)
        X = data.drop(columns=[self.target])
        y = data[self.target].astype(int)
        return mapper, (X, y)

    def read(self, path: str) -> pd.DataFrame:
        url = f"{self.URL}/{path}?raw=true"
        return pd.read_csv(url)
