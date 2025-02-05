import pandas as pd

from ..feature import Feature
from .load import Loader


class COMPASLoader(Loader):
    def __init__(self) -> None:
        super().__init__(
            name="COMPAS",
            target="Class",
            names=[
                "AgeGroup",
                "Race",
                "Sex",
                "PriorsCount",
                "ChargeDegree",
            ],
            features=[
                Feature(ftype=Feature.Type.DISCRETE, levels=(1, 4)),
                Feature(ftype=Feature.Type.BINARY),
                Feature(ftype=Feature.Type.BINARY),
                Feature(ftype=Feature.Type.DISCRETE, levels=(0, 39)),
                Feature(ftype=Feature.Type.BINARY),
            ],
            columns=pd.MultiIndex.from_tuples([
                ("AgeGroup", ""),
                ("Race", ""),
                ("Sex", ""),
                ("PriorsCount", ""),
                ("ChargeDegree", ""),
            ]),
        )
