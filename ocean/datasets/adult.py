import pandas as pd

from ..feature import Feature
from .load import Loader


class AdultLoader(Loader):
    def __init__(self) -> None:
        super().__init__(
            name="Adult",
            target="Class",
            names=[
                "Sex",
                "Age",
                "NativeCountry",
                "EducationNumber",
                "CapitalGain",
                "CapitalLoss",
                "HoursPerWeek",
                "WorkClass",
                "MaritalStatus",
                "Occupation",
                "Relationship",
            ],
            features=[
                Feature(ftype=Feature.Type.BINARY),
                Feature(ftype=Feature.Type.DISCRETE, levels=(17, 91)),
                Feature(ftype=Feature.Type.BINARY),
                Feature(ftype=Feature.Type.DISCRETE, levels=range(1, 17)),
                Feature(
                    ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 99999 + 1)
                ),
                Feature(
                    ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 4356 + 1)
                ),
                Feature(ftype=Feature.Type.DISCRETE, levels=range(1, 100)),
                Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=range(1, 8)),
                Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=range(1, 8)),
                Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=range(1, 15)),
                Feature(ftype=Feature.Type.ONE_HOT_ENCODED, codes=range(6)),
            ],
            columns=pd.MultiIndex.from_tuples([
                ("Sex", ""),
                ("Age", ""),
                ("NativeCountry", ""),
                ("EducationNumber", ""),
                ("CapitalGain", ""),
                ("CapitalLoss", ""),
                ("HoursPerWeek", ""),
                ("WorkClass", 1),
                ("WorkClass", 2),
                ("WorkClass", 3),
                ("WorkClass", 4),
                ("WorkClass", 5),
                ("WorkClass", 6),
                ("WorkClass", 7),
                ("MaritalStatus", 1),
                ("MaritalStatus", 2),
                ("MaritalStatus", 3),
                ("MaritalStatus", 4),
                ("MaritalStatus", 5),
                ("MaritalStatus", 6),
                ("MaritalStatus", 7),
                ("Occupation", 1),
                ("Occupation", 2),
                ("Occupation", 3),
                ("Occupation", 4),
                ("Occupation", 5),
                ("Occupation", 6),
                ("Occupation", 7),
                ("Occupation", 8),
                ("Occupation", 9),
                ("Occupation", 10),
                ("Occupation", 11),
                ("Occupation", 12),
                ("Occupation", 13),
                ("Occupation", 14),
                ("Relationship", 0),
                ("Relationship", 1),
                ("Relationship", 2),
                ("Relationship", 3),
                ("Relationship", 4),
                ("Relationship", 5),
            ]),
        )
