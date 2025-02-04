from pathlib import Path

import pandas as pd

from ocean.feature import Feature, FeatureMapper


def load_compas(
    *,
    file: Path = Path(__file__).parent
    / "data"
    / "COMPAS-ProPublica_processedMACE.csv",
    target: str = "Class",
) -> tuple[FeatureMapper, tuple[pd.DataFrame, "pd.Series[int]"]]:
    names = [
        "AgeGroup",
        "Race",
        "Sex",
        "PriorsCount",
        "ChargeDegree",
    ]
    features = [
        Feature(ftype=Feature.Type.DISCRETE, levels=(1, 4)),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.DISCRETE, levels=(0, 39)),
        Feature(ftype=Feature.Type.BINARY),
    ]
    columns = pd.MultiIndex.from_tuples([
        ("AgeGroup", ""),
        ("Race", ""),
        ("Sex", ""),
        ("PriorsCount", ""),
        ("ChargeDegree", ""),
    ])
    mapper = FeatureMapper(names=names, features=features, columns=columns)
    data = pd.read_csv(file)
    return mapper, (data.drop(columns=[target]), data[target].astype(int))
