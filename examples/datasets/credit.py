from pathlib import Path

import pandas as pd

from ocean.feature import Feature, FeatureMapper


def load_credit(
    *,
    file: Path = Path(__file__).parent
    / "data"
    / "default_credit_numerical.csv",
    target: str = "DEFAULT_PAYEMENT",
) -> tuple[FeatureMapper, tuple[pd.DataFrame, "pd.Series[int]"]]:
    names = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "Sex",
        "Education",
        "Marriage",
        "Pay_0",
        "Pay_2",
        "Pay_3",
        "Pay_4",
        "Pay_5",
        "Pay_6",
    ]
    features = [
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(10000 - 1, 1000000 + 1)),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(20, 80)),
        Feature(
            ftype=Feature.Type.CONTINUOUS, levels=(-165580 - 1, 964511 + 1)
        ),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(-69777 - 1, 983931 + 1)),
        Feature(
            ftype=Feature.Type.CONTINUOUS, levels=(-157264 - 1, 1664089 + 1)
        ),
        Feature(
            ftype=Feature.Type.CONTINUOUS, levels=(-170000 - 1, 891586 + 1)
        ),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(-81334 - 1, 927171 + 1)),
        Feature(
            ftype=Feature.Type.CONTINUOUS, levels=(-339603 - 1, 961664 + 1)
        ),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 873552 + 1)),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 1684259 + 1)),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 896040 + 1)),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 621000 + 1)),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 426529 + 1)),
        Feature(ftype=Feature.Type.CONTINUOUS, levels=(0 - 1, 528666 + 1)),
        Feature(ftype=Feature.Type.BINARY),
        Feature(
            ftype=Feature.Type.ONE_HOT_ENCODED,
            codes=("Graduate_School", "High_School", "Others", "University"),
        ),
        Feature(
            ftype=Feature.Type.ONE_HOT_ENCODED,
            codes=("Married", "Others", "Single"),
        ),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.BINARY),
        Feature(ftype=Feature.Type.BINARY),
    ]
    columns = pd.MultiIndex.from_tuples([
        ("LIMIT_BAL", ""),
        ("AGE", ""),
        ("BILL_AMT1", ""),
        ("BILL_AMT2", ""),
        ("BILL_AMT3", ""),
        ("BILL_AMT4", ""),
        ("BILL_AMT5", ""),
        ("BILL_AMT6", ""),
        ("PAY_AMT1", ""),
        ("PAY_AMT2", ""),
        ("PAY_AMT3", ""),
        ("PAY_AMT4", ""),
        ("PAY_AMT5", ""),
        ("PAY_AMT6", ""),
        ("Sex", ""),
        ("Education", "Graduate_School"),
        ("Education", "High_School"),
        ("Education", "Others"),
        ("Education", "University"),
        ("Marriage", "Married"),
        ("Marriage", "Others"),
        ("Marriage", "Single"),
        ("Pay_0", ""),
        ("Pay_2", ""),
        ("Pay_3", ""),
        ("Pay_4", ""),
        ("Pay_5", ""),
        ("Pay_6", ""),
    ])
    data = pd.read_csv(file)

    mapper = FeatureMapper(names=names, features=features, columns=columns)

    return mapper, (data.drop(columns=[target]), data[target])
