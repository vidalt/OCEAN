"""Regression coverage for discrete threshold decoding."""

import pandas as pd
import pytest

from ocean.feature import Feature
from ocean.maxsat import BaseModel, Explanation, FeatureVar
from ocean.typing import Key


class _HardVotingModel(BaseModel):
    _hard_voting = True


@pytest.mark.parametrize(
    ("thresholds", "expected"),
    [((-2.0, -1.0), 0.0), ((1.0, 2.0), 10.0), ((11.0, 12.0), 10.0)],
)
def test_empty_discrete_interval_clamps_insertion_index(
    monkeypatch: pytest.MonkeyPatch,
    thresholds: tuple[float, float],
    expected: float,
) -> None:
    feature = FeatureVar(
        Feature(
            Feature.Type.DISCRETE, levels=[0.0, 10.0], thresholds=thresholds
        ),
        name="feature",
    )
    feature.build(_HardVotingModel())
    explanation = Explanation(
        {"feature": feature}, columns=pd.Index(["feature"])
    )

    # Exercise the decoder's defensive fallback for an empty middle interval.
    # The normal solver encoding excludes such intervals from feasible models.
    def threshold_states(_self: Explanation, _name: Key) -> list[bool]:
        return [False, True]

    monkeypatch.setattr(Explanation, "_get_threshold_states", threshold_states)

    assert explanation.format_threshold_discrete_value(0, "feature") == expected
