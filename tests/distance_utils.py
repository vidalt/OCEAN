from typing import Any, cast

import numpy as np

from ocean.typing import BaseExplanation


def manual_postprocessed_distance(
    explanation: BaseExplanation,
    *,
    norm: int,
) -> float:
    """
    Compute the decoded post-processed distance independently.

    Returns
    -------
    float
        Post-processed :math:`L_p` distance for the given explanation.

    """
    query = np.asarray(explanation.query, dtype=float).ravel()
    counterfactual = explanation.x

    distance = 0.0
    explanation_any = cast("Any", explanation)
    for name, feature in explanation_any.items():
        if feature.is_one_hot_encoded:
            feature_distance = 0.0
            for code in feature.codes:
                idx = explanation_any.idx.get(name, code)
                delta = float(counterfactual[idx]) - float(query[idx])
                feature_distance += abs(delta) ** norm
            distance += feature_distance / 2.0
        else:
            idx = explanation_any.idx.get(name)
            delta = float(counterfactual[idx]) - float(query[idx])
            distance += abs(delta) ** norm

    if norm != 1:
        distance **= 1.0 / norm
    return float(distance)
