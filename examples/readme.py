"""Generate the README comparison tables from actual solver outputs."""

import json
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.abc import Mapper
from ocean.datasets import load_adult
from ocean.feature import Feature
from ocean.typing import Actionability, Array1D, Key

type Values = dict[Key, Key | float]
type Result = tuple[Values, float]


# Display labels only: verified against UCI Adult after dropping missing rows.
# The solver still receives the original encoded values from load_adult().
ADULT_CATEGORIES = cast(
    "dict[str, dict[str, str]]",
    json.loads(Path(__file__).with_name("adult_categories.json").read_text()),
)
FEATURE_TYPES = {
    Feature.Type.CONTINUOUS: "Continuous",
    Feature.Type.DISCRETE: "Discrete",
    Feature.Type.BINARY: "Binary",
    Feature.Type.ONE_HOT_ENCODED: "Categorical (one-hot)",
}


def decode(values: Array1D, mapper: Mapper[Feature]) -> Values:
    """
    Decode processed columns into one row per original feature.

    Returns
    -------
    Values
        Numeric values and category codes keyed by original feature name.

    """
    decoded: Values = {}
    for name, feature in mapper.items():
        if feature.is_one_hot_encoded:
            decoded[name] = next(
                code
                for code in feature.codes
                if values[mapper.idx.get(name, code)] > 0.5
            )
        else:
            decoded[name] = float(values[mapper.idx.get(name)])
    return decoded


def format_value(value: Key | float) -> str:
    if isinstance(value, str):
        return escape(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def display_value(name: Key, value: Key | float) -> str:
    labels = ADULT_CATEGORIES.get(str(name))
    if labels is not None:
        key = value if isinstance(value, str) else f"{value:g}"
        return escape(labels[key])
    return format_value(value)


def print_table(
    query: Values, results: Mapping[str, Result | None], mapper: Mapper[Feature]
) -> None:
    """Print an HTML table with bold changes and an objective separator."""
    print("<table>\n<thead>")
    headers = ["Feature", "Type", "Query", *results]
    print(
        "<tr>"
        + "".join(f"<th>{escape(name)}</th>" for name in headers)
        + "</tr>"
    )
    print("</thead>\n<tbody>")
    for name, value in query.items():
        cells = [
            f'<th scope="row">{escape(str(name))}</th>',
            f"<td>{FEATURE_TYPES[mapper[name].ftype]}</td>",
            f"<td>{display_value(name, value)}</td>",
        ]
        for result in results.values():
            if result is None:
                cells.append("<td>—</td>")
                continue
            candidate = result[0][name]
            rendered = display_value(name, candidate)
            if candidate != value:
                rendered = f"<strong>{rendered}</strong>"
            cells.append(f"<td>{rendered}</td>")
        print("<tr>" + "".join(cells) + "</tr>")
    print(f'<tr><td colspan="{len(headers)}"><hr></td></tr>')
    cells = [
        '<th scope="row" colspan="2">Objective value (L1)</th>',
        "<td>—</td>",
    ]
    for result in results.values():
        value_text = "—" if result is None else f"{result[1]:.1f}"
        cells.append(f"<td>{value_text}</td>")
    print("<tr>" + "".join(cells) + "</tr>")
    print("</tbody>\n</table>")


def main() -> None:
    (data, target), mapper = load_adult()
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(data, target)
    x = data.iloc[0].to_numpy(dtype=float).flatten()
    y = int(np.asarray(rf.predict(data.iloc[[0]]), dtype=np.int64).item())
    query = decode(x, mapper)

    mip_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
    cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
    results: dict[str, Result | None] = {}
    for label, model in (
        ("MIP", mip_model),
        ("CP", cp_model),
        ("MaxSAT", MaxSATExplainer(rf, mapper=mapper)),
    ):
        explanation = model.explain(
            x, y=1 - y, norm=1, num_workers=1, random_seed=42
        )
        assert explanation is not None
        values = explanation.to_numpy().copy()
        assert (
            int(
                np.asarray(
                    rf.predict(pd.DataFrame([values], columns=data.columns)),
                    dtype=np.int64,
                ).item()
            )
            == 1 - y
        )
        results[label] = (decode(values, mapper), model.get_objective_value())
    print(f"Query row: 0; predicted class: {y}; target class: {1 - y}")
    print("Unrestricted explanations:")
    print_table(query, results, mapper)

    rules: Actionability = {
        "Age": "increase",
        "EducationNumber": "increase",
        "HoursPerWeek": "increase",
        "Sex": "fixed",
        "NativeCountry": "fixed",
        "MaritalStatus": "fixed",
        "Relationship": "fixed",
    }
    results.clear()
    for label, actionable_model in (("MIP", mip_model), ("CP", cp_model)):
        explanation = actionable_model.explain(
            x,
            y=1 - y,
            norm=1,
            actionability=rules,
            num_workers=1,
            random_seed=42,
        )
        assert explanation is not None
        values = explanation.to_numpy().copy()
        assert (
            int(
                np.asarray(
                    rf.predict(pd.DataFrame([values], columns=data.columns)),
                    dtype=np.int64,
                ).item()
            )
            == 1 - y
        )
        decoded = decode(values, mapper)
        for name, rule in rules.items():
            if rule == "fixed":
                assert decoded[name] == query[name]
            else:
                position = mapper.idx.get(name)
                assert values[position] >= x[position]
        results[label] = (decoded, actionable_model.get_objective_value())
    results["MaxSAT (unsupported)"] = None
    print("Actionable explanations:")
    print_table(query, results, mapper)


if __name__ == "__main__":
    main()
