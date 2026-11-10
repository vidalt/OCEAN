# Optimal Counterfactual Explanations in Tree Ensembles

[![Maintained](https://img.shields.io/badge/Maintained-YES-14b8a6?style=for-the-badge&logo=github)](https://github.com/vidalt/OCEAN/graphs/commit-activity)
[![License](https://img.shields.io/github/license/vidalt/OCEAN?style=for-the-badge&color=0ea5e9&logo=unlicense&logoColor=white)](https://github.com/vidalt/OCEAN/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-YES-06b6d4?style=for-the-badge&logo=readthedocs&logoColor=white)](https://ocean-py.readthedocs.io/en/latest/)
[![Contributors](https://img.shields.io/github/contributors/vidalt/OCEAN?style=for-the-badge&color=38bdf8&logo=github)](https://github.com/vidalt/OCEAN/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/vidalt/OCEAN?style=for-the-badge&color=0284c7&logo=github)](https://github.com/vidalt/OCEAN/stargazers)
[![Watchers](https://img.shields.io/github/watchers/vidalt/OCEAN?style=for-the-badge&color=2563eb&logo=github)](https://github.com/vidalt/OCEAN/watchers)
[![Forks](https://img.shields.io/github/forks/vidalt/OCEAN?style=for-the-badge&color=1d4ed8&logo=github)](https://github.com/vidalt/OCEAN/network/members)
[![PRs](https://img.shields.io/github/issues-pr/vidalt/OCEAN?style=for-the-badge&color=22c55e&logo=github)](https://github.com/vidalt/OCEAN/pulls)



![Logo](https://github.com/eminyous/ocean/blob/main/logo.svg?raw=True)

**ocean** is a full package dedicated to counterfactual explanations for **tree ensembles**.  
The **mixed-integer programming (MIP)** formulation builds on
[Parmentier and Vidal (2021), *Optimal Counterfactual Explanations in Tree Ensembles*](https://proceedings.mlr.press/v139/parmentier21a.html).
The **constraint programming (CP)** formulation is presented in
[Khouna et al. (2026), *Optimal Counterfactual Search in Tree Ensembles: A Study Across Modeling and Solution Paradigms*](https://arxiv.org/abs/2605.06561).
The package also includes a **weighted MaxSAT** backend for exact counterfactual
search on the same parsed tree ensembles.

## Installation

You can install the package with the following command:

```bash
pip install oceanpy
```
*Note : The MIP method requires the gurobi solver access. You can request for a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Once you have installed gurobi, you can install the package with the command above. However, you can also use the CP method without gurobi.*

## From prediction to possibility

[![OCEAN: one query, three solver backends, and a counterfactual explanation](docs/_static/ocean-overview.svg)](https://ocean-py.readthedocs.io/en/latest/visual-examples.html)

Explore the [visual examples](https://ocean-py.readthedocs.io/en/latest/visual-examples.html),
try the [quickstart](https://ocean-py.readthedocs.io/en/latest/quickstart.html),
or [contribute an idea](https://github.com/vidalt/OCEAN/issues).

## Usage

The package provides multiple classes and functions to wrap the tree ensemble models from the `scikit-learn` library. A minimal example is provided below:

```python
from sklearn.ensemble import RandomForestClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
from ocean.datasets import load_adult

# Load the adult dataset
(data, target), mapper = load_adult()

# Select an instance to explain from the dataset
x = data.iloc[0].to_frame().T

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

# Predict the class of the selected instance
y = int(rf.predict(x).item())
x = x.to_numpy().flatten()

# Explain the prediction using the MIP backend
mip_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
mip_explanation = mip_model.explain(
    x, y=1 - y, norm=1, num_workers=1, random_seed=42
)

# Explain the prediction using the CP backend
cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
cp_explanation = cp_model.explain(
    x, y=1 - y, norm=1, num_workers=1, random_seed=42
)

# Explain the prediction using the MaxSAT backend
maxsat_model = MaxSATExplainer(rf, mapper=mapper)
maxsat_explanation = maxsat_model.explain(
    x, y=1 - y, norm=1, num_workers=1, random_seed=42
)

```

Output for Adult row `0`, predicted class `0`, targeting class `1`.
The **Type** column shows the encoding used by the dataset loader; **bold**
values differ from the query. Category labels are decoded for readability using
[the Adult label mapping](examples/adult_categories.json), verified against the
[original UCI Adult data](https://archive.ics.uci.edu/dataset/2/adult).
`NativeCountry` is binary in this dataset: United States versus other countries.

<table>
<thead>
<tr><th>Feature</th><th>Type</th><th>Query</th><th>MIP</th><th>CP</th><th>MaxSAT</th></tr>
</thead>
<tbody>
<tr><th scope="row">Age</th><td>Discrete</td><td>39</td><td>39</td><td>39</td><td>39</td></tr>
<tr><th scope="row">CapitalGain</th><td>Continuous</td><td>2174</td><td>2174</td><td>2174</td><td>2174</td></tr>
<tr><th scope="row">CapitalLoss</th><td>Continuous</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
<tr><th scope="row">EducationNumber</th><td>Continuous</td><td>13</td><td>13</td><td>13</td><td>13</td></tr>
<tr><th scope="row">HoursPerWeek</th><td>Discrete</td><td>40</td><td>40</td><td>40</td><td>40</td></tr>
<tr><th scope="row">MaritalStatus</th><td>Categorical (one-hot)</td><td>Never married</td><td><strong>Married (civilian spouse)</strong></td><td><strong>Married (civilian spouse)</strong></td><td><strong>Married (civilian spouse)</strong></td></tr>
<tr><th scope="row">NativeCountry</th><td>Binary</td><td>United States</td><td>United States</td><td>United States</td><td>United States</td></tr>
<tr><th scope="row">Occupation</th><td>Categorical (one-hot)</td><td>Administrative / clerical</td><td><strong>Professional specialty</strong></td><td>Administrative / clerical</td><td>Administrative / clerical</td></tr>
<tr><th scope="row">Relationship</th><td>Categorical (one-hot)</td><td>Not in family</td><td><strong>Husband</strong></td><td><strong>Husband</strong></td><td><strong>Husband</strong></td></tr>
<tr><th scope="row">Sex</th><td>Binary</td><td>Male</td><td>Male</td><td>Male</td><td>Male</td></tr>
<tr><th scope="row">WorkClass</th><td>Categorical (one-hot)</td><td>State government</td><td>State government</td><td><strong>Self-employed (incorporated)</strong></td><td><strong>Self-employed (incorporated)</strong></td></tr>
<tr><td colspan="6"><hr></td></tr>
<tr><th scope="row" colspan="2">Objective value (L1)</th><td>—</td><td>3.0</td><td>3.0</td><td>3.0</td></tr>
</tbody>
</table>

The objective row reports each backend's `get_objective_value()` for `norm=1`.
Feature values are rounded to six decimal places and objectives to one decimal
place; solvers may select different tied solutions. Each one-hot category change
counts as one unit of distance, so all three unrestricted objectives are `3.0`. Run [examples/readme.py](examples/readme.py) to regenerate both tables
and verify that the explanations reach the target class.

### Actionability (CP and MIP)

The unrestricted explanations above change `MaritalStatus` and `Relationship`.
For the same query, fix those features, `Sex`, and `NativeCountry`, and allow
`Age`, `EducationNumber`, and `HoursPerWeek` only to increase or stay unchanged:

```python
from ocean.typing import Actionability

rules: Actionability = {
    "Age": "increase",
    "EducationNumber": "increase",
    "HoursPerWeek": "increase",
    "Sex": "fixed",
    "NativeCountry": "fixed",
    "MaritalStatus": "fixed",
    "Relationship": "fixed",
}

mip_explanation = mip_model.explain(
    x, y=1 - y, norm=1, actionability=rules, num_workers=1, random_seed=42
)
cp_explanation = cp_model.explain(
    x, y=1 - y, norm=1, actionability=rules, num_workers=1, random_seed=42
)
```

Both explanations now keep the fixed features unchanged and increase
`HoursPerWeek` from `40` to `44`. They also change the unrestricted features
`CapitalGain`, `CapitalLoss`, and `WorkClass`, at a higher objective value.
MaxSAT does not support actionability, so its entries are marked unavailable.

<table>
<thead>
<tr><th>Feature</th><th>Type</th><th>Query</th><th>MIP</th><th>CP</th><th>MaxSAT (unsupported)</th></tr>
</thead>
<tbody>
<tr><th scope="row">Age</th><td>Discrete</td><td>39</td><td>39</td><td>39</td><td>—</td></tr>
<tr><th scope="row">CapitalGain</th><td>Continuous</td><td>2174</td><td><strong>5119.000488</strong></td><td><strong>5119.000488</strong></td><td>—</td></tr>
<tr><th scope="row">CapitalLoss</th><td>Continuous</td><td>0</td><td><strong>1820.500122</strong></td><td><strong>1820.500122</strong></td><td>—</td></tr>
<tr><th scope="row">EducationNumber</th><td>Continuous</td><td>13</td><td>13</td><td>13</td><td>—</td></tr>
<tr><th scope="row">HoursPerWeek</th><td>Discrete</td><td>40</td><td><strong>44</strong></td><td><strong>44</strong></td><td>—</td></tr>
<tr><th scope="row">MaritalStatus</th><td>Categorical (one-hot)</td><td>Never married</td><td>Never married</td><td>Never married</td><td>—</td></tr>
<tr><th scope="row">NativeCountry</th><td>Binary</td><td>United States</td><td>United States</td><td>United States</td><td>—</td></tr>
<tr><th scope="row">Occupation</th><td>Categorical (one-hot)</td><td>Administrative / clerical</td><td>Administrative / clerical</td><td>Administrative / clerical</td><td>—</td></tr>
<tr><th scope="row">Relationship</th><td>Categorical (one-hot)</td><td>Not in family</td><td>Not in family</td><td>Not in family</td><td>—</td></tr>
<tr><th scope="row">Sex</th><td>Binary</td><td>Male</td><td>Male</td><td>Male</td><td>—</td></tr>
<tr><th scope="row">WorkClass</th><td>Categorical (one-hot)</td><td>State government</td><td><strong>Self-employed (incorporated)</strong></td><td><strong>Self-employed (incorporated)</strong></td><td>—</td></tr>
<tr><td colspan="6"><hr></td></tr>
<tr><th scope="row" colspan="2">Objective value (L1)</th><td>—</td><td>4770.5</td><td>4770.5</td><td>—</td></tr>
</tbody>
</table>

`fixed` preserves the input value; `increase` and `decrease` allow equality.
Omitted features are unrestricted. One-hot groups support `fixed` using the
original group name. Rules apply to decoded values in the processed feature
space and are removed by the default query cleanup, including after an
infeasible solve. See the [explainer guide](docs/explainer-guide.rst) for details.

See the [examples folder](https://github.com/vidalt/OCEAN/tree/main/examples) or the [documentation](https://ocean-py.readthedocs.io/en/latest/) for more usage examples.


## Feature Preview & Roadmap

| Area                            | Status     | Notes / References                         |
| ------------------------------- | ---------- | ------------------------------------------ |
| **MIP formulation**             | ✅ Done     | Based on [Parmentier & Vidal (2021)](https://proceedings.mlr.press/v139/parmentier21a.html).   |
| **Constraint Programming (CP)** | ✅ Done     | Based on [Khouna et al. (2026)](https://arxiv.org/abs/2605.06561).                |
| **MaxSAT formulation**          | ✅ Done     | Based on [Raevskaya & Lehtonen (2025)](https://doi.org/10.3233/FAIA250895).      |
| **Heuristics**                  | ⏳ Upcoming | Fast approximate methods.                  |
| **Other methods**               | ⏳ Upcoming | Additional formulations under exploration. |
| **AdaBoost support**            | ✅ Ready    | Fully supported in ocean.                  |
| **Random Forest support**       | ✅ Ready    | Fully supported in ocean.                  |
| **XGBoost support**             | ✅ Ready    | Fully supported in ocean.                  |

> Legend: ✅ available · ⏳ upcoming

## Citation

If you use OCEAN in your research, please cite the two main papers:

```bibtex
@misc{khouna2026optimalcounterfactualsearch,
  title         = {Optimal Counterfactual Search in Tree Ensembles: A Study Across Modeling and Solution Paradigms},
  author        = {Khouna, Awa and Emine, Youssouf and Ferry, Julien and Vidal, Thibaut},
  year          = {2026},
  eprint        = {2605.06561},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.06561},
  url           = {https://arxiv.org/abs/2605.06561}
}

@inproceedings{pmlr-v139-parmentier21a,
  title     = {Optimal Counterfactual Explanations in Tree Ensembles},
  author    = {Parmentier, Axel and Vidal, Thibaut},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages     = {8422--8431},
  year      = {2021},
  editor    = {Meila, Marina and Zhang, Tong},
  volume    = {139},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v139/parmentier21a.html}
}
```

## References

- Awa Khouna, Youssouf Emine, Julien Ferry, and Thibaut Vidal. 2026. *Optimal Counterfactual Search in Tree Ensembles: A Study Across Modeling and Solution Paradigms*. arXiv:2605.06561. [arXiv](https://arxiv.org/abs/2605.06561).

- Axel Parmentier and Thibaut Vidal. 2021. *Optimal Counterfactual Explanations in Tree Ensembles*. Proceedings of the 38th International Conference on Machine Learning, PMLR 139:8422–8431. [PMLR](https://proceedings.mlr.press/v139/parmentier21a.html).
- Raevskaya, Alesya & Lehtonen, Tuomo. (2025). Optimal Counterfactual Explanations for Random Forests with MaxSAT. 10.3233/FAIA250895. [Available here](https://aaltodoc.aalto.fi/server/api/core/bitstreams/36760903-9b05-491d-b744-ea4309bdf538/content).
