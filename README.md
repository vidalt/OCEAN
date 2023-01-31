<img src="ocean_logo.svg" width="400">

# Optimal Counterfactual Explanations in Tree Ensembles

![Tests badge](https://github.com/vidalt/OCEAN/actions/workflows/python-app.yml/badge.svg?branch=main)

This repository provides methods to generate optimal counterfactual explanations in tree ensembles.
It is based on the paper *Optimal Counterfactual Explanations in Tree Ensemble* by Axel Parmentier and Thibaut Vidal in the *Proceedings of the thirty-eighth International Conference on Machine Learning*, 2021, in press. The article is [available here](http://proceedings.mlr.press/v139/parmentier21a/parmentier21a.pdf).

### Installation

This project requires the gurobi solver. Free academic licenses are available. Please consult:

 - https://www.gurobi.com/academia/academic-program-and-licenses/
 - https://www.gurobi.com/downloads/end-user-license-agreement-academic/

Run the following commands from the project root to install the requirements. You may have to install python and venv before.

```shell
    virtualenv -p python3.10 env
    source env/bin/activate
    pip install -r requirements.txt
    python -m pip install -i https://pypi.gurobi.com gurobipy
    pip install -e .
```

The installation can be checked by running the test suite:
```shell
   python -m pytest test\
```
The integration tests require a working Gurobi license. If a license is not available, the tests will pass and print a warning. 

### Getting started
A minimal working example using OCEAN to derive an optimal counterfactual explanation is presented below.

```python
  # Load packages
  import os as os
  from sklearn.ensemble import RandomForestClassifier
  # Load OCEAN modules
  from src.DatasetReader import DatasetReader
  from src.RfClassifierCounterFactual import RfClassifierCounterFactualMilp

  # - Specify the path to your data set -
  #   Add your data set to a "datasets" folder and specify the name of the csv file.
  #   Note that the specific structure of the csv file should be respected :
  #   -  The first row specifies the features names; the name should be label column should be "Class".
  #   -  The second row specifies the features types (B=binary, C=categorical, D=discrete, N=numerical).
  #   -  The third row specifies the features actionability (FREE, INC=increasing, FIXED, and PREDICT for the "Class").
  #   -  The remaining rows form the training data.
  DATASET = "Phishing.csv"
  dirname = os.path.dirname(__file__)
  datasetPath = os.path.join(dirname, "datasets", DATASET)
  # Load and read data from file
  #    The 'DatasetReader' class will read the type and actionability of features,
  #    normalize the features to [0,1], and encode the categorical features to
  #    one-hot encodded binary features.
  reader = DatasetReader(datasetPath)

  # Train a random forest using sklearn
  rf = RandomForestClassifier(max_depth=6, random_state=1, n_estimators=100)
  rf.fit(reader.X_train.values, reader.y_train.values)

  # - Select initial observation for which to compute a counterfactual -
  #   For instance, here, we select the first training sample as the initial observation.
  x0 = [reader.X_train.values[0]]
  y0 = rf.predict(x0)
  targetClass = 1 - y0
  print('Initial observation x0: ', x0)
  print('Current class: ', y0)
  print('Target class: ', targetClass)

  # - Solve OCEAN to find counterfactual -
  #   The feature types and actionability are read from the 'reader' object.
  randomForestMilp = RfClassifierCounterFactualMilp(
      rf, x0, targetClass,
      featuresActionnability=reader.featuresActionnability,
      featuresType=reader.featuresType,
      featuresPossibleValues=reader.featuresPossibleValues,
      verbose=True)
  randomForestMilp.buildModel()
  randomForestMilp.solveModel()
```

## Reproducing the paper results
This project enables to reproduce the numerical experiments used to produce the tables and figures of the paper.
The folder datasets contains the datasets on which the numerical experiments are performed.

The folder `src` contains all the source code. Launching the script `src/runPaperExperiments.py` will build all the numerical experiments of the paper (after a significant amount of computing time).

Once the numerical experiments have been run, the folder `datasets/counterfactuals` contains all the inputs for which counterfactuals are sought. The folder `results` contains `csv` files with the results of the numerical experiments used to build the figures and tables of the paper.

Author: Axel Parmentier

### Launching experiments

Run the following commands from the project root to launch the numerical experiments:
( If you have run the mace experiments, then you must deactivate your venv environment, either by running the `deactivate` command, or by opening a new console.)

```shell
    source env/bin/activate
    python src/runPaperExperiments.py
```

At the end of the experiments, the folder results contains the csv files that have been used to produce the figures and tables of the paper, except the benchmark with mace (see Section at the end of this ReadMe).    

### Results files format

The results files are csv files. The meaning of the different columns is described below.

| Column        |             |
| ------------- |-------------|
| trainingSetFile   | file containing the training data |
| rf_max_depth      | max_depth parameter of the (sklearn) random forest trained, corresponding to the depth of the trees |
| rf_n_estimators   | n_estimators parameter of the (sklearn) random forest trained, corresponding to the number of trees in the forest|
| ilfActivated   | is the isolation forest taken into account when seeking counterfactuals |
| ilf_max_samples   | max_samples parameter of the (sklearn) isolation forest trained (~number of nodes in the trees) |
| ilf_n_estimators   | n_estimators parameter of the (sklearn) isolation forest trained, corresponding to the number of trees in the forest |
| random_state   | random number generator seed for sklearn |
| train_score   | sklearn train_score of the random forest on the training set |
| test_score   | sklearn test_score of the random forest on the test set |
| counterfactualsFile   | file containing the counterfactuals sought |
| counterfactual_index   | index of the counterfactual in counterfactualsFile |
| useCui   | if True, use OAE to solve the model; Otherwise use OCEAN |
| objectiveNorm   | 0, 1, or 2: indicates the norm used in objective, l0, l1, or l2 (OAE re-implementation is restricted to norm 1) |
| randomForestMilp.model.status   | Gurobi's status at the end |
| randomForestMilp.runTime   | Gurobi's runtime |
| randomForestMilp.objValue   | Gurobi's objective value |
| plausible   | Is the result plausible |
| solution   | (all the columns starting from this one): optimal solution (using the rescaled features) |

### Run Benchmark with mace

Build numerical results using [mace](https://github.com/amirhk/mace) by going to folder `src\benchmarks\maceUpdatedForOceanBenchmark` and following the instructions in `maceUpdatedForOceanBenchmark/ReadMe.md`

You can then get back to the root directory, and launch the benchmark with

```shell
    source env/bin/activate
    python src/benchmarks/runBenchmarkWithMace.py
```
## User Interface
Two user interfaces are available. The `static` and `iterative` interfaces can be started using:
```shell
    source env/bin/activate
    cd ui
    python ui\main_static_interface.py
```
and
```shell
    source env/bin/activate
    cd ui
    python main_iterative_interface.py
```
resepctively. The interfaces allow the user to analyze their own dataset, which has to be placed inside the `datasets` folder in the root directory. The `static` interface allows to generate optimal counterfactual explanation with user-specified constraints on the allowed feature changes. The following gif demonstrates the use of the static interface:

![](ui_gif_v1.gif)

The `iterative` interface allows the user to iteratively modify the initial observation for which to derive counterfactual explanations. It shows the different counterfactual explanations generated through the iterations. A tutorial is included in this interface in the main menu: 'About'->'Help'.
