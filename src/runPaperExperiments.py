from src.RunExperimentsRoutines import runNumericalExperiments
from src.BuildCounterFactualSeekedSet import buildCounterFactualSeekedFile
from src.CounterFactualParameters import BinaryDecisionVariables

# Define data sets, desired outcome, and path to data and cf files
dataDir = './datasets/'
cfDir = './datasets/counterfactuals/'
datasetsWithDesiredOutcome = {
    dataDir+'Adult_processedMACE.csv': 1,
    dataDir+'COMPAS-ProPublica_processedMACE.csv': 1,
    dataDir+'Credit-Card-Default_processedMACE.csv': 1,
    dataDir+'German-Credit.csv': 1,
    dataDir+'Phishing.csv': 1,
    dataDir+'Spambase.csv': 1,
    dataDir+'Students-Performance-MAT.csv': 1,
    dataDir+'OnlineNewsPopularity.csv': 1}
ourDatasetsWithCounterfactualsDict = {
    dataDir+'Adult_processedMACE.csv': cfDir+'OneHot_Adult_processedMACE.csv',
    dataDir+'COMPAS-ProPublica_processedMACE.csv': cfDir+'OneHot_COMPAS-ProPublica_processedMACE.csv',
    dataDir+'German-Credit.csv': cfDir+'OneHot_German-Credit.csv',
    dataDir+'Phishing.csv': cfDir+'OneHot_Phishing.csv',
    dataDir+'Spambase.csv': cfDir+'OneHot_Spambase.csv',
    dataDir+'Students-Performance-MAT.csv': cfDir+'OneHot_Students-Performance-MAT.csv',
    dataDir+'Credit-Card-Default_processedMACE.csv': cfDir+'OneHot_Credit-Card-Default_processedMACE.csv',
    dataDir+'OnlineNewsPopularity.csv': cfDir+'OneHot_OnlineNewsPopularity.csv'}
mediumDatasetsDict = {
    dataDir+'Adult_processedMACE.csv': cfDir+'OneHot_Adult_processedMACE.csv',
    dataDir+'Credit-Card-Default_processedMACE.csv': cfDir+'OneHot_Credit-Card-Default_processedMACE.csv'}

# Build from datasets the set of counterfactuals
# that will be used in the numerical experiments
for dataset in datasetsWithDesiredOutcome:
    nbCounterFactuals = 20
    buildCounterFactualSeekedFile(
        dataset, datasetsWithDesiredOutcome[dataset], nbCounterFactuals, False)

# ----------------- FIGURE 1 -----------------
# Compute the numerical experiments needed to build figure 1
# and store in results/Figure1.csv
runNumericalExperiments(
    ourDatasetsWithCounterfactualsDict,
    rf_max_depthList=[5], rf_n_estimatorsList=[100],
    ilfActivatedList=[False], ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100], random_stateList=[1],
    useCuiList=[False], objectiveNormList=[0, 1, 2],
    binaryDecisionVariablesList=[BinaryDecisionVariables.PathFlow_y],
    randomCostsActivated=False, numericalResultsFileName="results/Figure1.csv")

# ----------------- FIGURE 2 -----------------
objectiveNormList = [1]
# Compute the numerical results necessary to plot the result of OCEAN
# for Figure 2 and store them in results/Figure2_OCEAN.csv"
runNumericalExperiments(
    mediumDatasetsDict, rf_max_depthList=range(3, 9),
    rf_n_estimatorsList=[100], ilfActivatedList=[False],
    ilf_max_samplesList=[32], ilf_n_estimatorsList=[100],
    random_stateList=[1], useCuiList=[False],
    objectiveNormList=objectiveNormList, randomCostsActivated=False,
    numericalResultsFileName="results/Figure2_OCEAN.csv")

# Compute the numerical results necessary to plot the result of OAE
# for Figure 2 and store them in results/Figure2_OAE.csv"
runNumericalExperiments(
    mediumDatasetsDict, rf_max_depthList=range(3, 9),
    rf_n_estimatorsList=[100], ilfActivatedList=[False],
    ilf_max_samplesList=[32], ilf_n_estimatorsList=[100],
    random_stateList=[1], useCuiList=[True],
    objectiveNormList=objectiveNormList, randomCostsActivated=False,
    numericalResultsFileName="results/Figure2_OAE.csv")

# ----------------- FIGURE 3 -----------------
objectiveNormList = [1]
# Compute the numerical results necessary to plot the result of OCEAN
# for Figure 3 and store them in results/Figure3_OCEAN.csv
runNumericalExperiments(
    mediumDatasetsDict, rf_max_depthList=[5],
    rf_n_estimatorsList=[10, 20, 50, 100, 200, 500],
    ilfActivatedList=[False], ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100], random_stateList=[1],
    useCuiList=[False], objectiveNormList=objectiveNormList,
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure3_OCEAN.csv")

# Compute the numerical results necessary to plot the result of OAE
# for Figure 3 and store them in results/Figure3_OAE.csv
#   - Categorical features are handled as suggested by Cui et al.
#       (even if in mediumDatasetsDicts the counterfactuals
#        are provided with './counterfactuals/OneHot_...'
#        we get the information on categorical features
#        from the dataset ./datasets/...')
runNumericalExperiments(
    mediumDatasetsDict, rf_max_depthList=[5],
    rf_n_estimatorsList=[10, 20, 50, 100, 200, 500],
    ilfActivatedList=[False], ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100], random_stateList=[1],
    useCuiList=[True], objectiveNormList=objectiveNormList,
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure3_OAE.csv")

# ----------------- TABLE 4 -----------------
# Compute plausibility results for table 4
# and store them in results/Table4.csv"
runNumericalExperiments(
    ourDatasetsWithCounterfactualsDict,
    rf_max_depthList=[5], rf_n_estimatorsList=[100],
    ilfActivatedList=[True], ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100], random_stateList=[1],
    useCuiList=[False], objectiveNormList=objectiveNormList,
    randomCostsActivated=False,
    numericalResultsFileName="results/Table3.csv")
