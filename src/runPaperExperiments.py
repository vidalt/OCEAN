# Build from datasets the set of counterfactuals that will be used in the numerical experiments

from BuildCounterFactualSeekedSet import *

datasetsWithDesiredOutcome = {
    './datasets/Adult_processedMACE.csv':1,
    './datasets/COMPAS-ProPublica_processedMACE.csv':1,
    './datasets/Credit-Card-Default_processedMACE.csv':1,
    './datasets/German-Credit.csv':1,
    './datasets/Phishing.csv':1,
    './datasets/Spambase.csv':1,
    './datasets/Students-Performance-MAT.csv':1,
    './datasets/OnlineNewsPopularity.csv':1
}

for dataset in datasetsWithDesiredOutcome:
    buildCounterFactualSeekedFile(dataset, datasetsWithDesiredOutcome[dataset], 20, False)

# Compute and store in results/Figure1.csv the numerical experiments needed to build figure 1

ourDatasetsWithCounterfactualsDict = {
    './datasets/Adult_processedMACE.csv':'./counterfactuals/OneHot_Adult_processedMACE.csv',
    './datasets/COMPAS-ProPublica_processedMACE.csv':'./counterfactuals/OneHot_COMPAS-ProPublica_processedMACE.csv',
    './datasets/German-Credit.csv':'./counterfactuals/OneHot_German-Credit.csv',
    './datasets/Phishing.csv':'./counterfactuals/OneHot_Phishing.csv',
    './datasets/Spambase.csv':'./counterfactuals/OneHot_Spambase.csv',
    './datasets/Students-Performance-MAT.csv':'./counterfactuals/OneHot_Students-Performance-MAT.csv',
    './datasets/Credit-Card-Default_processedMACE.csv':'./counterfactuals/OneHot_Credit-Card-Default_processedMACE.csv',
    './datasets/OnlineNewsPopularity.csv':'./counterfactuals/OneHot_OnlineNewsPopularity.csv',
}

from RunExperimentsRoutines import *

runNumericalExperiments(ourDatasetsWithCounterfactualsDict,
    rf_max_depthList=[5],
    rf_n_estimatorsList=[100],
    ilfActivatedList=[False],
    ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100],
    random_stateList=[1],
    useCuiList=[False],
    objectiveNormList=[0,1,2],
    binaryDecisionVariablesList=[BinaryDecisionVariables.PathFlow_y],
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure1.csv"
)

# Compute the numerical results necessary to plot the result of OCEAN for Figure 2 and store them in results/Figure2_OCEAN.csv"

mediumDatasetsDict = {
    './datasets/Adult_processedMACE.csv':'./counterfactuals/OneHot_Adult_processedMACE.csv',
    './datasets/Credit-Card-Default_processedMACE.csv':'./counterfactuals/OneHot_Credit-Card-Default_processedMACE.csv'
}

runNumericalExperiments(mediumDatasetsDict,
    rf_max_depthList=range(3,9),
    rf_n_estimatorsList=[100],
    ilfActivatedList=[False],
    ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100],
    random_stateList=[1],
    useCuiList=[False],
    objectiveNormList=[0,1,2],
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure2_OCEAN.csv"
)

# Compute the numerical results necessary to plot the result of OAE for Figure 2 and store them in results/Figure2_OAE.csv"

runNumericalExperiments(mediumDatasetsDict,
    rf_max_depthList=range(3,9),
    rf_n_estimatorsList=[100],
    ilfActivatedList=[False],
    ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100],
    random_stateList=[1],
    useCuiList=[True],
    objectiveNormList=[0,1,2],
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure2_OAE.csv"
)

# Compute the numerical results necessary to plot the result of OCEAN for Figure 3 and store them in results/Figure3_OCEAN.csv

runNumericalExperiments(mediumDatasetsDict,
    rf_max_depthList=[5],
    rf_n_estimatorsList=[10,20,50,100,200,500],
    ilfActivatedList=[False],
    ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100],
    random_stateList=[1],
    useCuiList=[False],
    objectiveNormList=[0,1,2],
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure3_OCEAN.csv"
)

# Compute the numerical results necessary to plot the result of OAE for Figure 3 and store them in results/Figure3_OAE.csv
# Categorical features are handled as suggested by Cui et al. (even if in mediumDatasetsDicts the counterfactuals are provided with './counterfactuals/OneHot_...' 
# we get the information on categorical features from the dataset ./datasets/...')

runNumericalExperiments(mediumDatasetsDict,
    rf_max_depthList=[5],
    rf_n_estimatorsList=[10,20,50,100,200,500],
    ilfActivatedList=[False],
    ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100],
    random_stateList=[1],
    useCuiList=[True],
    objectiveNormList=[0,1,2],
    randomCostsActivated=False,
    numericalResultsFileName="results/Figure3_OAE.csv"
)

# Compute plausibility results for table 3 and store them in results/Table3.csv"
runNumericalExperiments(ourDatasetsWithCounterfactualsDict,
    rf_max_depthList=[5],
    rf_n_estimatorsList=[100],
    ilfActivatedList=[True],
    ilf_max_samplesList=[32],
    ilf_n_estimatorsList=[100],
    random_stateList=[1],
    useCuiList=[False],
    objectiveNormList=[0,1,2],
    randomCostsActivated=False,
    numericalResultsFileName="results/Table3.csv"
)

