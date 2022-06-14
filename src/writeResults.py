import os


def writeLegend(numericalResultsFileName):
    """ Write header of csv file if file is just created. """
    # Check that 'results' directory exists, create it if needed
    if not os.path.exists('results'):
        os.makedirs('results')
    write = open(numericalResultsFileName, "a")
    # Instance
    write.write("trainingSetFile"+","+"rf_max_depth"+','+"rf_n_estimators"
                + ','+"ilfActivated"+',' + "ilf_max_samples"
                + ','+"ilf_n_estimators" + ','+"random_state"
                + ','+"train_score"+","+"test_score")
    # Counterfactual
    write.write("," + "counterfactualsFile"+"," + "counterfactual_index")
    # Solver Parameters
    write.write(','+"useCui"+','+"constraintsType"
                + ','+"actionnabilityActivated"+','+"objectiveNorm"
                + ','+"mutuallyExclusivePlanesCutsActivated"
                + ','+"strictCounterFactual"+','+"binaryDecisionVariables")
    # Solver results
    write.write(","+"randomForestMilp.model.status"
                + ","+"randomForestMilp.runTime"
                + ","+"randomForestMilp.objValue"
                + ","+"notOuputDesired"+","+"maxSkLearnError"
                + ","+"maxMyMilpError"+","+"plausible")
    # Solver Solution
    write.write(','+"solution")
    # Finish
    write.write("\n")
    write.close()
