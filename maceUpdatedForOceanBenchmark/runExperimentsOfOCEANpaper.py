from newBatchTest import *
import os

if __name__ == '__main__':

    # Create target Directory if don't exist
    dirName = "_experiments"
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

    for depth in range(6,9):
        runSeparateExperiments(['compass','credit','adult','german','phishing','students','online','spambase'],['forest'],['one_norm'],['MACE_eps_1e-5','MACE_eps_1e-3','FT'],1,20,'neg_only','2',depth,100, nbCounterfactualsComputed='all', maxTime=900)

    # depth 100 is not included because already computed elsewhere
    for nbTree in [10,20,50,200,500]:
        runSeparateExperiments(['compass','credit','adult','german','phishing','students','online','spambase'],['forest'],['one_norm'],['MACE_eps_1e-5','MACE_eps_1e-3','FT'],1,20,'neg_only','2',5,nbTree, nbCounterfactualsComputed='all', maxTime=900)        



