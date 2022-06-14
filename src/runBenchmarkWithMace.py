# Important note: This file is given to reproduce the Benchmark of OCEAN paper.
# It is not part of the OCEAN package and will not be updated with new versions
# of OCEAN.

from benchmarks.RunExperimentsOnMaceOutput import runExperimentsOnAllMaceSubfolders


maceFolder = 'path/to/mace/_experiments'
runExperimentsOnAllMaceSubfolders(maceFolder, objectiveNorm=1, runMilp=True)
