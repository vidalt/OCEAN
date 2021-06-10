import os
import copy
import pickle
import argparse
import numpy as np
import pandas as pd

from pprint import pprint
from datetime import datetime

import newLoadData
import newLoadModel

import multiprocessing as mp
import time

# from debug import ipsh

try:
  import generateSATExplanations
except:
  print('[ENV WARNING] activate virtualenv to allow for testing MACE or MINT')
import generateMOExplanations
import generateFTExplanations
try:
  import generateARExplanations
except:
  print('[ENV WARNING] deactivate virtualenv to allow for testing Actionable Recourse')


from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


def getEpsilonInString(approach_string):
  tmp_index = approach_string.find('eps')
  epsilon_string = approach_string[tmp_index + 4 : tmp_index + 8]
  return float(epsilon_string)

def generateExplanationsWithMaxTime(maxTime,
  approach_string,
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type_string,
  observable_data_dict,
  standard_deviations):
  ctx = mp.get_context('spawn')
  q = ctx.Queue()
  p = ctx.Process(target=generateExplanationsWithQueueCatch, args=(q,
    approach_string,
    explanation_file_name,
    model_trained,
    dataset_obj,
    factual_sample,
    norm_type_string,
    observable_data_dict,
    standard_deviations,)
  )
  p.start()
  p.join(maxTime)
  if p.is_alive():
    p.terminate()
    print("killing after", maxTime, "second")
    return {
    'fac_sample': factual_sample,
    'cfe_found': False,
    'cfe_plausible': False,
    'cfe_time': maxTime,
    'cfe_sample': "stopped",
    'cfe_distance': float('inf'),
    }
  else:
    return q.get()


def generateExplanationsWithQueueCatch(queue,
  approach_string,
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type_string,
  observable_data_dict,
  standard_deviations):
  try:
    generateExplanationsWithQueue(queue,
    approach_string,
    explanation_file_name,
    model_trained,
    dataset_obj,
    factual_sample,
    norm_type_string,
    observable_data_dict,
    standard_deviations)
  except:
    print("solver returned error for", approach_string)
    queue.put({
    'fac_sample': factual_sample,
    'cfe_found': False,
    'cfe_plausible': False,
    'cfe_time': -1,
    'cfe_sample': "solver returned error",
    'cfe_distance': float('inf'),
    })

def generateExplanationsWithQueue(queue,
  approach_string,
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type_string,
  observable_data_dict,
  standard_deviations):
  queue.put(generateExplanations(
    approach_string,
    explanation_file_name,
    model_trained,
    dataset_obj,
    factual_sample,
    norm_type_string,
    observable_data_dict,
    standard_deviations
  ))

def generateExplanations(
  approach_string,
  explanation_file_name,
  model_trained,
  dataset_obj,
  factual_sample,
  norm_type_string,
  observable_data_dict,
  standard_deviations):

  if 'MACE' in approach_string: # 'MACE_counterfactual':

    return generateSATExplanations.genExp(
      explanation_file_name,
      model_trained,
      dataset_obj,
      factual_sample,
      norm_type_string,
      'mace',
      getEpsilonInString(approach_string)
    )

  elif 'MINT' in approach_string: # 'MINT_counterfactual':

    return generateSATExplanations.genExp(
      explanation_file_name,
      model_trained,
      dataset_obj,
      factual_sample,
      norm_type_string,
      'mint',
      getEpsilonInString(approach_string)
    )

  elif approach_string == 'MO': # 'minimum_observable':

    return generateMOExplanations.genExp(
      explanation_file_name,
      dataset_obj,
      factual_sample,
      observable_data_dict,
      norm_type_string
    )

  elif approach_string == 'FT': # 'feature_tweaking':

    possible_labels = [0, 1]
    epsilon = .5
    perform_while_plausibility = False
    return generateFTExplanations.genExp(
      model_trained,
      factual_sample,
      possible_labels,
      epsilon,
      norm_type_string,
      dataset_obj,
      standard_deviations,
      perform_while_plausibility
    )

  elif approach_string == 'PFT': # 'plausible_feature_tweaking':

    possible_labels = [0, 1]
    epsilon = .5
    perform_while_plausibility = True
    return generateFTExplanations.genExp(
      model_trained,
      factual_sample,
      possible_labels,
      epsilon,
      norm_type_string,
      dataset_obj,
      standard_deviations,
      perform_while_plausibility
    )

  elif approach_string == 'AR': # 'actionable_recourse':

    return generateARExplanations.genExp(
      model_trained,
      factual_sample,
      norm_type_string,
      dataset_obj
    )

  else:

    raise Exception(f'{approach_string} not recognized as a valid `approach_string`.')


def executeCommand(triple):
	count,total,command = triple
	print(count, "/", total, command)
	os.system(command)

def runSeparateExperiments(dataset_values, model_class_values, norm_values, approaches_values, batch_number, sample_count, gen_cf_for, process_id, max_depth, nb_estimators, nbCounterfactualsComputed = 'all', counterFactualFileUsed = False, maxTime=300):
  total = len(dataset_values) * len(model_class_values) * len(norm_values) * len(approaches_values)
  count = 0
  argList = []
  for dataset_string in dataset_values:

    print(f'\n\nExperimenting with dataset_string = `{dataset_string}`')

    for model_class_string in model_class_values:

      print(f'\tExperimenting with model_class_string = `{model_class_string}`')

      for norm_type_string in norm_values:

        print(f'\t\tExperimenting with norm_type_string = `{norm_type_string}`')

        for approach_string in approaches_values:

          print(f'\t\t\tExperimenting with approach_string = `{approach_string}`')

          counterfactual_filename = False
          if counterFactualFileUsed:
            counterfactual_filename = 'unknown'          
            if dataset_string == 'adult':
              counterfactual_filename = 'Adult_processedMACE.csv'
            elif dataset_string == 'compass':
              counterfactual_filename = 'COMPAS-ProPublica_processedMACE.csv'
            elif dataset_string == 'credit':
              counterfactual_filename = 'Credit-Card-Default_processedMACE.csv'
            elif dataset_string == 'german':
              counterfactual_filename = 'German-Credit.csv'
            elif dataset_string == 'online':
              counterfactual_filename = 'OnlineNewsPopularity.csv'
            elif dataset_string == "phishing":
              counterfactual_filename = 'Phishing.csv'
            elif dataset_string == 'spambase':
              counterfactual_filename = 'Spambase.csv'
            elif dataset_string == 'students':
              counterfactual_filename = 'Students-Performance-MAT.csv'

          count += 1
          command = "python newBatchTest.py -d " + dataset_string
          command += " -m " + model_class_string 
          command += " -md " + str(max_depth)
          command += " -ne " + str(nb_estimators)
          command += " -n " + norm_type_string
          command += " -a " + approach_string
          command += " -b " + str(batch_number)
          command += " -s " + str(sample_count)
          command += " -g " + gen_cf_for
          command += " -p " + str(process_id)
          command += " -t " + str(maxTime)
          argList.append((count,total,command))
          # print(command)
          # os.system(command)
          # runExperiment(dataset_string, counterfactual_filename, model_class_string, norm_type_string, approach_string, gen_cf_for, process_id, max_depth, nb_estimators, nbCounterfactualsComputed=nbCounterfactualsComputed, batch_number=batch_number, sample_count=sample_count)  

  #Parrallel Execution
  start = time.time()
  # nbCpu = 1
  nbCpu = mp.cpu_count() - 1
  print("use", nbCpu, "out of ", mp.cpu_count(), "cpu")
  # pool = mp.Pool(mp.cpu_count() - 2)
  pool = mp.Pool(nbCpu)
  pool.map(executeCommand, argList)

  end = time.time()
  print(end - start)

def runExperiments(dataset_values, model_class_values, norm_values, approaches_values, batch_number, sample_count, gen_cf_for, process_id, max_depth, nb_estimators, nbCounterfactualsComputed = 'all', counterFactualFileUsed = False, maxTime=300):

  for dataset_string in dataset_values:

    print(f'\n\nExperimenting with dataset_string = `{dataset_string}`')

    for model_class_string in model_class_values:

      print(f'\tExperimenting with model_class_string = `{model_class_string}`')

      for norm_type_string in norm_values:

        print(f'\t\tExperimenting with norm_type_string = `{norm_type_string}`')

        for approach_string in approaches_values:

          print(f'\t\t\tExperimenting with approach_string = `{approach_string}`')

          counterfactual_filename = False
          if counterFactualFileUsed:
            counterfactual_filename = 'unknown'          
            if dataset_string == 'adult':
              counterfactual_filename = 'Adult_processedMACE.csv'
            elif dataset_string == 'compass':
              counterfactual_filename = 'COMPAS-ProPublica_processedMACE.csv'
            elif dataset_string == 'credit':
              counterfactual_filename = 'Credit-Card-Default_processedMACE.csv'
            elif dataset_string == 'german':
              counterfactual_filename = 'German-Credit.csv'
            elif dataset_string == 'online':
              counterfactual_filename = 'OnlineNewsPopularity.csv'
            elif dataset_string == "phishing":
              counterfactual_filename = 'Phishing.csv'
            elif dataset_string == 'spambase':
              counterfactual_filename = 'Spambase.csv'
            elif dataset_string == 'students':
              counterfactual_filename = 'Students-Performance-MAT.csv'

          runExperiment(dataset_string, counterfactual_filename, model_class_string, norm_type_string, approach_string, gen_cf_for, process_id, max_depth, nb_estimators, nbCounterfactualsComputed=nbCounterfactualsComputed, batch_number=batch_number, sample_count=sample_count, maxTime=maxTime)

def runExperiment(dataset_string, counterfactual_filename, model_class_string, norm_type_string, approach_string, gen_cf_for, process_id, max_depth, nb_estimators, nbCounterfactualsComputed='all', batch_number=0,sample_count=20, maxTime = 3):
  # if norm_type_string == 'two_norm':
  #   raise Exception(f'{norm_type_string} not supported.')

  if model_class_string in {'tree', 'forest'}:
    one_hot = False
  elif model_class_string in {'lr', 'mlp'}:
    one_hot = True
  else:
    raise Exception(f'{model_class_string} not recognized as a valid `model_class_string`.')

  # prepare experiment folder
  experiment_name = f'{dataset_string}__{model_class_string}__{norm_type_string}__{approach_string}__pid{process_id}_md{max_depth}_ne{nb_estimators}'
  experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{experiment_name}"
  explanation_folder_name = f'{experiment_folder_name}/__explanation_log'
  minimum_distance_folder_name = f'{experiment_folder_name}/__minimum_distances'
  os.mkdir(f'{experiment_folder_name}')
  os.mkdir(f'{explanation_folder_name}')
  os.mkdir(f'{minimum_distance_folder_name}')
  log_file = open(f'{experiment_folder_name}/log_experiment.txt','w')

  # save some files
  dataset_obj = newLoadData.loadDataset(dataset_string, return_one_hot = one_hot, load_from_cache = False, debug_flag = False)
  pickle.dump(dataset_obj, open(f'{experiment_folder_name}/_dataset_obj', 'wb'))
  #     training portion used to train models
  #     testing portion used to compute counterfactuals
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()

  standard_deviations = list(X_train.std())

  # train the model
  # model_trained = modelTraining.trainAndSaveModels(
  #   model_class_string,
  #   dataset_string,
  #   experiment_folder_name,
  # )
  model_trained = newLoadModel.loadModelForDataset(
    model_class_string,
    dataset_string,
    experiment_folder_name = experiment_folder_name, max_depth=max_depth, nb_estimators=nb_estimators)

  # get the predicted labels (only test set)
  # X_test = pd.concat([X_train, X_test]) # ONLY ACTIVATE THIS WHEN TEST SET IS NOT LARGE ENOUGH TO GEN' MODEL RECON DATASET
  X_test_pred_labels = model_trained.predict(X_test)

  all_pred_data_df = X_test
  # IMPORTANT: note that 'y' is actually 'pred_y', not 'true_y'
  all_pred_data_df['y'] = X_test_pred_labels
  neg_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 0).dropna()
  pos_pred_data_df = all_pred_data_df.where(all_pred_data_df['y'] == 1).dropna()

  counterfactualsData = False
  if counterfactual_filename:
    # NEW WAY OF GENERATING counterfactuals

    counterfactualsData = pd.read_csv("_data_counterfactuals/" + counterfactual_filename)
    # normalize
    dataset_forNormalization = newLoadData.loadDataset(dataset_string, return_one_hot = one_hot, load_from_cache = False, debug_flag = False)
    count = 1
    iterate_over_data_dict = dict()
    for index in counterfactualsData.index:
      x0 = [counterfactualsData.iloc[index, counterfactualsData.columns != 'DesiredOutcome']]
      for longName in x0[0].keys():
        kurzName = dataset_forNormalization.attributes_long[longName].attr_name_kurz
        iterate_over_data_dict[count] = dict()
        iterate_over_data_dict[count][kurzName] = x0[0][longName]
      #   if counterfactualsData.columns[i] != dataset_forNormalization.attributes_kurz['x'+str(i)].attr_name_long:
      #     print("Error", counterfactualsData.columns[i], "instead of", dataset_forNormalization.attributes_kurz['x'+str(i)].attr_name_long)
      iterate_over_data_dict[count] = {'x'+str(i):x0[0][i] for i in range(len(x0[0]))}
      iterate_over_data_dict[count]['y'] = not counterfactualsData['DesiredOutcome'][index]
      # if preprocessing == 'normalize': # THis should not be used because features are normalized within one norm computation, and so the normalize preprocessing is useless and makes other things bug
      #   print("Normalize !!!")
      #   for attr_name_kurz in dataset_forNormalization.getNonHotAttributesNames('kurz'):
      #     attr_obj = dataset_forNormalization.attributes_kurz[attr_name_kurz]
      #     lower_bound = attr_obj.lower_bound
      #     upper_bound =attr_obj.upper_bound
      #     iterate_over_data_dict[count][attr_name_kurz] = (iterate_over_data_dict[count][attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)
      if type(nbCounterfactualsComputed) == int and nbCounterfactualsComputed == count:
        break
    count += 1
  else:
      # OLD WAY OF GENERATING counterfactuals (kept for observableDataDict)
      batch_start_index = batch_number * sample_count
      batch_end_index = (batch_number + 1) * sample_count

      # generate counterfactuals for {only negative, negative & positive} samples
      if gen_cf_for == 'neg_only':
        iterate_over_data_df = neg_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
      elif gen_cf_for == 'pos_only':
        iterate_over_data_df = pos_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
      elif gen_cf_for == 'neg_and_pos':
        iterate_over_data_df = all_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
      else:
        raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

      iterate_over_data_dict = iterate_over_data_df.T.to_dict()

  # OLD WAY OF GENERATING counterfactuals (kept for observableDataDict)
  # batch_start_index = batch_number * sample_count
  # batch_end_index = (batch_number + 1) * sample_count

  # generate counterfactuals for {only negative, negative & positive} samples
  if gen_cf_for == 'neg_only':
    # iterate_over_data_df = neg_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
    observable_data_df = pos_pred_data_df
  elif gen_cf_for == 'pos_only':
    # iterate_over_data_df = pos_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
    observable_data_df = neg_pred_data_df
  elif gen_cf_for == 'neg_and_pos':
    # iterate_over_data_df = all_pred_data_df[batch_start_index : batch_end_index] # choose only a subset to compare
    observable_data_df = all_pred_data_df
  else:
    raise Exception(f'{gen_cf_for} not recognized as a valid `gen_cf_for`.')

  # # convert to dictionary for easier enumeration (iteration)
  # iterate_over_data_dict = iterate_over_data_df.T.to_dict()
  observable_data_dict = observable_data_df.T.to_dict()

  # loop through samples for which we desire a counterfactual,
  # (to be saved as part of the same file of minimum distances)
  explanation_counter = 1
  all_minimum_distances = {}
  for factual_sample_index, factual_sample in iterate_over_data_dict.items():

    factual_sample['y'] = bool(factual_sample['y'])

    print(
      '\t\t\t\t'
      f'Generating explanation for\t'
      f'sample #{explanation_counter}/{len(iterate_over_data_dict.keys())}\t'
      f'(sample index {factual_sample_index}): ', end = '') # , file=log_file)
    explanation_counter = explanation_counter + 1
    explanation_file_name = f'{explanation_folder_name}/sample_{factual_sample_index}.txt'

    # new : normalize factual sample

  
    explanation_object = generateExplanationsWithMaxTime(
      maxTime,
      approach_string,
      explanation_file_name,
      model_trained,
      dataset_obj,
      factual_sample,
      norm_type_string,
      observable_data_dict, # used solely for minimum_observable method
      standard_deviations, # used solely for feature_tweaking method
    )

    if 'MINT' in approach_string:
      print(
        f'\t- scf_found: {explanation_object["scf_found"]} -'
        f'\t- scf_plaus: {explanation_object["scf_plausible"]} -'
        f'\t- scf_time: {explanation_object["scf_time"]:.4f} -'
        f'\t- int_cost: {explanation_object["int_cost"]:.4f} -'
        f'\t- scf_dist: {explanation_object["scf_distance"]:.4f} -'
      ) # , file=log_file)
    else: # 'MACE' or other..
      print(
        f'\t- cfe_found: {explanation_object["cfe_found"]} -'
        f'\t- cfe_plaus: {explanation_object["cfe_plausible"]} -'
        f'\t- cfe_time: {explanation_object["cfe_time"]:.4f} -'
        f'\t- int_cost: N/A -'
        f'\t- cfe_dist: {explanation_object["cfe_distance"]:.4f} -'
      ) # , file=log_file)

    all_minimum_distances[f'sample_{factual_sample_index}'] = explanation_object

  pickle.dump(all_minimum_distances, open(f'{experiment_folder_name}/_minimum_distances', 'wb'))
  pprint(all_minimum_distances, open(f'{experiment_folder_name}/minimum_distances.txt', 'w'))

  def printNumericalResults(all_minimum_distances, dataset_obj):
    results = open(experiment_folder_name + '/NumericalResults.csv', 'w')
    for sampleName in all_minimum_distances:
      sample = all_minimum_distances[sampleName]
      results.write(sampleName + " ")
      results.write(str(sample['cfe_found']) + " ")
      results.write(str(sample['cfe_distance']) + " ")
      results.write(str(sample['cfe_plausible']) + " ")
      results.write(str(sample['cfe_time']) + " ")
      if sample['cfe_found']:
        oneNorm = 0.0
        x1 = sample['cfe_sample']
        facSampleName = 'factual_sample'
        if facSampleName not in sample:
          facSampleName = 'fac_sample'
        x2 = sample[facSampleName]
        for x in x1:
          if x == 'y':
            continue
          oneNorm += abs(x1[x] - x2[x])/(dataset_obj.attributes_kurz[x].upper_bound - dataset_obj.attributes_kurz[x].lower_bound)
        results.write(str(oneNorm))
      else:
        results.write('inf')
      results.write("\n")
    results.close()
  
  printNumericalResults(all_minimum_distances, dataset_obj)

  def saveSamplesProducedToCsv(all_minimum_distances, dataset_obj, counterfactualsData):
    results = open(experiment_folder_name + '/samplesProduced.csv', 'w')
    firstLine = ""
    for col in counterfactualsData:
      firstLine += col+','
    results.write(firstLine[:-1]+'\n')
    for sampleName in all_minimum_distances:
      line = ""
      sample = all_minimum_distances[sampleName]
      if sample['cfe_distance'] != float('inf'):
        for col in counterfactualsData:
          kurzName = 'y'
          if col != "DesiredOutcome":
            kurzName = dataset_obj.attributes_long[col].attr_name_kurz
          line += str(sample['cfe_sample'][kurzName])+","
      else:
        for col in counterfactualsData:
          if col != "DesiredOutcome":
            line += "inf,"
          else:
            line += "False,"
      results.write(line[:-1]+'\n')
    results.close()

  if counterfactualsData:
    saveSamplesProducedToCsv(all_minimum_distances, dataset_obj, counterfactualsData)


  

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '-d', '--dataset',
      nargs = '+',
      type = str,
      default = 'compass',
      help = 'Name of dataset to train model on: compass, credit, adult, german, online, phishing, spambase, students')

  parser.add_argument(
      '-m', '--model_class',
      nargs = '+',
      type = str,
      default = 'tree',
      help = 'Model class that will learn data: tree, forest, lr, mlp')

  parser.add_argument(
      '-md', '--max_depth',
      type=int,
      default=5,
      help= 'max depth of the trees used by the random forest'
  )

  parser.add_argument(
      '-ne', '--nb_estimators',
      type=int,
      default=10,
      help= 'nb estimators used by the random forest'
  )

  parser.add_argument(
      '-n', '--norm_type',
      nargs = '+',
      type = str,
      default = 'zero_norm',
      help = 'Norm used to evaluate distance to counterfactual: zero_norm, one_norm, infty_norm') # two_norm

  parser.add_argument(
      '-a', '--approach',
      nargs = '+',
      type = str,
      default = 'MACE_eps_1e-5',
      help = 'Approach used to generate counterfactual: MACE_eps_1e-3, MINT_eps_1e-3, MO, FT, AR.') # ES

  parser.add_argument(
      '-b', '--batch_number',
      type = int,
      default = -1,
      help = 'If b = b, s = s, compute explanations for samples in range( b * s, (b + 1) * s )).')

  parser.add_argument(
      '-s', '--sample_count',
      type = int,
      default = 5,
      help = 'Number of samples seeking explanations.')

  parser.add_argument(
      '-g', '--gen_cf_for',
      type = str,
      default = 'neg_only',
      help = 'Decide whether to generate counterfactuals for negative pred samples only, or for both negative and positive pred samples.')

  parser.add_argument(
      '-p', '--process_id',
      type = str,
      default = '0',
      help = 'When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')

  parser.add_argument(
      '-t', '--max_time_sec',
      type = int,
      default = '300',
      help = 'Maximum time for one single run in seconds (int). Default=300 (which makes 300s)'
  )


  # parsing the args
  args = parser.parse_args()

  if 'FT' in args.approach or 'PFT' in args.approach:
    assert len(args.model_class) == 1, 'FeatureTweaking approach only works with forests.'
    assert \
      args.model_class[0] == 'tree' or args.model_class[0] == 'forest', \
      'FeatureTweaking approach only works with forests.'
  elif 'AR' in args.approach:
    assert len(args.model_class) == 1, 'actionableRecourse approach only works with larger.'
    assert args.model_class[0] == 'lr', 'actionableRecourse approach only works with larger.'

  runExperiments(
    args.dataset,
    args.model_class,
    args.norm_type,
    args.approach,
    args.batch_number,
    args.sample_count,
    args.gen_cf_for,
    args.process_id,
    args.max_depth,
    args.nb_estimators,
    maxTime = args.max_time_sec)










