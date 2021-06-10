import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def load_spambase_data():

  # input vars
  data_name = 'spambase'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'Spambase.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'Spambase-processed.csv')

  ##### Spambase Data Processing
  raw_df = pd.read_csv(raw_data_file) # , index_col = 0)
  processed_df = pd.DataFrame()

  processed_df['Class (label)'] = raw_df['Class']
  processed_df['F1'] = raw_df['F1']
  processed_df['F2'] = raw_df['F2']
  processed_df['F3'] = raw_df['F3']
  processed_df['F4'] = raw_df['F4']
  processed_df['F5'] = raw_df['F5']
  processed_df['F6'] = raw_df['F6']
  processed_df['F7'] = raw_df['F7']
  processed_df['F8'] = raw_df['F8']
  processed_df['F9'] = raw_df['F9']
  processed_df['F10'] = raw_df['F10']
  processed_df['F11'] = raw_df['F11']
  processed_df['F12'] = raw_df['F12']
  processed_df['F13'] = raw_df['F13']
  processed_df['F14'] = raw_df['F14']
  processed_df['F15'] = raw_df['F15']
  processed_df['F16'] = raw_df['F16']
  processed_df['F17'] = raw_df['F17']
  processed_df['F18'] = raw_df['F18']
  processed_df['F19'] = raw_df['F19']
  processed_df['F20'] = raw_df['F20']
  processed_df['F21'] = raw_df['F21']
  processed_df['F22'] = raw_df['F22']
  processed_df['F23'] = raw_df['F23']
  processed_df['F24'] = raw_df['F24']
  processed_df['F25'] = raw_df['F25']
  processed_df['F26'] = raw_df['F26']
  processed_df['F27'] = raw_df['F27']
  processed_df['F28'] = raw_df['F28']
  processed_df['F29'] = raw_df['F29']
  processed_df['F30'] = raw_df['F30']
  processed_df['F31'] = raw_df['F31']
  processed_df['F32'] = raw_df['F32']
  processed_df['F33'] = raw_df['F33']
  processed_df['F34'] = raw_df['F34']
  processed_df['F35'] = raw_df['F35']
  processed_df['F36'] = raw_df['F36']
  processed_df['F37'] = raw_df['F37']
  processed_df['F38'] = raw_df['F38']
  processed_df['F39'] = raw_df['F39']
  processed_df['F40'] = raw_df['F40']
  processed_df['F41'] = raw_df['F41']
  processed_df['F42'] = raw_df['F42']
  processed_df['F43'] = raw_df['F43']
  processed_df['F44'] = raw_df['F44']
  processed_df['F45'] = raw_df['F45']
  processed_df['F46'] = raw_df['F46']
  processed_df['F47'] = raw_df['F47']
  processed_df['F48'] = raw_df['F48']
  processed_df['F49'] = raw_df['F49']
  processed_df['F50'] = raw_df['F50']
  processed_df['F51'] = raw_df['F51']
  processed_df['F52'] = raw_df['F52']
  processed_df['F53'] = raw_df['F53']
  processed_df['F54'] = raw_df['F54']
  processed_df['F55'] = raw_df['F55']
  processed_df['F56'] = raw_df['F56']
  processed_df['F57'] = raw_df['F57']

  # Save to CSV
  processed_df.to_csv(processed_file, header = True, index = False)
  return processed_df.astype('float64')
