import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def load_german_data():

  # input vars
  data_name = 'german'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'German-Credit.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'German-Credit-processed.csv')

  ##### German Data Processing -- TV: Updated preprocessing to use the same data set version
  raw_df = pd.read_csv(raw_data_file) # , index_col = 0)
  processed_df = pd.DataFrame()

  processed_df['Class (label)'] = raw_df['Class']
  processed_df['Age'] = raw_df['Age']
  processed_df.loc[raw_df['Sex'] == 'male', 'Sex'] = 1
  processed_df.loc[raw_df['Sex'] == 'female', 'Sex'] = 0
  processed_df['Job'] = raw_df['Job']+1
  processed_df['Housing'] = raw_df['Housing']+1
  processed_df['SavingAccounts'] = raw_df['SavingAccounts']+1
  processed_df['CheckingAccount'] = raw_df['CheckingAccount']+1
  processed_df['CreditAmount'] = raw_df['CreditAmount']
  processed_df['Duration'] = raw_df['Duration']
  processed_df['Purpose'] = raw_df['Purpose']

  # Save to CSV
  processed_df.to_csv(processed_file, header = True, index = False)
  return processed_df.astype('float64')
