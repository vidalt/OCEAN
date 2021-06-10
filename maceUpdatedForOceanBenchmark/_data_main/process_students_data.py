import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def load_students_data():

  # input vars
  data_name = 'students'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'Students-Performance-MAT.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'Students-Performance-MAT-processed.csv')

  ##### Students Data Processing
  raw_df = pd.read_csv(raw_data_file) # , index_col = 0)
  processed_df = pd.DataFrame()

  processed_df['Class (label)'] = raw_df['Class']
  processed_df.loc[raw_df['school'] == 'GP', 'school'] = 0
  processed_df.loc[raw_df['school'] == 'MS', 'school'] = 1
  processed_df.loc[raw_df['sex'] == 'F', 'sex'] = 0
  processed_df.loc[raw_df['sex'] == 'M', 'sex'] = 1
  processed_df['age'] = raw_df['age']
  processed_df.loc[raw_df['address'] == 'U', 'address'] = 0
  processed_df.loc[raw_df['address'] == 'R', 'address'] = 1
  processed_df.loc[raw_df['famsize'] == 'LE3', 'famsize'] = 0
  processed_df.loc[raw_df['famsize'] == 'GT3', 'famsize'] = 1
  processed_df.loc[raw_df['Pstatus'] == 'A', 'Pstatus'] = 0
  processed_df.loc[raw_df['Pstatus'] == 'T', 'Pstatus'] = 1
  processed_df['Medu'] = raw_df['Medu']+1
  processed_df['Fedu'] = raw_df['Fedu']+1

  processed_df.loc[raw_df['Mjob'] == 'teacher', 'Mjob'] = 1
  processed_df.loc[raw_df['Mjob'] == 'other', 'Mjob'] = 2
  processed_df.loc[raw_df['Mjob'] == 'services', 'Mjob'] = 3
  processed_df.loc[raw_df['Mjob'] == 'health', 'Mjob'] = 4
  processed_df.loc[raw_df['Mjob'] == 'at_home', 'Mjob'] = 5

  processed_df.loc[raw_df['Fjob'] == 'teacher', 'Fjob'] = 1
  processed_df.loc[raw_df['Fjob'] == 'other', 'Fjob'] = 2
  processed_df.loc[raw_df['Fjob'] == 'services', 'Fjob'] = 3
  processed_df.loc[raw_df['Fjob'] == 'health', 'Fjob'] = 4
  processed_df.loc[raw_df['Fjob'] == 'at_home', 'Fjob'] = 5

  processed_df.loc[raw_df['reason'] == 'course', 'reason'] = 1
  processed_df.loc[raw_df['reason'] == 'other', 'reason'] = 2
  processed_df.loc[raw_df['reason'] == 'home', 'reason'] = 3
  processed_df.loc[raw_df['reason'] == 'reputation', 'reason'] = 4

  processed_df.loc[raw_df['guardian'] == 'mother', 'guardian'] = 1
  processed_df.loc[raw_df['guardian'] == 'father', 'guardian'] = 2
  processed_df.loc[raw_df['guardian'] == 'other', 'guardian'] = 3

  processed_df['traveltime'] = raw_df['traveltime']
  processed_df['studytime'] = raw_df['studytime']
  processed_df['failures'] = raw_df['failures']
  processed_df['schoolsup'] = raw_df['schoolsup']
  processed_df['famsup'] = raw_df['famsup']
  processed_df['paid'] = raw_df['paid']
  processed_df['activities'] = raw_df['activities']
  processed_df['nursery'] = raw_df['nursery']
  processed_df['higher'] = raw_df['higher']
  processed_df['internet'] = raw_df['internet']
  processed_df['romantic'] = raw_df['romantic']
  processed_df['famrel'] = raw_df['famrel']
  processed_df['freetime'] = raw_df['freetime']
  processed_df['goout'] = raw_df['goout']
  processed_df['Dalc'] = raw_df['Dalc']
  processed_df['Walc'] = raw_df['Walc']
  processed_df['health'] = raw_df['health']
  processed_df['absences'] = raw_df['absences']

  # Save to CSV
  processed_df.to_csv(processed_file, header = True, index = False)
  return processed_df.astype('float64')
