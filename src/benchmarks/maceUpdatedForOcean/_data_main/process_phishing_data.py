import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def load_phishing_data():

  # input vars
  data_name = 'phishing'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'Phishing.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'Phishing-processed.csv')

  ##### Phishing Data Processing 
  raw_df = pd.read_csv(raw_data_file) # , index_col = 0)
  processed_df = pd.DataFrame()

  processed_df['Class (label)'] = raw_df['Class']
  processed_df['havingIPAddress'] = raw_df['havingIPAddress']
  processed_df['URLLength'] = raw_df['URLLength']+1
  processed_df['ShortiningService'] = raw_df['ShortiningService']
  processed_df['havingAtSymbol'] = raw_df['havingAtSymbol']
  processed_df['doubleslashredirecting'] = raw_df['doubleslashredirecting']
  processed_df['PrefixSuffix'] = raw_df['PrefixSuffix']
  processed_df['havingSubDomain'] = raw_df['havingSubDomain']+1
  processed_df['SSLfinalState'] = raw_df['SSLfinalState']+1
  processed_df['Domainregisterationlength'] = raw_df['Domainregisterationlength']
  processed_df['Favicon'] = raw_df['Favicon']
  processed_df['port'] = raw_df['port']
  processed_df['HTTPStoken'] = raw_df['HTTPStoken']
  processed_df['RequestURL'] = raw_df['RequestURL']
  processed_df['URLofAnchor'] = raw_df['URLofAnchor']+1
  processed_df['Linksintags'] = raw_df['Linksintags']+1
  processed_df['SFH'] = raw_df['SFH']+1
  processed_df['Submittingtoemail'] = raw_df['Submittingtoemail']
  processed_df['AbnormalURL'] = raw_df['AbnormalURL']
  processed_df['Redirect'] = raw_df['Redirect']
  processed_df['onmouseover'] = raw_df['onmouseover']
  processed_df['RightClick'] = raw_df['RightClick']
  processed_df['popUpWidnow'] = raw_df['popUpWidnow']
  processed_df['Iframe'] = raw_df['Iframe']
  processed_df['ageofdomain'] = raw_df['ageofdomain']
  processed_df['DNSRecord'] = raw_df['DNSRecord']
  processed_df['webtraffic'] = raw_df['webtraffic']+1
  processed_df['PageRank'] = raw_df['PageRank']
  processed_df['GoogleIndex'] = raw_df['GoogleIndex']
  processed_df['Linkspointingtopage'] = raw_df['Linkspointingtopage']+1
  processed_df['Statisticalreport'] = raw_df['Statisticalreport']

  # Save to CSV
  processed_df.to_csv(processed_file, header = True, index = False)
  return processed_df.astype('float64')
