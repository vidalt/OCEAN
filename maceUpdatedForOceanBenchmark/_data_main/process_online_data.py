import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def load_online_data():

  # input vars
  data_name = 'online'
  raw_data_file = os.path.join(os.path.dirname(__file__), 'OnlineNewsPopularity.csv')
  processed_file = os.path.join(os.path.dirname(__file__), 'OnlineNewsPopularity-processed.csv')

  ##### OnlineNewsPopularity Data Processing
  raw_df = pd.read_csv(raw_data_file) # , index_col = 0)
  processed_df = pd.DataFrame()

  processed_df['Class (label)'] = raw_df['Class']
  processed_df['n_tokens_title'] = raw_df['n_tokens_title']
  processed_df['n_tokens_content'] = raw_df['n_tokens_content']
  processed_df['n_unique_tokens'] = raw_df['n_unique_tokens']
  processed_df['n_non_stop_words'] = raw_df['n_non_stop_words']
  processed_df['n_non_stop_unique_tokens'] = raw_df['n_non_stop_unique_tokens']
  processed_df['num_hrefs'] = raw_df['num_hrefs']
  processed_df['num_self_hrefs'] = raw_df['num_self_hrefs']
  processed_df['num_imgs'] = raw_df['num_imgs']
  processed_df['num_videos'] = raw_df['num_videos']
  processed_df['average_token_length'] = raw_df['average_token_length']
  processed_df['num_keywords'] = raw_df['num_keywords']
  processed_df.loc[raw_df['Data_Channel'] == 'Entertainment', 'Data_Channel'] = 1
  processed_df.loc[raw_df['Data_Channel'] == 'Bus', 'Data_Channel'] = 2
  processed_df.loc[raw_df['Data_Channel'] == 'Tech', 'Data_Channel'] = 3
  processed_df.loc[raw_df['Data_Channel'] == 'Lifestyle', 'Data_Channel'] = 4
  processed_df.loc[raw_df['Data_Channel'] == 'World', 'Data_Channel'] = 5
  processed_df.loc[raw_df['Data_Channel'] == 'Other', 'Data_Channel'] = 6
  processed_df.loc[raw_df['Data_Channel'] == 'Socmed', 'Data_Channel'] = 7
  processed_df['kw_min_min'] = raw_df['kw_min_min']
  processed_df['kw_max_min'] = raw_df['kw_max_min']
  processed_df['kw_avg_min'] = raw_df['kw_avg_min']
  processed_df['kw_min_max'] = raw_df['kw_min_max']
  processed_df['kw_max_max'] = raw_df['kw_max_max']
  processed_df['kw_avg_max'] = raw_df['kw_avg_max']
  processed_df['kw_min_avg'] = raw_df['kw_min_avg']
  processed_df['kw_max_avg'] = raw_df['kw_max_avg']
  processed_df['kw_avg_avg'] = raw_df['kw_avg_avg']
  processed_df['self_reference_min_shares'] = raw_df['self_reference_min_shares']
  processed_df['self_reference_max_shares'] = raw_df['self_reference_max_shares']
  processed_df['self_reference_avg_sharess'] = raw_df['self_reference_avg_sharess']
  processed_df.loc[raw_df['WeekDay'] == 'Monday', 'WeekDay'] = 1
  processed_df.loc[raw_df['WeekDay'] == 'Tuesday', 'WeekDay'] = 2
  processed_df.loc[raw_df['WeekDay'] == 'Wednesday', 'WeekDay'] = 3
  processed_df.loc[raw_df['WeekDay'] == 'Thursday', 'WeekDay'] = 4
  processed_df.loc[raw_df['WeekDay'] == 'Friday', 'WeekDay'] = 5
  processed_df.loc[raw_df['WeekDay'] == 'Saturday', 'WeekDay'] = 6
  processed_df.loc[raw_df['WeekDay'] == 'Sunday', 'WeekDay'] = 7
  processed_df['is_weekend'] = raw_df['is_weekend']
  processed_df['LDA_00'] = raw_df['LDA_00']
  processed_df['LDA_01'] = raw_df['LDA_01']
  processed_df['LDA_02'] = raw_df['LDA_02']
  processed_df['LDA_03'] = raw_df['LDA_03']
  processed_df['LDA_04'] = raw_df['LDA_04']
  processed_df['global_subjectivity'] = raw_df['global_subjectivity']
  processed_df['global_sentiment_polarity'] = raw_df['global_sentiment_polarity']
  processed_df['global_rate_positive_words'] = raw_df['global_rate_positive_words']
  processed_df['global_rate_negative_words'] = raw_df['global_rate_negative_words']
  processed_df['rate_positive_words'] = raw_df['rate_positive_words']
  processed_df['rate_negative_words'] = raw_df['rate_negative_words']
  processed_df['avg_positive_polarity'] = raw_df['avg_positive_polarity']
  processed_df['min_positive_polarity'] = raw_df['min_positive_polarity']
  processed_df['max_positive_polarity'] = raw_df['max_positive_polarity']
  processed_df['avg_negative_polarity'] = raw_df['avg_negative_polarity']
  processed_df['min_negative_polarity'] = raw_df['min_negative_polarity']
  processed_df['max_negative_polarity'] = raw_df['max_negative_polarity']
  processed_df['title_subjectivity'] = raw_df['title_subjectivity']
  processed_df['title_sentiment_polarity'] = raw_df['title_sentiment_polarity']
  processed_df['abs_title_subjectivity'] = raw_df['abs_title_subjectivity']
  processed_df['abs_title_sentiment_polarity'] = raw_df['abs_title_sentiment_polarity']

  # Save to CSV
  processed_df.to_csv(processed_file, header = True, index = False)
  return processed_df.astype('float64')
