import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split

import utils
# from debug import ipsh

sys.path.insert(0, '_data_main')

try:
  from _data_main.fair_adult_data import *
except:
  print('[ENV WARNING] fair_adult_data not available')

try:
  from _data_main.fair_compas_data import *
except:
  print('[ENV WARNING] fair_compas_data not available')

try:
  from _data_main.process_credit_data import *
except:
  print('[ENV WARNING] process_credit_data not available')

try:
  from _data_main.process_german_data import *
except:
  print('[ENV WARNING] process_german_data not available')

try:
  from _data_main.process_online_data import *
except:
  print('[ENV WARNING] process_german_data not available')

try:
  from _data_main.process_phishing_data import *
except:
  print('[ENV WARNING] process_german_data not available')

try:
  from _data_main.process_spambase_data import *
except:
  print('[ENV WARNING] process_german_data not available')

try:
  from _data_main.process_students_data import *
except:
  print('[ENV WARNING] process_german_data not available')

try:
  from _data_main.process_synthetic_data import *
except:
  print('[ENV WARNING] process_synthetic_data not available')

try:
  from _data_main.process_mortgage_data import *
except:
  print('[ENV WARNING] process_mortgage_data not available')

try:
  from _data_main.process_twomoon_data import *
except:
  print('[ENV WARNING] process_twomoon_data not available')

try:
  from _data_main.process_test_data import *
except:
  print('[ENV WARNING] process_test_data not available')

VALID_ATTRIBUTE_DATA_TYPES = { \
  'numeric-int', \
  'numeric-real', \
  'binary', \
  'categorical', \
  'sub-categorical', \
  'ordinal', \
  'sub-ordinal'}
VALID_ATTRIBUTE_NODE_TYPES = { \
  'meta', \
  'input', \
  'output'}
VALID_ACTIONABILITY_TYPES = { \
  'none', \
  'any', \
  'same-or-increase', \
  'same-or-decrease'}
VALID_MUTABILITY_TYPES = { \
  True, \
  False}

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

class Dataset(object):

  # TODO: getOneHotEquivalent can be a class method, and this object can store
  # both one-hot and non-hot versions!

  def __init__(self, data_frame, attributes, is_one_hot, dataset_name):

    self.dataset_name = dataset_name

    self.is_one_hot = is_one_hot

    attributes_long = attributes
    data_frame_long = data_frame
    self.data_frame_long = data_frame_long # i.e., data_frame is indexed by attr_name_long
    self.attributes_long = attributes_long # i.e., attributes is indexed by attr_name_long

    attributes_kurz = dict((attributes[key].attr_name_kurz, value) for (key, value) in attributes_long.items())
    data_frame_kurz = copy.deepcopy(data_frame_long)
    data_frame_kurz.columns = self.getAllAttributeNames('kurz')
    self.data_frame_kurz = data_frame_kurz # i.e., data_frame is indexed by attr_name_kurz
    self.attributes_kurz = attributes_kurz # i.e., attributes is indexed by attr_name_kurz

    # assert that data_frame and attributes match on variable names (long)
    assert len(np.setdiff1d(
      data_frame.columns.values,
      np.array(self.getAllAttributeNames('long'))
    )) == 0

    # assert attribute type matches what is in the data frame
    for attr_name in np.setdiff1d(
      self.getInputAttributeNames('long'),
      self.getRealBasedAttributeNames('long'),
    ):
      unique_values = np.unique(data_frame_long[attr_name].to_numpy())
      # all non-numerical-real values should be integer or {0,1}
      for value in unique_values:
        assert value == np.floor(value)
      if is_one_hot and attributes_long[attr_name].attr_type != 'numeric-int': # binary, sub-categorical, sub-ordinal
        try:
          assert \
            np.array_equal(unique_values, [0,1]) or \
            np.array_equal(unique_values, [1,2]) or \
            np.array_equal(unique_values, [1]) # the first sub-ordinal attribute is always 1
            # race (binary) in compass is encoded as {1,2}
        except:
          ipsh()

    # # assert attributes and is_one_hot agree on one-hot-ness (i.e., if is_one_hot,
    # # then at least one attribute should be encoded as one-hot (w/ parent reference))
    # tmp_is_one_hot = False
    # for attr_name in attributes.keys():
    #   attr_obj = attributes[attr_name]
    #   # this simply checks to make sure that at least one elem is one-hot encoded
    #   if attr_obj.parent_name_long != -1 or attr_obj.parent_name_kurz != -1:
    #     tmp_is_one_hot = True
    # # TODO: assert only if there is a cat/ord variable!
    # assert is_one_hot == tmp_is_one_hot, "Dataset object and actual attributes don't agree on one-hot"

    self.assertSiblingsShareAttributes('long')
    self.assertSiblingsShareAttributes('kurz')

  def getAttributeNames(self, allowed_node_types, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_name
    for attr_name in self.attributes_long.keys():
      attr_obj = self.attributes_long[attr_name]
      if attr_obj.node_type not in allowed_node_types:
        continue
      if long_or_kurz == 'long':
        names.append(attr_obj.attr_name_long)
      elif long_or_kurz == 'kurz':
        names.append(attr_obj.attr_name_kurz)
      else:
        raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getAllAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'meta', 'input', 'output'}, long_or_kurz)

  def getInputOutputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'input', 'output'}, long_or_kurz)

  def getMetaInputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'meta', 'input'}, long_or_kurz)

  def getMetaAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'meta'}, long_or_kurz)

  def getInputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'input'}, long_or_kurz)

  def getOutputAttributeNames(self, long_or_kurz = 'kurz'):
    return self.getAttributeNames({'output'}, long_or_kurz)

  def getBinaryAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check binary
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.node_type == 'input' and attr_obj.attr_type == 'binary':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getActionableAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check actionability
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.node_type == 'input' and attr_obj.actionability != 'none':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getNonActionableAttributeNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getActionableAttributeNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getMutableAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check mutability
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.node_type == 'input' and attr_obj.mutability != False:
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getNonMutableAttributeNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getMutableAttributeNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getIntegerBasedAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_type
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.attr_type == 'numeric-int':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def getRealBasedAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_type
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.attr_type == 'numeric-real':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')
    return np.array(names)

  def assertSiblingsShareAttributes(self, long_or_kurz = 'kurz'):
    # assert elems of dictOfSiblings share attr_type, node_type, parent, actionability, and mutability
    dict_of_siblings = self.getDictOfSiblings(long_or_kurz)
    for parent_name in dict_of_siblings['cat'].keys():
      siblings = dict_of_siblings['cat'][parent_name]
      assert len(siblings) > 1
      for sibling in siblings:
        if long_or_kurz == 'long':
          self.attributes_long[sibling].attr_type = self.attributes_long[siblings[0]].attr_type
          self.attributes_long[sibling].node_type = self.attributes_long[siblings[0]].node_type
          self.attributes_long[sibling].actionability = self.attributes_long[siblings[0]].actionability
          self.attributes_long[sibling].mutability = self.attributes_long[siblings[0]].mutability
          self.attributes_long[sibling].parent_name_long = self.attributes_long[siblings[0]].parent_name_long
          self.attributes_long[sibling].parent_name_kurz = self.attributes_long[siblings[0]].parent_name_kurz
        elif long_or_kurz == 'kurz':
          self.attributes_kurz[sibling].attr_type = self.attributes_kurz[siblings[0]].attr_type
          self.attributes_kurz[sibling].node_type = self.attributes_kurz[siblings[0]].node_type
          self.attributes_kurz[sibling].actionability = self.attributes_kurz[siblings[0]].actionability
          self.attributes_kurz[sibling].mutability = self.attributes_kurz[siblings[0]].mutability
          self.attributes_kurz[sibling].parent_name_long = self.attributes_kurz[siblings[0]].parent_name_long
          self.attributes_kurz[sibling].parent_name_kurz = self.attributes_kurz[siblings[0]].parent_name_kurz
        else:
          raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

  def getSiblingsFor(self, attr_name_long_or_kurz):
    # If attr_name_long is given, we will return siblings_long (the same length)
    # but not siblings_kurz. Same for the opposite direction.
    assert \
      'cat' in attr_name_long_or_kurz or 'ord' in attr_name_long_or_kurz, \
      'attr_name must include either `cat` or `ord`.'
    if attr_name_long_or_kurz in self.getInputOutputAttributeNames('long'):
      attr_name_long = attr_name_long_or_kurz
      dict_of_siblings_long = self.getDictOfSiblings('long')
      for parent_name_long in dict_of_siblings_long['cat']:
        siblings_long = dict_of_siblings_long['cat'][parent_name_long]
        if attr_name_long_or_kurz in siblings_long:
          return siblings_long
      for parent_name_long in dict_of_siblings_long['ord']:
        siblings_long = dict_of_siblings_long['ord'][parent_name_long]
        if attr_name_long_or_kurz in siblings_long:
          return siblings_long
    elif attr_name_long_or_kurz in self.getInputOutputAttributeNames('kurz'):
      attr_name_kurz = attr_name_long_or_kurz
      dict_of_siblings_kurz = self.getDictOfSiblings('kurz')
      for parent_name_kurz in dict_of_siblings_kurz['cat']:
        siblings_kurz = dict_of_siblings_kurz['cat'][parent_name_kurz]
        if attr_name_long_or_kurz in siblings_kurz:
          return siblings_kurz
      for parent_name_kurz in dict_of_siblings_kurz['ord']:
        siblings_kurz = dict_of_siblings_kurz['ord'][parent_name_kurz]
        if attr_name_long_or_kurz in siblings_kurz:
          return siblings_kurz
    else:
      raise Exception(f'{attr_name_long_or_kurz} not recognized as a valid `attr_name_long_or_kurz`.')

  def getDictOfSiblings(self, long_or_kurz = 'kurz'):
    if long_or_kurz == 'long':

      dict_of_siblings_long = {}
      dict_of_siblings_long['cat'] = {}
      dict_of_siblings_long['ord'] = {}

      for attr_name_long in self.getInputAttributeNames('long'):
        attr_obj = self.attributes_long[attr_name_long]
        if attr_obj.attr_type == 'sub-categorical':
          if attr_obj.parent_name_long not in dict_of_siblings_long['cat'].keys():
            dict_of_siblings_long['cat'][attr_obj.parent_name_long] = [] # initiate key-value pair
          dict_of_siblings_long['cat'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
        elif attr_obj.attr_type == 'sub-ordinal':
          if attr_obj.parent_name_long not in dict_of_siblings_long['ord'].keys():
            dict_of_siblings_long['ord'][attr_obj.parent_name_long] = [] # initiate key-value pair
          dict_of_siblings_long['ord'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)

      # sort sub-arrays
      for key in dict_of_siblings_long['cat'].keys():
        dict_of_siblings_long['cat'][key] = sorted(dict_of_siblings_long['cat'][key], key = lambda x : int(x.split('_')[-1]))

      for key in dict_of_siblings_long['ord'].keys():
        dict_of_siblings_long['ord'][key] = sorted(dict_of_siblings_long['ord'][key], key = lambda x : int(x.split('_')[-1]))

      return dict_of_siblings_long

    elif long_or_kurz == 'kurz':

      dict_of_siblings_kurz = {}
      dict_of_siblings_kurz['cat'] = {}
      dict_of_siblings_kurz['ord'] = {}

      for attr_name_kurz in self.getInputAttributeNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        if attr_obj.attr_type == 'sub-categorical':
          if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['cat'].keys():
            dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
          dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
        elif attr_obj.attr_type == 'sub-ordinal':
          if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['ord'].keys():
            dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
          dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)

      # sort sub-arrays
      for key in dict_of_siblings_kurz['cat'].keys():
        dict_of_siblings_kurz['cat'][key] = sorted(dict_of_siblings_kurz['cat'][key], key = lambda x : int(x.split('_')[-1]))

      for key in dict_of_siblings_kurz['ord'].keys():
        dict_of_siblings_kurz['ord'][key] = sorted(dict_of_siblings_kurz['ord'][key], key = lambda x : int(x.split('_')[-1]))

      return dict_of_siblings_kurz

    else:

      raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

  def getOneHotAttributesNames(self, long_or_kurz = 'kurz'):
    tmp = self.getDictOfSiblings(long_or_kurz)
    names = []
    for key1 in tmp.keys():
      for key2 in tmp[key1].keys():
        names.extend(tmp[key1][key2])
    return np.array(names)

  def getNonHotAttributesNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getOneHotAttributesNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getVariableRanges(self):
    return dict(zip(
      self.getInputAttributeNames('kurz'),
      [
        self.attributes_kurz[attr_name_kurz].upper_bound -
        self.attributes_kurz[attr_name_kurz].lower_bound
        for attr_name_kurz in self.getInputAttributeNames('kurz')
      ],
    ))

  def printDataset(self, long_or_kurz = 'kurz'):
    if long_or_kurz == 'long':
      for attr_name_long in self.attributes_long:
        print(self.attributes_long[attr_name_long].__dict__)
    elif long_or_kurz == 'kurz':
      for attr_name_kurz in self.attributes_kurz:
        print(self.attributes_kurz[attr_name_kurz].__dict__)
    else:
      raise Exception(f'{long_or_kurz} not recognized as a valid `long_or_kurz`.')

  def getBalancedDataFrame(self):
    balanced_data_frame = copy.deepcopy(self.data_frame_kurz)

    meta_cols = self.getMetaAttributeNames()
    input_cols = self.getInputAttributeNames()
    output_col = self.getOutputAttributeNames()[0]

    # assert only two classes in label (maybe relax later??)
    assert np.array_equal(
      np.unique(balanced_data_frame[output_col]),
      np.array([0, 1]) # only allowing {0, 1} labels
    )

    # # get balanced dataframe (take minimum of the count, then round down to nearest 250)
    # unique_values_and_count = balanced_data_frame[output_col].value_counts()
    # # number_of_subsamples_in_each_class = unique_values_and_count.min() // 250 * 250
    # number_of_subsamples_in_each_class = unique_values_and_count.min() // 100 * 100
    # balanced_data_frame = pd.concat([
    #     balanced_data_frame[balanced_data_frame.loc[:,output_col] == 0].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
    #     balanced_data_frame[balanced_data_frame.loc[:,output_col] == 1].sample(number_of_subsamples_in_each_class, random_state = RANDOM_SEED),
    # ]).sample(frac = 1, random_state = RANDOM_SEED)
    # # balanced_data_frame = pd.concat([
    # #     balanced_data_frame[balanced_data_frame.loc[:,output_col] == 0],
    # #     balanced_data_frame[balanced_data_frame.loc[:,output_col] == 1],
    # # ]).sample(frac = 1, random_state = RANDOM_SEED)

    return balanced_data_frame, meta_cols, input_cols, output_col

  # (2020.04.15) perhaps we need a memoize here... but I tried calling this function
  # multiple times in a row from another file and it always returned the same slice
  # of data... weird.
  def getTrainTestSplit(self, preprocessing = None, with_meta = False):

    # When working only with normalized data in [0, 1], data ranges must change to [0, 1] as well
    # otherwise, in computing normalized distance we will normalize with intial ranges again!
    # pseudonym (2020.05.17) does this work with cat/ord and sub-cat/sub-ord data???
    def setBoundsToZeroOne():
      for attr_name_kurz in self.getNonHotAttributesNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        attr_obj.lower_bound = 0.0
        attr_obj.upper_bound = 1.0

        attr_obj = self.attributes_long[attr_obj.attr_name_long]
        attr_obj.lower_bound = 0.0
        attr_obj.upper_bound = 1.0

    # Normalize data: bring everything to [0, 1] - implemented for when feeding the model to DiCE
    def normalizeData(X_train, X_test):
      for attr_name_kurz in self.getNonHotAttributesNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        lower_bound = attr_obj.lower_bound
        upper_bound =attr_obj.upper_bound
        X_train[attr_name_kurz] = (X_train[attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)
        X_test[attr_name_kurz] = (X_test[attr_name_kurz] - lower_bound) / (upper_bound - lower_bound)

      setBoundsToZeroOne()

      return X_train, X_test

    # TODO: This should be used with caution... it messes things up in MACE as ranges
    # will differ between factual and counterfactual domains
    def standardizeData(X_train, X_test):
      x_mean = X_train.mean()
      x_std = X_train.std()
      for index in x_std.index:
        if '_ord_' in index or '_cat_' in index:
          x_mean[index] = 0
          x_std[index] = 1
      X_train = (X_train - x_mean) / x_std
      X_test = (X_test - x_mean) / x_std
      return X_train, X_test

    balanced_data_frame, meta_cols, input_cols, output_col = self.getBalancedDataFrame()

    if with_meta:
      all_data = balanced_data_frame.loc[:,np.array((input_cols, meta_cols)).flatten()]
      all_true_labels = balanced_data_frame.loc[:,output_col]
      if preprocessing is not None:
        assert with_meta == False, 'This feature is not built yet...'

      X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        all_true_labels,
        test_size=.2,
        random_state = RANDOM_SEED)

      # ordering of next two lines matters (shouldn't overwrite input_cols); silly code... :|
      U_train = X_train[self.getMetaAttributeNames()]
      U_test = X_test[self.getMetaAttributeNames()]
      X_train = X_train[self.getInputAttributeNames()]
      X_test = X_test[self.getInputAttributeNames()]
      y_train = y_train # noop
      y_test = y_test # noop

      return X_train, X_test, U_train, U_test, y_train, y_test
    else:
      all_data = balanced_data_frame.loc[:,input_cols]
      all_true_labels = balanced_data_frame.loc[:,output_col]

      X_train, X_test, y_train, y_test = train_test_split(
        all_data,
        all_true_labels,
        train_size=.7,
        random_state = RANDOM_SEED)

      # TODO (2020.05.18): this should be updated so as NOT to update meta variables
      if preprocessing == 'standardize':
        X_train, X_test = standardizeData(X_train, X_test)
      elif preprocessing == 'normalize':
        X_train, X_test = normalizeData(X_train, X_test)

      return X_train, X_test, y_train, y_test


class DatasetAttribute(object):

  def __init__(
    self,
    attr_name_long,
    attr_name_kurz,
    attr_type,
    node_type,
    actionability,
    mutability,
    parent_name_long,
    parent_name_kurz,
    lower_bound,
    upper_bound):

    if attr_type not in VALID_ATTRIBUTE_DATA_TYPES:
      raise Exception("`attr_type` must be one of %r." % VALID_ATTRIBUTE_DATA_TYPES)

    if node_type not in VALID_ATTRIBUTE_NODE_TYPES:
      raise Exception("`node_type` must be one of %r." % VALID_ATTRIBUTE_NODE_TYPES)

    if actionability not in VALID_ACTIONABILITY_TYPES:
      raise Exception("`actionability` must be one of %r." % VALID_ACTIONABILITY_TYPES)

    if mutability not in VALID_MUTABILITY_TYPES:
      raise Exception("`mutability` must be one of %r." % VALID_MUTABILITY_TYPES)

    if lower_bound > upper_bound:
      raise Exception("`lower_bound` must be <= `upper_bound`")

    if attr_type in {'sub-categorical', 'sub-ordinal'}:
      assert parent_name_long != -1, 'Parent ID set for non-hot attribute.'
      assert parent_name_kurz != -1, 'Parent ID set for non-hot attribute.'
      if attr_type == 'sub-categorical':
        assert lower_bound == 0
        assert upper_bound == 1
      if attr_type == 'sub-ordinal':
        # the first elem in thermometer is always on, but the rest may be on or off
        assert lower_bound == 0 or lower_bound == 1
        assert upper_bound == 1
    else:
      assert parent_name_long == -1, 'Parent ID set for non-hot attribute.'
      assert parent_name_kurz == -1, 'Parent ID set for non-hot attribute.'

    if attr_type in {'categorical', 'ordinal'}:
      assert lower_bound == 1 # setOneHotValue & setThermoValue assume this in their logic

    if attr_type in {'binary', 'categorical', 'sub-categorical'}: # not 'ordinal' or 'sub-ordinal'
      # IMPORTANT: surprisingly, it is OK if all sub-ordinal variables share actionability
      #            think about it, if each sub- variable is same-or-increase, along with
      #            the constraints that x0_ord_1 >= x0_ord_2, all variables can only stay
      #            the same or increase. It works :)
      assert actionability in {'none', 'any'}, f"{attr_type}'s actionability can only be in {'none', 'any'}, not `{actionability}`."

    if node_type != 'input':
      assert actionability == 'none', f'{node_type} attribute is not actionable.'
      assert mutability == False, f'{node_type} attribute is not mutable.'

    # We have introduced 3 types of variables: (actionable and mutable, non-actionable but mutable, immutable and non-actionable)
    if actionability != 'none':
      assert mutability == True
    # TODO: above/below seem contradictory... (2020.04.14)
    if mutability == False:
      assert actionability == 'none'

    if parent_name_long == -1 or parent_name_kurz == -1:
      assert parent_name_long == parent_name_kurz == -1

    self.attr_name_long = attr_name_long
    self.attr_name_kurz = attr_name_kurz
    self.attr_type = attr_type
    self.node_type = node_type
    self.actionability = actionability
    self.mutability = mutability
    self.parent_name_long = parent_name_long
    self.parent_name_kurz = parent_name_kurz
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound


def loadDataset(dataset_name, return_one_hot, load_from_cache = False, debug_flag = True, meta_param = None):

  def getInputOutputColumns(data_frame):
    all_data_frame_cols = data_frame.columns.values
    input_cols = [x for x in all_data_frame_cols if 'label' not in x.lower()]
    output_cols = [x for x in all_data_frame_cols if 'label' in x.lower()]
    assert len(output_cols) == 1
    return input_cols, output_cols[0]

  one_hot_string = 'one_hot' if return_one_hot else 'non_hot'

  save_file_path = os.path.join(
    os.path.dirname(__file__),
    f'_data_main/_cached/{dataset_name}_{one_hot_string}'
  )

  if load_from_cache:
    if debug_flag: print(f'[INFO] Attempting to load saved dataset (`{dataset_name}`) from cache...\t', end = '')
    try:
      tmp = pickle.load(open(save_file_path, 'rb'))
      if debug_flag: print('done.')
      return tmp
    except:
      if debug_flag: print('failed. Re-creating dataset...')

  if dataset_name == 'adult':

    data_frame_non_hot = load_adult_data_new()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'Sex':
        attr_type = 'binary'
        actionability = 'none'
        mutability = True
      elif col_name == 'Age':
        attr_type = 'numeric-int'
        actionability = 'same-or-increase' # 'none'
        mutability = True
      elif col_name == 'NativeCountry': #~ RACE
        attr_type = 'binary'
        actionability = 'any' # 'none'
        mutability = True
      elif col_name == 'WorkClass':
        attr_type = 'categorical'
        actionability = 'any'
        mutability = True
      elif col_name == 'EducationNumber':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'EducationLevel':
        attr_type = 'ordinal'
        actionability = 'any'
        mutability = True
      elif col_name == 'MaritalStatus':
        attr_type = 'categorical'
        actionability = 'any'
        mutability = True
      elif col_name == 'Occupation':
        attr_type = 'categorical'
        actionability = 'any'
        mutability = True
      elif col_name == 'Relationship':
        attr_type = 'categorical'
        actionability = 'any'
        mutability = True
      elif col_name == 'CapitalGain':
        attr_type = 'numeric-real'
        actionability = 'any'
        mutability = True
      elif col_name == 'CapitalLoss':
        attr_type = 'numeric-real'
        actionability = 'any'
        mutability = True
      elif col_name == 'HoursPerWeek':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'compass':

    data_frame_non_hot = load_compas_data_new()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'AgeGroup':
        attr_type = 'ordinal'
        actionability = 'same-or-increase' # 'none'
        mutability = True
      elif col_name == 'Race':
        attr_type = 'binary'
        actionability = 'any' # 'none'
        mutability = True
      elif col_name == 'Sex':
        attr_type = 'binary'
        actionability = 'none' # 'none'
        mutability = True
      elif col_name == 'PriorsCount':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'ChargeDegree':
        attr_type = 'binary'
        actionability = 'any'
        mutability = True

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'credit':

    data_frame_non_hot = load_credit_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'isMale':
        attr_type = 'binary'
        actionability = 'none'
        mutability = True
      elif col_name == 'isMarried':
        attr_type = 'binary'
        actionability = 'any'
        mutability = True
      elif col_name == 'AgeGroup':
        attr_type = 'ordinal'
        actionability = 'same-or-increase'
        mutability = True
      elif col_name == 'EducationLevel':
        attr_type = 'ordinal'
        actionability = 'any'
        mutability = True
      elif col_name == 'MaxBillAmountOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'MaxPaymentAmountOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'MonthsWithZeroBalanceOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'MonthsWithLowSpendingOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'MonthsWithHighSpendingOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'MostRecentBillAmount':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'MostRecentPaymentAmount':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'TotalOverdueCounts':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'TotalMonthsOverdue':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'HasHistoryOfOverduePayments':
        attr_type = 'binary'
        actionability = 'any'
        mutability = True

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'german':

    data_frame_non_hot = load_german_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'Sex':
        attr_type = 'binary'
        actionability = 'none'
        mutability = True
      elif col_name == 'Age':
        attr_type = 'numeric-int'
        actionability = 'same-or-increase'
        mutability = True
      elif col_name == 'Job':
        attr_type = 'ordinal'
        actionability = 'any'
        mutability = True
      elif col_name == 'Housing':
        attr_type = 'categorical'
        actionability = 'any'
        mutability = True
      elif col_name == 'SavingAccounts':
        attr_type = 'ordinal'
        actionability = 'any'
        mutability = True
      elif col_name == 'CheckingAccount':
        attr_type = 'ordinal'
        actionability = 'any'
        mutability = True
      elif col_name == 'CreditAmount':
        attr_type = 'numeric-real'
        actionability = 'any'
        mutability = True
      elif col_name == 'Duration':
        attr_type = 'numeric-int'
        actionability = 'any'
        mutability = True
      elif col_name == 'Purpose':
        attr_type = 'categorical'
        actionability = 'any'
        mutability = True

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'online':

    data_frame_non_hot = load_online_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      actionability = 'any'
      mutability = True

      if col_name == 'n_tokens_title': attr_type = 'numeric-int'
      elif col_name == 'n_tokens_content': attr_type = 'numeric-real'
      elif col_name == 'n_unique_tokens': attr_type = 'numeric-real'
      elif col_name == 'n_non_stop_words': attr_type = 'binary'
      elif col_name == 'n_non_stop_unique_tokens': attr_type = 'numeric-real'
      elif col_name == 'num_hrefs': attr_type = 'numeric-int'
      elif col_name == 'num_self_hrefs': attr_type = 'numeric-int'
      elif col_name == 'num_imgs': attr_type = 'numeric-int'
      elif col_name == 'num_videos': attr_type = 'numeric-int'
      elif col_name == 'average_token_length': attr_type = 'numeric-real'
      elif col_name == 'num_keywords': attr_type = 'numeric-int'
      elif col_name == 'Data_Channel': attr_type = 'categorical'
      elif col_name == 'kw_min_min': attr_type = 'numeric-real'
      elif col_name == 'kw_max_min': attr_type = 'numeric-real'
      elif col_name == 'kw_avg_min': attr_type = 'numeric-real'
      elif col_name == 'kw_min_max': attr_type = 'numeric-real'
      elif col_name == 'kw_max_max': attr_type = 'numeric-real'
      elif col_name == 'kw_avg_max': attr_type = 'numeric-real'
      elif col_name == 'kw_min_avg': attr_type = 'numeric-real'
      elif col_name == 'kw_max_avg': attr_type = 'numeric-real'
      elif col_name == 'kw_avg_avg': attr_type = 'numeric-real'
      elif col_name == 'self_reference_min_shares': attr_type = 'numeric-real'
      elif col_name == 'self_reference_max_shares': attr_type = 'numeric-real'
      elif col_name == 'self_reference_avg_sharess': attr_type = 'numeric-real'
      elif col_name == 'WeekDay': attr_type = 'categorical'
      elif col_name == 'is_weekend': attr_type = 'binary'
      elif col_name == 'LDA_00': attr_type = 'numeric-real'
      elif col_name == 'LDA_01': attr_type = 'numeric-real'
      elif col_name == 'LDA_02': attr_type = 'numeric-real'
      elif col_name == 'LDA_03': attr_type = 'numeric-real'
      elif col_name == 'LDA_04': attr_type = 'numeric-real'
      elif col_name == 'global_subjectivity': attr_type = 'numeric-real'
      elif col_name == 'global_sentiment_polarity': attr_type = 'numeric-real'
      elif col_name == 'global_rate_positive_words': attr_type = 'numeric-real'
      elif col_name == 'global_rate_negative_words': attr_type = 'numeric-real'
      elif col_name == 'rate_positive_words': attr_type = 'numeric-real'
      elif col_name == 'rate_negative_words': attr_type = 'numeric-real'
      elif col_name == 'avg_positive_polarity': attr_type = 'numeric-real'
      elif col_name == 'min_positive_polarity': attr_type = 'numeric-real'
      elif col_name == 'max_positive_polarity': attr_type = 'numeric-real'
      elif col_name == 'avg_negative_polarity': attr_type = 'numeric-real'
      elif col_name == 'min_negative_polarity': attr_type = 'numeric-real'
      elif col_name == 'max_negative_polarity': attr_type = 'numeric-real'
      elif col_name == 'title_subjectivity': attr_type = 'numeric-real'
      elif col_name == 'title_sentiment_polarity': attr_type = 'numeric-real'
      elif col_name == 'abs_title_subjectivity': attr_type = 'numeric-real'
      elif col_name == 'abs_title_sentiment_polarity': attr_type = 'numeric-real'

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'phishing':

    data_frame_non_hot = load_phishing_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      actionability = 'any'
      mutability = True
      
      if col_name == 'havingIPAddress': attr_type = 'binary'
      elif col_name == 'URLLength': attr_type = 'ordinal'
      elif col_name == 'ShortiningService': attr_type = 'binary'
      elif col_name == 'havingAtSymbol': attr_type = 'binary'
      elif col_name == 'doubleslashredirecting': attr_type = 'binary'
      elif col_name == 'PrefixSuffix': attr_type = 'binary'
      elif col_name == 'havingSubDomain': attr_type = 'ordinal'
      elif col_name == 'SSLfinalState': attr_type = 'ordinal'
      elif col_name == 'Domainregisterationlength': attr_type = 'binary'
      elif col_name == 'Favicon': attr_type = 'binary'
      elif col_name == 'port': attr_type = 'binary'
      elif col_name == 'HTTPStoken': attr_type = 'binary'
      elif col_name == 'RequestURL': attr_type = 'binary'
      elif col_name == 'URLofAnchor': attr_type = 'ordinal'
      elif col_name == 'Linksintags': attr_type = 'ordinal'
      elif col_name == 'SFH': attr_type = 'ordinal'
      elif col_name == 'Submittingtoemail': attr_type = 'binary'
      elif col_name == 'AbnormalURL': attr_type = 'binary'
      elif col_name == 'Redirect': attr_type = 'binary'
      elif col_name == 'onmouseover': attr_type = 'binary'
      elif col_name == 'RightClick': attr_type = 'binary'
      elif col_name == 'popUpWidnow': attr_type = 'binary'
      elif col_name == 'Iframe': attr_type = 'binary'
      elif col_name == 'ageofdomain': attr_type = 'binary'
      elif col_name == 'DNSRecord': attr_type = 'binary'
      elif col_name == 'webtraffic': attr_type = 'ordinal'
      elif col_name == 'PageRank': attr_type = 'binary'
      elif col_name == 'GoogleIndex': attr_type = 'binary'
      elif col_name == 'Linkspointingtopage': attr_type = 'ordinal'
      elif col_name == 'Statisticalreport': attr_type = 'binary'

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'spambase':

    data_frame_non_hot = load_spambase_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      actionability = 'any'
      mutability = True
      attr_type = 'numeric-real'
      
      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'students':

    data_frame_non_hot = load_students_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      node_type = 'output',
      actionability = 'none',
      mutability = False,
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      actionability = 'any'
      mutability = True

      if col_name == 'school': attr_type = 'binary'
      elif col_name == 'sex': 
        attr_type = 'binary'
        actionability = 'none'
      elif col_name == 'age': 
        attr_type = 'numeric-int'
        actionability = 'same-or-increase'
      elif col_name == 'address': attr_type = 'binary'
      elif col_name == 'famsize': attr_type = 'binary'
      elif col_name == 'Pstatus': attr_type = 'binary'
      elif col_name == 'Medu': attr_type = 'ordinal'
      elif col_name == 'Fedu': attr_type = 'ordinal'
      elif col_name == 'Mjob': attr_type = 'categorical'
      elif col_name == 'Fjob': attr_type = 'categorical'
      elif col_name == 'reason': attr_type = 'categorical'
      elif col_name == 'guardian': attr_type = 'categorical'
      elif col_name == 'traveltime': attr_type = 'ordinal'
      elif col_name == 'studytime': attr_type = 'ordinal'
      elif col_name == 'failures': attr_type = 'numeric-int'
      elif col_name == 'schoolsup': attr_type = 'binary'
      elif col_name == 'famsup': attr_type = 'binary'
      elif col_name == 'paid': attr_type = 'binary'
      elif col_name == 'activities': attr_type = 'binary'
      elif col_name == 'nursery': attr_type = 'binary'
      elif col_name == 'higher': attr_type = 'binary'
      elif col_name == 'internet': attr_type = 'binary'
      elif col_name == 'romantic': attr_type = 'binary'
      elif col_name == 'famrel': attr_type = 'ordinal'
      elif col_name == 'freetime': attr_type = 'ordinal'
      elif col_name == 'goout': attr_type = 'ordinal'
      elif col_name == 'Dalc': attr_type = 'ordinal'
      elif col_name == 'Walc': attr_type = 'ordinal'
      elif col_name == 'health': attr_type = 'ordinal'
      elif col_name == 'absences': attr_type = 'numeric-int'

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = f'x{col_idx}',
        attr_type = attr_type,
        node_type = 'input',
        actionability = actionability,
        mutability = mutability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  else:

    raise Exception(f'{dataset_name} not recognized as a valid dataset.')

  if return_one_hot:
    data_frame, attributes = getOneHotEquivalent(data_frame_non_hot, attributes_non_hot)
  else:
    data_frame, attributes = data_frame_non_hot, attributes_non_hot

  # save then return
  dataset_obj = Dataset(data_frame, attributes, return_one_hot, dataset_name)
  # if not loading from cache, we always overwrite the cache
  pickle.dump(dataset_obj, open(save_file_path, 'wb'))
  return dataset_obj


# TODO: consider moving into Dataset class with getOneHot and getNonHot methods
def getOneHotEquivalent(data_frame_non_hot, attributes_non_hot):

  # TODO: see how we can switch between feature_names = col names for kurz and long (also maybe ordered)

  data_frame = copy.deepcopy(data_frame_non_hot)
  attributes = copy.deepcopy(attributes_non_hot)

  def setOneHotValue(val):
    return np.append(np.append(
      np.zeros(val - 1),
      np.ones(1)),
      np.zeros(num_unique_values - val)
    )

  def setThermoValue(val):
    return np.append(
      np.ones(val),
      np.zeros(num_unique_values - val)
    )

  for col_name in data_frame.columns.values:

    if attributes[col_name].attr_type not in {'categorical', 'ordinal'}:
      continue

    old_col_name_long = col_name
    new_col_names_long = []
    new_col_names_kurz = []

    old_attr_name_long = attributes[old_col_name_long].attr_name_long
    old_attr_name_kurz = attributes[old_col_name_long].attr_name_kurz
    old_attr_type = attributes[old_col_name_long].attr_type
    old_node_type = attributes[old_col_name_long].node_type
    old_actionability = attributes[old_col_name_long].actionability
    old_mutability = attributes[old_col_name_long].mutability
    old_lower_bound = attributes[old_col_name_long].lower_bound
    old_upper_bound = attributes[old_col_name_long].upper_bound

    num_unique_values = int(old_upper_bound - old_lower_bound + 1)

    assert old_col_name_long == old_attr_name_long

    new_attr_type = 'sub-' + old_attr_type
    new_node_type = old_node_type
    new_actionability = old_actionability
    new_mutability = old_mutability
    new_parent_name_long = old_attr_name_long
    new_parent_name_kurz = old_attr_name_kurz


    if attributes[col_name].attr_type == 'categorical': # do not do this for 'binary'!

      new_col_names_long = [f'{old_attr_name_long}_cat_{i}' for i in range(num_unique_values)]
      new_col_names_kurz = [f'{old_attr_name_kurz}_cat_{i}' for i in range(num_unique_values)]
      print(f'Replacing column {col_name} with {{{", ".join(new_col_names_long)}}}')
      tmp = np.array(list(map(setOneHotValue, list(data_frame[col_name].astype(int).values))))
      data_frame_dummies = pd.DataFrame(data=tmp, columns=new_col_names_long)

    elif attributes[col_name].attr_type == 'ordinal':

      new_col_names_long = [f'{old_attr_name_long}_ord_{i}' for i in range(num_unique_values)]
      new_col_names_kurz = [f'{old_attr_name_kurz}_ord_{i}' for i in range(num_unique_values)]
      print(f'Replacing column {col_name} with {{{", ".join(new_col_names_long)}}}')
      tmp = np.array(list(map(setThermoValue, list(data_frame[col_name].astype(int).values))))
      data_frame_dummies = pd.DataFrame(data=tmp, columns=new_col_names_long)

    # Update data_frame
    data_frame = pd.concat([data_frame.drop(columns = old_col_name_long), data_frame_dummies], axis=1)

    # Update attributes
    del attributes[old_col_name_long]
    for col_idx in range(len(new_col_names_long)):
      new_col_name_long = new_col_names_long[col_idx]
      new_col_name_kurz = new_col_names_kurz[col_idx]
      attributes[new_col_name_long] = DatasetAttribute(
        attr_name_long = new_col_name_long,
        attr_name_kurz = new_col_name_kurz,
        attr_type = new_attr_type,
        node_type = new_node_type,
        actionability = new_actionability,
        mutability = new_mutability,
        parent_name_long = new_parent_name_long,
        parent_name_kurz = new_parent_name_kurz,
        lower_bound = data_frame[new_col_name_long].min(),
        upper_bound = data_frame[new_col_name_long].max())

  return data_frame, attributes

