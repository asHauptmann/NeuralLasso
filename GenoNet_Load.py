# Learned lasso-type sparse network for genetic data
# Andreas Hauptmann, University of Oulu/UCL, 2022

import numpy as np



# Create the dataset for tensorboard with logging of epochs and separation into train and test data
class DataSet(object):

  def __init__(self, geno, pheno):
    """Construct a DataSet"""

    assert geno.shape[0] == pheno.shape[0], (
        'geno.shape: %s labels.shape: %s' % (geno.shape,
                                                 pheno.shape))
    self._num_examples = geno.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
#    assert geno.shape[3] == 1
    geno = geno.reshape(geno.shape[0],
                            geno.shape[1])
    pheno = pheno.reshape(pheno.shape[0],
                            pheno.shape[1])
    

    self._geno = geno
    self._pheno = pheno
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def geno(self):
    return self._geno

  @property
  def pheno(self):
    return self._pheno

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._geno = self._geno[perm]
      self._pheno = self._pheno[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._geno[start:end], self._pheno[start:end]


# Main call to create the dataset for tensorflow
def read_data_sets(X_train, X_test, y_train, y_test):
  class DataSets(object):
    pass
  data_sets = DataSets()
  print('Start loading data') 
  
  data_sets.train = DataSet(X_train, y_train)
  data_sets.test = DataSet(X_test, y_test)

  return data_sets


