import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_dataset(path, model_type):

  df = pd.read_csv(path)

  features = df.columns.to_list()[1:]

  data = df.loc[:, df.columns != 'SleepStage'].values
  target = df['SleepStage'].values

  X_train, X_rem, y_train, y_rem = train_test_split(data, target, train_size=0.8)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  unique, counts = np.unique(y_train, return_counts=True)
  D = dict(zip(unique, counts))
  D[1] = int(0.4*max(D.values()))

  smot = SMOTE(D, random_state=42)
  X_train, y_train = smot.fit_resample(X_train, y_train)

  if model_type == 'MLP':
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    valid_dataset = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
  elif model_type == 'CNN':
    train_dataset = TensorDataset(torch.from_numpy(X_train.astype('float32')).unsqueeze(1), torch.from_numpy(y_train.astype('long')))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid.astype('float32')).unsqueeze(1), torch.from_numpy(y_valid.astype('long')))
    test_dataset = TensorDataset(torch.from_numpy(X_test.astype('float32')).unsqueeze(1), torch.from_numpy(y_test.astype('long')))
  else:
    raise AssertionError("Wrong Model Type!")

  return train_dataset, valid_dataset, test_dataset, features, list(D.values())
