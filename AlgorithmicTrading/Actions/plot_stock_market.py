print(__doc__)

import datetime
import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
from matplotlib import finance
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

#df_training = pd.read_csv('../csv/training.csv')
#df_testing  = pd.read_csv('../csv/testing.csv')
#print "df_training=",df_training.describe()
#print "df_testing =",df_testing.describe()

df_entry_linear = pd.read_csv('../csv/example_entry_linear.csv')
