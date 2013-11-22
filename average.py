import pandas as pd
import numpy as np

data = pd.read_csv('submission/submission_lr.csv')
temp = pd.read_csv('submission/submission_nb.csv')
data['NB'] = temp['ACTION']
temp = pd.read_csv('submission/submission_svm.csv')
data['SVM'] = temp['ACTION']