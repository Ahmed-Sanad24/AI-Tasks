# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:32:32 2021

@author: workstation
"""
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wuzzuf_Jobs.csv')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# (1)  factorize 

dataset['fact_YearsExp'] = pd.factorize(dataset['YearsExp'])[0]



