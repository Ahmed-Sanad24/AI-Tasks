#!/usr/bin/env python
# coding: utf-8

# In[53]:


#importing libraries
import pandas as pd
import numpy as np


# In[54]:


#importing data file 
dataset = pd.read_csv("diabetes.csv")


# In[55]:


#showing the data
dataset.head()


# In[56]:


dbts_feats = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction","Age"]
X = dataset[dbts_feats]
dbts_op = ["Outcome"]
y = dataset[dbts_op]
# showing all coulomns except outcome 
X.head()


# In[57]:


# outcome coulomn
y.head()


# split the data into 4 categories

# In[58]:


#splitting the data to train and test , x and y 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.2)


# In[59]:


#handling the missing values using imputer
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns


# In[62]:


# **Random forest model**
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_forest = forest_model.predict(imputed_X_test)
random_forest_error = mean_absolute_error(y_test, X_validation_forest)
print(random_forest_error)


# In[63]:


# **Logistic regression model**
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter= 180)
logistic_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_logistic = logistic_model.predict(imputed_X_test)
logistic_error = mean_absolute_error(y_test, X_validation_logistic)
print(logistic_error)


# In[60]:


# fit the **decision tree model** with the data 
# then find the the mean absolute error

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
dbts_mod = DecisionTreeRegressor()
dbts_mod.fit(imputed_X_train, y_train)
X_validation = dbts_mod.predict(imputed_X_test)
decision_tree_error = mean_absolute_error( y_test ,X_validation)
print(decision_tree_error)


# In[64]:


# **KNN model**
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

#feature scaling using standard scaler 
X_scale = StandardScaler()
imputed_X_train = X_scale.fit_transform(imputed_X_train)
imputed_X_test = X_scale.transform(imputed_X_test)


# In[65]:


# using matrices to understand data
from scipy.sparse.sputils import matrix
KNN_model = KNeighborsClassifier(n_neighbors= 27, p=2, metric='euclidean')
KNN_model.fit(imputed_X_train, y_train.values.ravel())
X_validation_KNN = KNN_model.predict(imputed_X_test)


# In[66]:


# confusion matrix to show the errors
cm = confusion_matrix(y_test, X_validation_KNN)
print(cm)


# In[67]:


print(f"the f1 score for KNN method is {f1_score(y_test, X_validation_KNN)}")
print(f"the accuracy for KNN method is {accuracy_score(y_test, X_validation_KNN)} ")
print(f"the f1 score for decision tree method is {f1_score(y_test, X_validation)}")
print(f"the accuracy for decision tree method is {accuracy_score(y_test, X_validation)} ")
print(f"the f1 score for logistic regression method is {f1_score(y_test, X_validation_logistic)}")
print(f"the accuracy for logistic regression method is {accuracy_score(y_test, X_validation_logistic)} ")

