#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing, model_selection, metrics
import warnings
warnings.filterwarnings("ignore")
import os


# In[2]:


data = pd.read_csv("C:/Prathamesh/Semester 6/IOT/Project/New_IOT_Data.csv", encoding= 'unicode_escape')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


print('The number of rows in this dataset is - ', data.shape[0])
print('The number of columns in this dataset is- ', data.shape[1])


# In[7]:


#Number of null values in all columns
data.isnull().sum().sort_values(ascending = True)


# In[8]:


keep_col = ['Date','Time','Time.1','Total CPU Usage [%]', 'CPU SOC [°C]','Core+SoC Power (SVI2 TFN) [W]','GPU Memory Usage [%]','GPU Temperature [°C]','GPU Power [W]']
new_data = data[keep_col]
new_data.to_csv("newdata_IOT.csv", index = False)


# In[9]:


data = pd.read_csv("newdata_IOT.csv")


# In[10]:


data.head()


# In[11]:


path = 'C:/Prathamesh/Semester 6/IOT/Project'
new_data.to_csv(path + 'newdata_IOT.csv')


# In[12]:


data = pd.read_csv("C:/Prathamesh/Semester 6/IOT/Project/newdata_IOT.csv")


# In[13]:


data.head()


# In[14]:


#Number of null values in all columns
data.isnull().sum().sort_values(ascending = True)


# In[15]:


from sklearn.model_selection import train_test_split

#for our dataset, 75% of the datta is used for training the models and the rest is used for testing
train, test = train_test_split(data,test_size=0.25,random_state=40)


# In[16]:


train.describe()


# In[17]:


total_power = data['Core+SoC Power (SVI2 TFN) [W]'] + data['GPU Power [W]']
data = data.assign(total_power = total_power)


# In[18]:


data.head()


# In[19]:


feature_vars1 = train[["Total CPU Usage [%]"]+["CPU SOC [°C]"]]
target_vars1 = train[["Core+SoC Power (SVI2 TFN) [W]"]]
feature_vars2 = train[["GPU Memory Usage [%]"]+["GPU Temperature [°C]"]]
target_vars2 = train[["GPU Power [W]"]]


# In[20]:


feature_vars1.describe()


# In[21]:


target_vars1.describe()


# In[22]:


feature_vars2.describe()


# In[23]:


target_vars2.describe()


# CPU Related Analysis

# In[24]:


#Splitting the training dataset into independent and dependent variables
train_X1 = train[feature_vars1.columns]
train_y1 = train[target_vars1.columns]


# In[25]:


#Splitting the test dataset into independent and dependent variables
test_X1 = test[feature_vars1.columns]
tesy_y1 = test[target_vars1.columns]


# In[26]:


train_X1.columns


# In[27]:


test_X1.columns


# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Creating the test and training set by including the Core+SOC power column

train1 = train[list(train_X1.columns.values) + ["Core+SoC Power (SVI2 TFN) [W]"]]
test1 = test[list(test_X1.columns.values) + ["Core+SoC Power (SVI2 TFN) [W]"]]

#Creating dummy test and training set to hold scaled values

sc_train1 = pd.DataFrame(columns = train1.columns, index = train1.index)
sc_train1[sc_train1.columns] = sc.fit_transform(train1)

sc_test1 = pd.DataFrame(columns = test1.columns, index = test1.index)
sc_test1[sc_test1.columns] = sc.fit_transform(test1)


# In[29]:


sc_train1.head()


# In[30]:


sc_test1.head()


# In[31]:


# Removing Core+Soc power column from the training set

train_X1 = sc_train1.drop(["Core+SoC Power (SVI2 TFN) [W]"], axis=1)
train_y1 = sc_train1["Core+SoC Power (SVI2 TFN) [W]"]

test_X1 = sc_test1.drop(["Core+SoC Power (SVI2 TFN) [W]"], axis=1)
test_y1 = sc_test1["Core+SoC Power (SVI2 TFN) [W]"]


# In[32]:


train_X1.head()


# In[33]:


train_y1.head()


# In[34]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[35]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn import neighbors
from sklearn.svm import SVR


# In[36]:


models = [
    ['Lasso: ', Lasso()],
    ['Ridge: ', Ridge()],
    ['KNeighborsRegressor: ', neighbors.KNeighborsRegressor()],
    ['SVR: ', SVR(kernel= 'rbf')],
    ['RandomForest: ', RandomForestRegressor()],
    ['ExtraTreeRegressor: ', ExtraTreesRegressor()],
    ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
    ['MLPRegressor: ', MLPRegressor(activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000, learning_rate_init=0.01, alpha=0.01)]
]


# In[37]:


# Run all the proposed models and update the information in a list model_data
import time
from math import sqrt
from sklearn.metrics import mean_squared_error

model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(train_X1,train_y1)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y1,curr_model.predict(train_X1))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y1,curr_model.predict(test_X1))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y1,curr_model.predict(test_X1)))
    model_data.append(curr_model_data)


# In[38]:


model_data


# In[39]:


#Converting this list to dataframe
df1 = pd.DataFrame(model_data)


# In[40]:


df1


# In[41]:


df1.plot(x="Name", y=['Test_R2_Score', 'Train_R2_Score', 'Test_RMSE_Score'], kind="bar", title="R2 Score Results for CPU Analysis", figsize = (10,8));


# 1. The best results over the test set were given by GradientBoostingClassifier with R2 score of 0.968674	
# 2. Least RMSE Score is also given by GradientBoostingClassifier: 0.176992
# 3. Lasso Regularization over Linear Regression was the worst performing model

# GPU Related Analysis

# In[42]:


#Splitting the training dataset into independent and dependent variables
train_X2 = train[feature_vars2.columns]
train_y2 = train[target_vars2.columns]


# In[43]:


#Splitting the test dataset into independent and dependent variables
test_X2 = test[feature_vars2.columns]
tesy_y2 = test[target_vars2.columns]


# In[44]:


train_X2.columns


# In[45]:


test_X2.columns


# In[46]:


# Creating the test and training set by including the GPU power column

train2 = train[list(train_X2.columns.values) + ["GPU Power [W]"]]
test2 = test[list(test_X2.columns.values) + ["GPU Power [W]"]]

#Creating dummy test and training set to hold scaled values

sc_train2 = pd.DataFrame(columns = train2.columns, index = train2.index)
sc_train2[sc_train2.columns] = sc.fit_transform(train2)

sc_test2 = pd.DataFrame(columns = test2.columns, index = test2.index)
sc_test2[sc_test2.columns] = sc.fit_transform(test2)


# In[47]:


sc_train2.head()


# In[48]:


sc_test2.head()


# In[49]:


# Removing GPU power column from the training set

train_X2 = sc_train2.drop(["GPU Power [W]"], axis=1)
train_y2 = sc_train2["GPU Power [W]"]

test_X2 = sc_test2.drop(["GPU Power [W]"], axis=1)
test_y2 = sc_test2["GPU Power [W]"]


# In[50]:


train_X2.head()


# In[51]:


train_y2.head()


# In[52]:


# Run all the proposed models and update the information in a list model_data
model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(train_X2,train_y2)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y2,curr_model.predict(train_X2))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y2,curr_model.predict(test_X2))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y2,curr_model.predict(test_X2)))
    model_data.append(curr_model_data)


# In[53]:


model_data


# In[54]:


#Converting this list to dataframe
df2 = pd.DataFrame(model_data)


# In[55]:


df2


# In[56]:


df2.plot(x="Name", y=['Test_R2_Score', 'Train_R2_Score', 'Test_RMSE_Score'], kind="bar", title="R2 Score Results for GPU Analysis", figsize = (10,8));


# 1. The best results over the test set were given by ExtraTreeRegressor with R2 score of 0.993489	
# 2. Least RMSE Score is also given by ExtraTreeRegressor: 0.080691
# 3. Lasso Regularization over Linear Regression was the worst performing model

# Hence while training the ml model on CPU data, we will use the GradientBoostingClassifier algorithm.
# And while training the ml model on GPU data, we will use the ExtraTreeRegressor algorithm.

# In[ ]:




