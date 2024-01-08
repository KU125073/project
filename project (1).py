#!/usr/bin/env python
# coding: utf-8

# In[19]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[20]:


df = pd.read_csv('1.csv')
df.head()


# In[21]:


df.info()


# In[22]:


obj_col = df.columns
for col in obj_col:
    print("{}: {}".format(col,len(df[col].unique())))


# In[23]:


df.isnull().sum()


# In[24]:


df = df.drop(columns=['Invoice ID','gross margin percentage'])
df.head()


# In[25]:


df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = pd.to_datetime(df['Date'])

df['Hour'] = df['Time'].dt.hour
#df['Minute'] = df['Time'].dt.minute

df['Day'] = (df['Date']).dt.day
df['Month'] = (df['Date']).dt.month

df = df.drop(columns=['Time','Date'])

df.head()


# In[26]:


mean_q = df['Quantity'].mean()
median_q = df['Quantity'].median()

plt.figure(figsize=(10,8))
sns.kdeplot(data=df,x='Quantity')
plt.axvline(mean_q,c='r',linestyle='--')
plt.axvline(median_q,c='g',linestyle=':')


# In[27]:


df.describe()


# In[28]:


def check_outliers(df):
    num_col = df.select_dtypes(['float64','int64']).columns
    nCols = len(num_col)//3+1

    fig, ax = plt.subplots(nCols,3,figsize=(30,8*nCols))
    for i, col in enumerate(num_col):
        sns.boxplot(data=df,x=col,ax=ax[i//3][i%3])

    for i in range(len(num_col),3*nCols):
        fig.delaxes(ax[i//3][i%3])

    plt.show()
    
check_outliers(df)


# ### Remove outliers
# From the graphs above, we can see that Tax 5%, cogs, Total and gross income have outliers. So now I will remove outlier. Regarding to an attribute, if a value of it is out size 1.5 times IQR from mean, it will be treated as outliers.

# In[29]:


def remove_outlier(df, col_name):
    q = df[col_name].quantile([0.25,0.5,0.75])
    IQR = q[0.75]-q[0.25]
    lower = q[0.5]-1.5*IQR
    upper = q[0.5]+1.5*IQR
    return df[(df[col_name]>=lower) & (df[col_name]<=upper)]

df = remove_outlier(df,'Tax 5%')
df = remove_outlier(df,'cogs')
df = remove_outlier(df,'Total')
df = remove_outlier(df,'gross income')
check_outliers(df)


# ### Pearson's correlation heatmap
# By using pearson correlation, you can see the "linear" correlation between two attributes. By this way, we can remove the attribute that cause multicollinearity which is bad for modelling.

# In[30]:


corr = df.corr(method='pearson')
plt.figure(figsize=(15,12))
sns.heatmap(corr,annot=True,cmap='YlGnBu')
plt.show()


# In[31]:


plt.figure(figsize=(10,8))
sns.pairplot(data=df,vars=['Total','cogs','gross income','Tax 5%'])
plt.show()


# From the heatmap above, we can see that the Total, cogs, Tax 5%, gross income are perfectly correlated. This is very considerable, so let drop 3 of 4 attributes above (I choose cogs, gross income and Tax 5%).

# In[32]:


df = df.drop(columns=['cogs','gross income','Tax 5%'])
df.head()


# ### Relationship of cities and branches
# However, Pearson's correlation only works for continous value. For the category attribute, we can predict and check their correlation by scatter plot. For instance, let check the relationship between city and branch.

# In[33]:


plt.figure(figsize=(10,8))
sns.scatterplot(data=df,x='City',y='Branch')
plt.show()


# As a result, there is also a relationship between city and branch. Each branch is located at a specific city. So, we can also remove 1 of those two (I remove City in this case).

# In[34]:


df = df.drop(columns="City")
df.head()


# In[35]:


df.describe()


# In[36]:


y = df['Quantity']
X = df.drop(columns="Quantity")

#num_col = X.select_dtypes(['float64','int64']).columns
cat_col = X.select_dtypes(['object']).columns

X.head()


# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[38]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[39]:


ohe = OneHotEncoder(drop = 'first',sparse=False)
X_train_cat = ohe.fit_transform(X_train[['Branch','Customer type','Gender','Product line','Payment']])

# allso apply for X_test
X_test_cat = ohe.transform(X_test[['Branch','Customer type','Gender','Product line','Payment']])


# In[40]:


X_train_cat


# In[41]:


# Extracting 
X_train_age = X_train.drop(columns = ['Branch','Customer type','Gender','Product line','Payment'],axis=1).values

# also form test data
X_test_age = X_test.drop(columns = ['Branch','Customer type','Gender','Product line','Payment']).values


# In[42]:


X_train_age


# In[43]:


import numpy as np
X_train_transformed = np.concatenate((X_train_age,X_train_cat),axis=1)

# also for test data
X_test_transformed = np.concatenate((X_test_age,X_test_cat),axis=1)


# In[44]:


X_train_transformed.shape


# In[45]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train_transformed)
X_test = sc.transform(X_test_transformed)


# In[46]:


X_train.shape,X_test.shape


# In[47]:


from sklearn.linear_model import SGDRegressor, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error


# In[48]:


model2 = LinearRegression()
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
print("Linera Regression Score:",model2.score(X_test,y_test))
MSE2= mean_squared_error(y_test, y_pred)
print("MSE:",MSE2)


# In[49]:


model4 = Ridge()
model4.fit(X_train,y_train)
y_pred=model4.predict(X_test)
print("Ridge Model Score:",model4.score(X_test,y_test))
MSE4= mean_squared_error(y_test, y_pred)
print("MSE :",MSE4)


# In[50]:


from sklearn import svm
model5 = svm.SVR(gamma='auto',kernel='linear')
model5.fit(X_train,y_train)
y_pred=model5.predict(X_test)
print("svm Model Score:",model5.score(X_test,y_test))
MSE5= mean_squared_error(y_test, y_pred)
print("MSE :",MSE5)


# In[51]:


from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor(random_state=42,n_estimators=400)
model7 = RandomForestRegressor()
model7.fit(X_train,y_train)
y_pred=model7.predict(X_test)
print("RF Model Score:",model7.score(X_test,y_test))
MSE7= mean_squared_error(y_test, y_pred)
print("MSE :",MSE7)


# In[52]:


from xgboost import XGBRFRegressor
model8 = XGBRFRegressor(random_state=42,n_estimators=400)
model8.fit(X_train,y_train)
y_pred=model8.predict(X_test)
print("XGB Model Score:",model8.score(X_test,y_test))
MSE8= mean_squared_error(y_test, y_pred)
print("XGB MSE :",MSE8)


# In[53]:


model_names = ['Linear_reg','Ridge','svm_linear','RF_Model', 'XGB_Model']
accuracies = [model2.score(X_test,y_test)*100, 
              model4.score(X_test,y_test)*100, 
              model5.score(X_test,y_test)*100,
              model7.score(X_test,y_test)*100,
              model8.score(X_test,y_test)*100]


# In[54]:


from matplotlib.cm import rainbow

plt.figure(figsize = (16, 10))
colors = rainbow(np.linspace(0, 1),len(model_names))
barplot = plt.bar(model_names,accuracies,color =['red','yellow','blue','turquoise','purple'])
plt.xticks(fontsize = 12)
#plt.xticks(rotation=45)
plt.xlabel("Classifiers", fontsize = 14)
plt.ylabel("Accuracy", fontsize = 12)
plt.title("Accuracy Scores", fontsize = 16)
for i, bar in enumerate(barplot):
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, 
             bar.get_height()*1.02, 
             s = '{:.2f}%'.format(accuracies[i]), 
             fontsize = 15)


# In[55]:


model_names = ['Linear_reg','Ridge','svm_linear','RF_Mode', 'XGB_Model']
MSE = [ 
              MSE2,MSE4,MSE5,MSE7,MSE8]
'''
              model2.score(X_test,y_test)*100, 
              model4.score(X_test,y_test)*100, 
              model5.score(X_test,y_test)*100,
              model7.score(X_test,y_test)*100,
              model8.score(X_test,y_test)*100]
              '''


# In[56]:


from matplotlib.cm import rainbow

plt.figure(figsize = (16, 10))
colors = rainbow(np.linspace(0, 1),len(MSE))
barplot = plt.bar(model_names,MSE,color =['red','yellow','blue','turquoise','purple'])
plt.xticks(fontsize = 12)
#plt.xticks(rotation=45)
plt.xlabel("Regressor", fontsize = 14)
plt.ylabel("Accuracy", fontsize = 12)
plt.title("Mean Square Erroer", fontsize = 16)
for i, bar in enumerate(barplot):
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, 
             bar.get_height()*1.02, 
             s = '{:.2f}%'.format(MSE[i]), 
             fontsize = 15)


# In[ ]:




