#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sp
import seaborn as sns
import itertools


# In[2]:


df=pd.read_csv('loan_data.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# # check the missing values

# In[8]:


df.isnull().sum()


# from the above data we can say that credit history and loanamnt terms are very importatnt which contains null values.

# In[9]:


df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())#filling null values with its mean


# In[10]:


df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median())#filling null values with its median


# In[11]:


df.isnull().sum()#confirming we get the null free values for our importatnt terms


# In[12]:


df.dropna(inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df.shape


# In[15]:


plt.figure(figsize=(100,50))  #figuresize define
sns.set(font_scale=2.5) #fontsize 
plt.subplot(331)
sns.countplot(df['Gender'],hue=df['Loan_Status'])

plt.subplot(332)
sns.countplot(df['Married'],hue=df['Loan_Status'])

plt.subplot(333)
sns.countplot(df['Dependents'],hue=df['Loan_Status'])

plt.subplot(334)
sns.countplot(df['Self_Employed'],hue=df['Loan_Status'])

plt.subplot(335)
sns.countplot(df['Loan_Amount_Term'],hue=df['Loan_Status'])


# # to covert variable data into numerical form

# In[16]:


df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)


# In[17]:


df['Married'].replace('Yes',1,inplace=True)
df['Married'].replace('No',0,inplace=True)


# In[18]:


df['Loan_Status'].value_counts()


# In[19]:


df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()


# In[20]:


df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependents'].value_counts()


# In[21]:


df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()


# In[22]:


df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()


# In[23]:


df.Property_Area=df.Property_Area.map({'Rural':0,'Semiurban':1,'Urban':2})
df['Property_Area'].value_counts()


# In[24]:


df['LoanAmount'].value_counts()


# In[25]:


df['Loan_Amount_Term'].value_counts()


# In[26]:


df['Credit_History'].value_counts()


# In[27]:


df.head()


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[29]:


#cols = ['Loan_ID']
#df.drop(cols, axis=1, inplace=True)


# In[30]:


df


# In[31]:


x=df.iloc[1:542,1:12].values
y=df.iloc[1:542,12].values


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# In[45]:


model=LogisticRegression()


# In[46]:


model.fit(x_train,y_train)


# In[47]:


d_prediction=model.predict(x_test)


# In[48]:


print('Logistic Regression Accuracy=',metrics.accuracy_score(d_prediction,y_test))

