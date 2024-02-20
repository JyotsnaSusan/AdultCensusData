#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install ucimlrepo


# In[2]:


from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


pip install streamlit


# **1. Reading Data**
# 
# 
# 

# In[7]:


df=adult.data.original
#df.head(10)
df.shape


# In[8]:


import streamlit as st


# **2. Setting Up Column Header**

# In[ ]:


#df.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']


# **3. Checking Data Types**

# In[9]:


df.info()


# **4. Checking Null Values**

# In[ ]:


df.isnull().sum()


# **5. Checking Unique Values and its count**

# In[ ]:


df['workclass'].unique()


# In[ ]:


df.workclass.value_counts()


# In[ ]:


df.occupation.value_counts()


# In[ ]:


df['native-country'].value_counts()


# In[ ]:


df['relationship'].value_counts()


# **6. Checking Stats**

# In[6]:


df.describe()


# In[ ]:



#df[df["capital-gain"] == 0]

df2 = len(df[df["capital-gain"]==0])
df2


# In[ ]:


df3 = len(df[df["capital-loss"]==0])
df3


# **7. Displaying Dataframe**

# In[ ]:


df.head(10)


# **8. Data Cleaning**
# 1. Replacing '?' with null
# 

# In[10]:


df.replace('?',np.nan, inplace=True)
df['workclass'].value_counts()


# In[11]:


df['education'].str.strip()


# In[12]:


df['workclass'].str.strip()


# In[10]:


df_obj = df.select_dtypes('object')
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
print (df)


# In[13]:


df['workclass'].value_counts()


# **ii. Fill occupation passed on similar income group, based on frequency of occurence**

# In[14]:


df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])


# **iii. Change  1-12th as 'Did not graduate'**

# In[15]:


df['education'].unique()


# In[13]:


df.replace(['5th-6th', '10th', '1st-4th', 'Preschool', '12th','11th', '9th','7th-8th'],'Did Not Graduate', inplace=True)
df


# **iii. Drop Education Num column**

# In[ ]:


#df.drop(columns=['education-num'],inplace=True)


# **iv. Add column CG_Category 0: No capital gains, 1: capital gains, 2: Undeclared capital gains**

# In[16]:


df['CG_Category']=np.where(df['capital-gain']==0,0,np.where(df['capital-gain']==99999,2,1))
df
df['CG_Category'].value_counts()


# In[17]:


df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])


# In[ ]:


df['occupation'].value_counts()


# **Histogram**

# In[25]:


st.set_page_config(
    page_title ='Trial 1',
    layout="wide")


# In[26]:


#fig, ax = plt.subplots()
#ax.set_title('Age')
df.hist( bins=10, sharey=True, figsize=[50,50],xlabelsize=20,ylabelsize=20)


# In[27]:


st.pyplot(plt.gcf())


# In[ ]:


df.describe()


# In[ ]:


df['income'] = df['income'].str.replace('.', '')


# In[ ]:


sns.boxplot(data=df).set(title="Box Plot ")


# In[ ]:


df.boxplot(column='education-num', by='income')


# In[ ]:


df.boxplot(column='age', by='income')


# In[ ]:


plt.figure(figsize=(12, 4))
sns.boxplot(x='workclass', y='age', hue='workclass', data=df, palette='Set3')


# In[ ]:


plt.figure(figsize=(25, 5))
sns.barplot(x = df['education'], y = df['income'],orient='v')


# In[ ]:


plt.figure(figsize=(25, 5))
sns.barplot(x = df['marital-status'], y = df['income'],orient='v')


# In[ ]:


plt.figure(figsize=(10, 5))
plt.scatter('race', 'hours-per-week', data=df)


# In[ ]:


plt.figure(figsize=(20, 5))
plt.scatter('occupation', 'hours-per-week', data=df)


# In[ ]:


#plt.figure(figsize=(10, 5))
plt.bar('sex', 'hours-per-week', data=df)


# In[ ]:


sns.heatmap(df.corr(),cmap='coolwarm', annot=True)


# In[ ]:


pearson_corr=df.corr(method='pearson')
pearson_corr
#sns.heatmap(pearson_corr,cmap='coolwarm', annot=True)


# In[ ]:


# @title age

from matplotlib import pyplot as plt
pearson_corr['age'].plot(kind='hist', bins=20, title='age')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[ ]:


import scipy.stats as stats


# In[ ]:


zsc_ed_num=stats.zscore(df['education-num'])
zsc_ed_num


# In[ ]:


df['education-num'].describe()


# In[ ]:


df['income'].describe()


# In[ ]:


sns.displot(data=df,x='age',y='income',)


# In[ ]:


q25=df['age'].quantile(0.25)
q75=df['age'].quantile(0.75)
iqr=q75-q25


# In[ ]:


upper_limit = q25 + 1.5 * iqr
lower_limit = q75 - 1.5 * iqr


# In[ ]:


df[df['age'] > upper_limit]
df[df['age'] < lower_limit]


# In[ ]:


new_df = df[df['age'] < upper_limit]
new_df.shape


# In[ ]:


new_df[(new_df.age<lower_limit)|(new_df.age> upper_limit)]


# In[ ]:


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.histplot(df['age'],kde=True)
plt.subplot(2,2,2)
sns.boxplot(df['age'])
plt.subplot(2,2,3)
sns.histplot(new_df['age'],kde=True)
plt.subplot(2,2,4)
sns.boxplot(new_df['age'])
plt.show()


# In[ ]:


pip install category_encoders


# In[ ]:


from sklearn.model_selection import train_test_split
import category_encoders as ce


# In[ ]:


X = new_df.drop(['income'], axis=1)

y = new_df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[ ]:


X_train, X_test, y_train, y_test


# In[ ]:


encoder = ce.OrdinalEncoder( cols=['workclass', 'education', 'marital-status', 'relationship', 'race', 'sex','native-country','occupation'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

