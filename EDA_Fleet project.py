#!/usr/bin/env python
# coding: utf-8

# In[7]:


# importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xlrd


# In[12]:


# read the csv file and make it as a dataframe
df=pd.read_csv('/users/apple/Downloads/ss.csv')


# In[14]:


# to dispay top 5 observations
df.head()


# In[4]:


# to check no. of values in Customer Category
df['Customer Category'].unique()


# In[5]:


#df['PinCode'].isnull().sum()
df.apply(lambda x:sum(x.isnull()),axis=0) 
#def num_missing(x):
 #   return sum(x.isnull())
#print(df.apply(num_missing,axis=1).head())


# <h1>Univariate Analysis</h1>

# In[6]:


# Check the univariate analysis for Customer Category
sns.countplot(df['Customer Category'])
plt.xticks(rotation=90)


# In[7]:


# Getting top Areas where the value count>18

area_counts=df['Area / Village'].value_counts()
#print(area_counts)
area_list=area_counts.nlargest(10).index.tolist()
print(area_list)
#top_Area=df[df['Area / Village'].isin(area_list)]
top_Area=df[df['Area / Village'].isin(area_list)]
top_Area


# In[10]:


# Univariate analysis for top10 area/village based on valuecount

plt.figure(figsize=(30,20))
sns.countplot(top_Area['Area / Village'])
plt.xticks(rotation=90)


# In[11]:


# filtering top ten pincodesbsed on value counts
pincode_count=df['PinCode'].value_counts()
Top_Pincode_list=(pincode_count.nlargest(10)).index.tolist()
Top_Pincode=df[df['PinCode'].isin(Top_Pincode_list)]
Top_Pincode_list
Top_Pincode


# In[12]:


# Univariate analysis for top10 pincodes based on valuecount

plt.figure(figsize=(20,20))
#sns.countplot(top_Area['Area / Village'],hue=df['Disrtict'])
sns.countplot(Top_Pincode['PinCode'])
plt.xticks(rotation=90)


# In[13]:


# check the no. of city available in the dataset
df['City / Hobli'].unique()


# In[26]:


# filtering top ten cities based on value counts
city_count=df['City / Hobli'].value_counts()
# Top_Pincode_list=(pincode_count[pincode_count>30]).index.tolist()
city_list=(city_count.nlargest(10)).index.tolist()
Top_cities=df[df['City / Hobli'].isin(city_list)]
city_list
Top_cities


# In[31]:


plt.figure(figsize=(20,10))
sns.countplot(df['Rating'])


# In[16]:


df['Rating Count'].unique()


# In[15]:


"""df['Rating Count'].isnull().sum()

df['Rating Count'].fillna(0,inplace=True)"""

df['Rating Count'].str.get(0).value_counts()
df['Rating Count']=df['Rating Count'].astype(int)


# In[ ]:





# <h1>Bivariate Analysis </h1>

# In[18]:


plt.figure(figsize=(30,30))
sns.countplot(top_Area['Area / Village'],hue=df['Customer Category'])
plt.xticks(rotation=90)
plt.legend(loc=1,prop={'size':20})


# In[ ]:





# In[18]:


dist=df['Area / Village'].groupby(df['Disrtict']).count()


# In[19]:


dist


# In[20]:


df['Disrtict'].unique()


# In[21]:


sns.countplot(df['Disrtict'])
plt.xticks(rotation=90)


# In[21]:


district_wise=pd.DataFrame(df.loc[df['Disrtict']=='Bengaluru'])
#df.groupby(df['Disrtict'])['Area / Village'].count().plot.bar()


# In[22]:


sns.countplot(district_wise['Customer Category'])
plt.xticks(rotation=90)


# In[23]:


#district_wise['Rating'].fillna(0,inplace=True)
district_wise['Rating'].isnull().count()
district_wise['Rating'].replace('na',0,inplace=True)


# In[24]:


district_wise['Rating'].unique()


# In[34]:


district_wise['Rating'].isnull().sum()


# In[26]:


district_wise['Rating'].astype('float')
#district_wise['PinCode'].astype('int')


# In[27]:


df['Area / Village'].groupby(df['Disrtict']=='Bengaluru').count().plot.bar()


# In[28]:


district_wise.groupby(['Area / Village','Customer Category'])['Area / Village'].count().sort_values(ascending=False)


# In[29]:


plt.figure(figsize=(50,30))
district_wise.groupby(['Area / Village','Customer Category'])['Area / Village'].count().plot.bar()
plt.xticks(rotation=90)


# In[30]:


plt.figure(figsize=(30,20))
district_wise.groupby(district_wise['Area / Village'])['Customer Category'].count().plot(kind='bar')
plt.xticks(rotation=90)


# In[31]:


plt.figure(figsize=(20,10))
#df['Area / Village'].groupby(df['Disrtict']=='Bengaluru').count().plot.bar()
district_wise.groupby(district_wise['Area / Village'])['Customer Category'].count().plot.bar()
plt.xticks(rotation=90)


# In[32]:


district_wise


# In[33]:


#plt.figure(figsize=(20,20))
district_wise.groupby([district_wise['Area / Village'],district_wise['PinCode']])['Rating'].mean()
#district_wise.groupby(district_wise['Area / Village'])['Rating'].count().plot.bar()
#plt.xticks(rotation=90)


# In[ ]:




