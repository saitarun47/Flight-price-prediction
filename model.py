#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[2]:


df=pd.read_excel("C:/Users/tanuj/Downloads/Data_Train.xlsx")
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# # Date_of_Journey	
# 

# In[5]:


df['Date_of_Journey']=df['Date_of_Journey'].astype(str)


# In[6]:


df['date_of_Journey']=df['Date_of_Journey'].str.split('/').str[0]
df['month_of_Journey']=df['Date_of_Journey'].str.split('/').str[1]
df['year_of_Journey']=df['Date_of_Journey'].str.split('/').str[2]


# In[7]:


df.head()


# In[8]:


df['date_of_Journey']=df['date_of_Journey'].astype(int)
df['month_of_Journey']=df['month_of_Journey'].astype(int)
df['year_of_Journey']=df['year_of_Journey'].astype(int)


# In[9]:


df.info()


# In[10]:


df.drop('Date_of_Journey',axis=1,inplace=True)
df.head()


# In[11]:


df['Arrival_Time']=df['Arrival_Time'].str.split(' ').str[0]


# In[12]:


df['arrival_hour']=df['Arrival_Time'].str.split(':').str[0]
df['arrival_min']=df['Arrival_Time'].str.split(':').str[1]
df.head()


# In[13]:


df.drop('Arrival_Time',axis=1,inplace=True)
df.head()


# In[14]:


df['Dep_hour']=df['Dep_Time'].str.split(':').str[0]
df['Dep_min']=df['Dep_Time'].str.split(':').str[1]
df.head()


# In[15]:


df.drop('Dep_Time',axis=1,inplace=True)
df.head()


# In[16]:


df['Duration']=df['Duration'].str.split(' ').str[0]
df.head()


# In[17]:


df['Duration']=df['Duration'].str.split('h').str[0]
df.head()


# In[18]:


df.drop('Route',axis=1,inplace=True)
df.head()


# In[19]:


df.info()


# In[20]:


df['arrival_hour']=df['arrival_hour'].astype(int)
df['arrival_min']=df['arrival_min'].astype(int)
df['Dep_hour']=df['Dep_hour'].astype(int)
df['Dep_min']=df['Dep_min'].astype(int)


# In[21]:


df.info()


# In[22]:


df['Total_Stops'].unique()


# In[23]:


df['Total_Stops']=df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})


# In[24]:


df.head()


# In[25]:


df['Additional_Info'].unique()


# In[26]:


df['Additional_Info'].value_counts()


# In[27]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

df['Airline']=labelencoder.fit_transform(df['Airline'])
df['Source']=labelencoder.fit_transform(df['Source'])
df['Destination']=labelencoder.fit_transform(df['Destination'])
df['Additional_Info']=labelencoder.fit_transform(df['Additional_Info'])


# In[28]:


df.head()


# In[29]:


df.info()


# In[30]:




# In[31]:


df['Duration'].value_counts()


# In[32]:


df[df['Duration']=='5m']


# In[33]:


df.drop(6474,axis=0,inplace=True)
df.drop(2660,axis=0,inplace=True)


# In[34]:


df.shape


# In[35]:


df.info()


# In[36]:


df['Duration']=df['Duration'].astype(int)
df.info()


# In[55]:




# # missing values

# In[56]:


df.isnull().sum()


# In[57]:


df[df['Total_Stops'].isnull()]


# In[58]:


df['Total_Stops']=df['Total_Stops'].fillna(1)


# In[59]:


df.isnull().sum()


# # train test split 

# In[60]:


y=df['Price']


# In[61]:


x=df.drop('Price',axis=1)


# In[62]:


x


# In[63]:


y


# In[64]:


x.shape


# In[65]:


y.shape


# In[66]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# In[67]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# # standardization
# 

# In[68]:





# In[69]:






# # Linear reg

# In[70]:


#from sklearn.linear_model import LinearRegression
#model=LinearRegression()

#model.fit(x_train_scaled,y_train_scaled)
#y_pred=model.predict(x_test_scaled)

#print(y_pred)


# In[71]:


#from sklearn.metrics import r2_score

#r2_score=r2_score(y_test,y_pred)
#print(r2_score)


# # Random forest

# In[76]:


from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[77]:


from sklearn.metrics import r2_score
r2_value=r2_score(y_test,y_pred)
print(r2_value)


# In[78]:




# In[ ]:

import pickle
pickle.dump(model,open('model.pkl','wb'))


