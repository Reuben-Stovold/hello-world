
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('happyscore_income .csv')

data


# In[14]:


happy = data['happyScore']
gdp = data['GDP']

income = data['avg_income']


# In[21]:


import matplotlib.pyplot as plt
#plotting gpd with happieness to see if there is any visual correlation
plt.xlabel('gdp')
plt.ylabel('happy score')
#changing plt*0.05 makes the plots smaller and easier to read
plt.scatter(gdp, happy, s = income*0.05, alpha = 0.5)
# the size of the dot represents the inequality


# In[22]:


data.sort_values('GDP', inplace=True)
#sorting the data via the gdp values
data


# In[25]:


import numpy as np

gpd_mean = np.mean(data['GDP'])
print


# In[38]:


import numpy as np

data.sort_values('GDP', inplace = True)

high_gdp = data[data['GDP'] > 0.5]

GDP_mean = np.mean(high_gdp['GDP'])

all_gdp_mean = np.mean(data['GDP'])

print(GDP_mean, all_gdp_mean)


# In[39]:


plt.scatter(high_gdp['GDP'], high_gdp['happyScore'])

plt.text(high_gdp.iloc[0]['GDP'],
        high_gdp.iloc[0]['happyScore'],
        high_gdp.iloc[0]['country'])

plt.text(high_gdp.iloc[-1]['GDP'],
        high_gdp.iloc[-1]['happyScore'],
        high_gdp.iloc[-1]['country'])


# In[40]:


#running through the code and printing out all the countries
for k,row in high_gdp.iterrows():
    print(row['country'])


# In[56]:


#using K-means to interpret the data

from sklearn.cluster import KMeans
import numpy as np

gdp_happy = np.column_stack((gdp, happy))

km_res = KMeans(n_clusters = 3).fit(gdp_happy)

clusters = km_res.cluster_centers_

plt.scatter(gdp, happy)
plt.scatter(clusters[:,0], clusters[:,1], s=1000)

plt.title('KMeans of GDP and Happy Score')
plt.xlabel('GDP')
plt.ylabel('Happy score')

plt.text()

