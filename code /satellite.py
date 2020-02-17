#!/usr/bin/env python
# coding: utf-8

# In[12]:


from scipy.io import loadmat
import numpy as np
import pandas as pd
from pandas import DataFrame


# In[13]:


# test data transformation

data = loadmat("/classes/ece2720/fpp/test_x_only.mat")
testx = data['test_x']


# In[14]:


# The goal is to build a Data frame, which uses the image as unit. There are 784 pixels to describe each unit and four channels (red, green, blue and NIR components) to describe each pixel.

x=np.zeros(((100000,784,4))) 
w=0;
for l in range(0,100000):
    for i in range(0,28):
        for j in range(0,28):
            for k in range(0,4):
                x[l][w][k]=testx[i][j][k][l]
            w=w+1
    w=0


# In[15]:


# transfer the RGB to HSV

import colorsys

v=np.zeros(((100000,784,4)))
w=0
for l in range(0,100000):
    for w in range(0,784):
        r=x[l][w][0]
        g=x[l][w][1]
        b=x[l][w][2]
        v[l][w][0]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[0]
        v[l][w][1]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[1]
        v[l][w][2]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[2]
        v[l][w][3]=x[l][w][3]


# In[16]:


# calculate the mean value and standard deviation value of H, S and V separately in each image.

f=np.zeros((100000,8))
h=[]
s=[]
q=[]
m=[]
for l in range(0,100000):
    for w in range(0,784):
        h.append(v[l][w][0])
        s.append(v[l][w][1])
        q.append(v[l][w][2])
        m.append(v[l][w][3])
    f[l][0]=np.mean(h)
    f[l][1]=np.std(h)
    f[l][2]=np.mean(s)
    f[l][3]=np.std(s)
    f[l][4]=np.mean(q)
    f[l][5]=np.std(q)
    f[l][6]=np.mean(m)
    f[l][7]=np.std(m)
    h=[]
    s=[]
    q=[]
    m=[]


# In[17]:


# create the pandas DataFrame
df_test = pd.DataFrame(f, columns = ['hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std', 'NIR_mean', 'NIR_std'])


# In[18]:


import pandas as pd
from sklearn import preprocessing

x = df_test.iloc[:,6:8].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df_test.iloc[:,6:8] = x_scaled


# In[19]:


test_x=df_test.iloc[:,:8]


# In[20]:


import pickle

# Load the SVM from model.dat using the 'pickle' module
f = open('model.dat','r')
loaded_model = pickle.load(f)
f.close()


# In[21]:


testy=loaded_model.predict(test_x)
r=map(int,testy)


# In[22]:


import numpy as np

L = ['barren land', 'trees', 'grassland', 'none']
s = ','.join([L[t] for t in r])
f = open('landuse.csv', 'w')
f.write(s)
f.close()

