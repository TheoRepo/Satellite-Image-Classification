#!/usr/bin/env python
# coding: utf-8

# In[34]:


from scipy.io import loadmat
import numpy as np
data = loadmat('/classes/ece2720/fpp/sat-4-full.mat')


# In[35]:


trainx = data['train_x']
trainy = data['train_y']


# ## Data Preprocessing

# In[36]:


# The goal is to build a Data frame, which uses the image as unit. There are 784 pixels to describe each unit and four channels (red, green, blue and NIR components) to describe each pixel.

x=np.zeros(((40000,784,4))) 
w=0;
for l in range(0,40000):
    for i in range(0,28):
        for j in range(0,28):
            for k in range(0,4):
                x[l][w][k]=trainx[i][j][k][l]
            w=w+1
    w=0


# In[38]:


# transfer the RGB to HSV

import colorsys

v=np.zeros(((40000,784,4)))
w=0
for l in range(0,40000):
    for w in range(0,784):
        r=x[l][w][0]
        g=x[l][w][1]
        b=x[l][w][2]
        v[l][w][0]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[0]
        v[l][w][1]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[1]
        v[l][w][2]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[2]
        v[l][w][3]=x[l][w][3]



# In[40]:


# transfer the original data set to a specific result of each image (0-barren land; 1-trees; 2-grassland; 3-none).
n=[]
for i in range(0,40000):
	for j in range(0,4):
		if trainy[j][i] != 0:
			n.append(j)


# In[41]:


# calculate the mean value and standard deviation value of H, S and V separately in each image.

f=np.zeros((40000,9))
h=[]
s=[]
q=[]
m=[]
for l in range(0,40000):
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
    f[l][8]=n[l]
    h=[]
    s=[]
    q=[]
    m=[]


# In[43]:


import pandas as pd

# create the pandas DataFrame
df = pd.DataFrame(f, columns = ['hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std', 'NIR_mean', 'NIR_std', 'land_type'])


# In[44]:


import pandas as pd
from sklearn import preprocessing

x = df.iloc[:,6:8].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df.iloc[:,6:8] = x_scaled

# 参考：https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c



# ## Parameter tuning - Cross validation in training data

# In[46]:


X=df.iloc[:,:8]
y=df.iloc[:,8]


# In[54]:


# train the SVM 
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C= 30, gamma= 16)
clf.fit(X,y)


# In[55]:


import pickle 
  
# Save the trained model as a pickle string. 
pickle.dump(clf, open("model.dat", "wb")) 

# 参考： https://machinelearningmastery.com/save-gradient-boosting-models-xgboost-python/

