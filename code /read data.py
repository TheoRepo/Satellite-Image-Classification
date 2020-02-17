from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import h5py
from scipy import io
import scipy
import colorsys
from sklearn import svm
from sklearn import metrics
import pickle

#read mat into python
datapath="/Users/chris/Desktop/sat-4-full.mat";
features_struct=scipy.io.loadmat("/Users/chris/Desktop/sat-4-full.mat");

trainx=features_struct["train_x"];
x=np.zeros(((30000,784,4))) # select 50000 from Trainx to Train the model
w=0;

for l in range(0,30000):
    for i in range(0,28):
        for j in range(0,28):
            for k in range(0,4):
                x[l][w][k]=trainx[i][j][k][l]
            w=w+1
    w=0
#print(x)

v=np.zeros(((30000,784,3)))
w=0

for l in range(0,30000):
    for w in range(0,784):
        r=x[l][w][0]/255
        g=x[l][w][1]/255
        b=x[l][w][2]/255
        v[l][w][0]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[0]
        v[l][w][1]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[1]
        v[l][w][2]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[2]

trainy=features_struct["train_y"]
n=[]
for i in range(0,30000):
	for j in range(0,4):
		if trainy[j][i] != 0:
			n.append(j)
			
#first 6 are variables, the last one is the result
        
f=np.zeros((30000,7))
h=[]
s=[]
q=[]


for l in range(0,30000):
    for w in range(0,784):
        h.append(v[l][w][0])
        s.append(v[l][w][1])
        q.append(v[l][w][2])
    f[l][0]=np.mean(h)
    f[l][1]=np.std(h)
    f[l][2]=np.mean(s)
    f[l][3]=np.std(s)
    f[l][4]=np.mean(q)
    f[l][5]=np.std(q)
    f[l][6]=n[l]
    h=[]
    s=[]
    q=[]


data1=DataFrame(f)
data1.to_csv("/Users/chris/Desktop/data1.csv")

Train_x=f[:,:6]
Train_y=f[:,6:]

model=svm.SVC(C=1100,gamma=125)
model.fit(Train_x,Train_y)
pickle.dump(model,open('/Users/chris/Desktop/model1.dat','wb'))

#read the test data

datapath="/Users/chris/Desktop/sat-4-full.mat";
features_struct=scipy.io.loadmat("/Users/chris/Desktop/sat-4-full.mat");

testx=features_struct["test_x"];
x=np.zeros(((100000,784,4))) # 
w=0;

for l in range(0,100000):
    for i in range(0,28):
        for j in range(0,28):
            for k in range(0,4):
                x[l][w][k]=testx[i][j][k][l]
            w=w+1
    w=0
#print(x)

v=np.zeros(((100000,784,3)))
w=0

for l in range(0,100000):
    for w in range(0,784):
        r=x[l][w][0]/255
        g=x[l][w][1]/255
        b=x[l][w][2]/255
        v[l][w][0]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[0]
        v[l][w][1]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[1]
        v[l][w][2]=colorsys.rgb_to_hsv(float(r)/float(255), float(g)/float(255), float(b)/float(255))[2]

testy=features_struct["test_y"]
n=[]
for i in range(0,100000):
	for j in range(0,4):
		if testy[j][i] != 0:
			n.append(j)
			
#first 6 are variables, the last one is the result
        
d=np.zeros((100000,7))
h=[]
s=[]
q=[]


for l in range(0,100000):
    for w in range(0,784):
        h.append(v[l][w][0])
        s.append(v[l][w][1])
        q.append(v[l][w][2])
    d[l][0]=np.mean(h)
    d[l][1]=np.std(h)
    d[l][2]=np.mean(s)
    d[l][3]=np.std(s)
    d[l][4]=np.mean(q)
    d[l][5]=np.std(q)
    d[l][6]=n[l]
    h=[]
    s=[]
    q=[]


data2=DataFrame(d)
data2.to_csv("/Users/chris/Desktop/data3.csv")


Test_x=d[:,:6]
Test_y=d[:,6:]
       
model=pickle.load(open('/Users/chris/Desktop/model.dat','rb'))
testy=model.predict(Test_x)




