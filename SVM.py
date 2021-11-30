#!/usr/bin/env python
# coding: utf-8

# In[51]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
import os
import graphviz
from sklearn.metrics import plot_confusion_matrix


# In[2]:


pi = math.pi
meanshift_0 = 5
meanshift_1 = 3

radius_inner = 2
var_inner = 0.5
num_inner = 499

radius_outer = 5
var_outer = 0.5
num_outer = 499

def PointsInCircum(r,n):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

a = PointsInCircum(radius_inner,num_inner)
a = np.array(a)
a = a +np.random.normal(0,var_inner,a.shape)
#shift it by some value "meanshift_x"
a[:,0] = a[:,0]+meanshift_0
a[:,1] = a[:,1]+meanshift_1

plt.scatter(a[:,0],a[:,1])
plt.show()


# In[3]:


classlist_a = np.zeros(a.shape[0])
b = PointsInCircum(radius_outer,num_outer)
b = np.array(b)
b = b +np.random.normal(0,var_outer,b.shape)
b[:,0] = b[:,0]+meanshift_0
b[:,1] = b[:,1]+meanshift_1
plt.scatter(b[:,0],b[:,1])
plt.show()


# In[4]:


classlist_b = np.ones(b.shape[0])
print(a.shape)
print(b.shape)
X = np.concatenate((a,b))
print(X.shape)
Y = np.concatenate((classlist_a, classlist_b))
print(Y.shape)
print(X[:,0].shape)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],c=Y.squeeze())
plt.title('Two classes of circles')
plt.xlabel('x_0')
plt.ylabel('x_1')
plt.show()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
clf = svm.SVC(kernel='rbf', C=1, random_state=0, gamma='scale') #000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cf = confusion_matrix(y_test, y_pred)
print(cf)
accuracy = np.trace(cf)/y_pred.shape[0]
print('accuracy', accuracy)


# In[6]:


Z1 = clf.decision_function(X)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


# In[7]:


xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')
plt.show()


# In[13]:


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
in_file = 'A:/Aadharsh/Repo/SVM/diabetes.csv'
dataset = pd.read_csv(in_file)
print('read success')
#print(dataset['SARS-Cov-2 exam result'])
#dataset["SARS-Cov-2 exam result"].replace({"negative": 0, "positive": 1}, inplace=True)
target = dataset["Outcome"]
print(target)
dataset.drop("Outcome", axis=1,inplace=True)


# In[58]:


cols = cols = list(np.array([1]))+list(np.arange(1,8))
x = dataset.iloc[:,cols]
y = np.expand_dims(target.to_numpy(),axis=1)
x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.2, random_state=0)
#print ("x_train =", x_train)
#print ("y_train =", y_train)
#print ("x_test =", x_test)
#print ("y_test =", y_test)
#print(y_test[0])
clf = svm.SVC(kernel='rbf', C=5, random_state=0, probability=True) #000)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
cf = confusion_matrix(y_test, y_pred)
print("confusion matrix\n", cf)
accuracy = np.trace(cf)/y_pred.shape[0]
print('accuracy =', accuracy)
print(clf.score(x_test, y_test))


# In[32]:


x_patient = x_test[0].reshape(1,-1)    #assume this is already binned
y_out = clf.predict_proba(x_patient.reshape(1,-1))
print('predicted first test sample')
print(y_out)
print (y_test[0])


# In[40]:


y_out = clf.predict_proba(x_test)
y_predict=clf.predict(x_test)
print("Probabilities of each class for each sample\n", y_out[0:4])
print('predicted values\n', y_predict[0:4])
print('actual values\n', y_test[0:4])


# In[54]:


clf=


# In[ ]:




