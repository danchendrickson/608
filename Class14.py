# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:09:43 2021

@author: Hendrickson
"""

import numpy as np 
import matplotlib.pyplot as plt 

pht = 0.1
pth = 0.5

phh = 1 - pht
ptt = 1 - pth

P = np.matrix([[phh, pht],[pth, ptt]])

xh0=1
xt0= 1 - xh0

nflips = 50

x = []
xc = np.array([xh0,xt0])
for i in range(nflips):
    x.append([i,xc[0],xc[1]])
    xc = np.dot(xc,P)
    xc=np.array(xc)
    xc=xc[0]

x.append([nflips+1,xc[0],xc[1]])

x = np.matrix(x)

fig = plt.figure()
plt.plot(x[:,0],x[:,1],'-',label='heads')
plt.plot(x[:,0],x[:,2],'-',label='tails')
plt.legend()
plt.title('Probabilities bynumber of trails')
plt.xlabel("Traial Number")
plt.ylabel('Probabilitiy')
plt.show()

