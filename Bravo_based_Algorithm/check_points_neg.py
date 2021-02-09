#!/usr/bin/python

from scipy.optimize.linesearch import _quadmin
import pinocchio as se3
from pinocchio.utils import *
import os
import numpy.matlib
from numpy.linalg import pinv,inv
from math import sqrt,cos,sin
from time import sleep

import numpy as np
from numpy.random import random
import plot_utils as plut
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patches as mpatches
import datetime
import itertools
from pylab import *

from scipy.optimize import minimize,Bounds
import matplotlib.patches as mpatches
from interval import imath,interval
import sys 

g=9.81

# torque=sys.argv[0]
# X_all=sys.argv[1]
# division=sys.argv[2]
# dt=sys.argv[3]
# m=sys.argv[4]  #*1E-1
# l=sys.argv[5]

data=np.load('data_neg.npy')
print(data)

torque=data[0]
X_all=data[1]
m=data[2]  #*1E-1
l=data[3]
division=data[4]
dt=data[5]

x_axis_division= np.arange(0.0,0.5,0.5/division)
y_axis_division= np.arange(-2.0,0.0,2.0/division)

q_viable_neg=[]
q_not_viable_neg=[]

print(x_axis_division)
for k in range(size(x_axis_division)):
    
    for j in range(size(y_axis_division)):
        
        q=np.zeros(10000)
        dq=np.zeros(10000)
        q0= x_axis_division[k]
        dq0= y_axis_division[j]   
        q[0]=q0
        dq[0]=dq0
         
        for i in range(size(q)):   
            q[i+1]=q[i]+dt*dq[i]+dt**2*((torque[1]-m*l*g*np.sin(q[i]))/(m*l**2))/2
            dq[i+1]=dq[i]+dt*((torque[1]-m*l*g*np.sin(q[i]))/(m*l**2))
    
            if (dq[i+1]>=0):
                if (q[i+1]>=X_all[0]):
                    q_viable_neg.append([q0,dq0])
                if (q[i+1]<=X_all[0]):
                    q_not_viable_neg.append([q0,dq0])
                break
        
        print(k,j)        

q_viable_neg = np.array(q_viable_neg)
q_not_viable_neg = np.array(q_not_viable_neg)

# with open('q_viable_neg.npy','wb') as f:
#     np.save(f,q_viable_neg)
# with open('q_not_viable_neg.npy','wb') as ff:   
#     np.save(ff,q_not_viable_neg)


np.save('q_viable_neg.npy',q_viable_neg) 
np.save('q_not_viable_neg.npy',q_not_viable_neg)