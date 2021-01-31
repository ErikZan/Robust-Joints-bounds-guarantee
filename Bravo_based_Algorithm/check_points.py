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

m=1.5 #*1E-1
l=2.0
g=9.81
torque=[-15.0,15.0]
X_all=[-0.0,0.5,0.0,2.0]
dt=0.001
division=500
x_axis_division= np.arange(0.0,0.5,0.5/division)
y_axis_division= np.arange(0.0,2.0,2.0/division)

q_viable=[]
q_not_viable=[]

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
            q[i+1]=q[i]+dt*dq[i]+dt**2*((torque[0]-m*l*g*np.sin(q[i]))/(m*l**2))/2
            dq[i+1]=dq[i]+dt*((torque[0]-m*l*g*np.sin(q[i]))/(m*l**2))
    
            if (dq[i+1]<=0):
                if (q[i+1]<=X_all[1]):
                    q_viable.append([q0,dq0])
                if (q[i+1]>=X_all[1]):
                    q_not_viable.append([q0,dq0])
                break
        
        print(k,j)        

q_viable = np.array(q_viable)
q_not_viable = np.array(q_not_viable)

# with open('q_viable.npy','wb') as f:
#     np.save(f,q_viable)
# with open('q_not_viable.npy','wb') as ff:   
#     np.save(ff,q_not_viable)


np.save('q_viable.npy',q_viable) 
np.save('q_not_viable.npy',q_not_viable)


