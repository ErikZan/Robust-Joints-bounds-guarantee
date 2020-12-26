# -*- coding: utf-8 -*-

from Bravo_based_Algorithm.Functions import  compute_torque
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
import Functions

# initialize lists
L=np.array([[0.0,0.0]]); # contains all the points that enter in the algorithm
P=np.array([[0.0,0.0]]); # contains all the points positicely evaluated by the algorithm
R=np.array([[0.0,0.0]]); # contains all the points rejected by the algorithm

# initialize bound on states

X=np.array([[-0.5,0.5],[-0.1,0.1]]); # position and velocity bound
U=np.array([[-80.0,80.0]]); # torque bounds

# Data of the system 
dt=0.1;
M=1.5;
g=9.81;
l=1.5;
tau_max=50;
tau_min=-50;

# definition of the initial omega

omega=np.array([0.0])

# starting points in our L list
# defined as the extremes verteces of our X-region

tmp_L=list(itertools.product(X.tolist()[0],X.tolist()[1]))
L=np.array(tmp_L); 
print('\n initial starting points in the list : \n')
print(L,'\n');


###############################################################
#################### Starting the Algorithm ###################
###############################################################

# Non linearity for a simple pendulum system
def nl_effect_pendulum(q,dq):
    nl= -M*g*l*sin(q)+M*l**2*dq/dt;
    return nl;

q=0
for i in range(len(L)):
    nonlinear= nl_effect_pendulum(L[i][0],L[i][1]);
    r=compute_torque(X,U,L[i],dt,nonlinear,M)
    print(r.x);
    
    if (r.x > tau_max or r.x <tau_min):
        if (abs(L[i][0]) >= abs(L[i][1])):
            #for q in range(len(L)):
                # if (L[i][0]==abs(L[q][0]):
                #     q=q
                #     # must select properly
            b = np.array([[L[i][0]/2.0,L[i][1]]])
            L=np.concatenate((L,b))
        L[i]=[0,0]

print(L)