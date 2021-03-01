# -*- coding: utf-8 -*-

from Bravo_based_Algorithm.Functions import  Approved_point_2, rejected_point,compute_torque,Approved_point, rejected_point_2
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
from pylab import *

# initialize lists
L=[];                               # contains all the points that enter in the algorithm
P=[(0.0,0.0)]; # contains all the points positicely evaluated by the algorithm
R=[];                      # contains all the points rejected by the algorithm

# initialize bound on states

X=np.array([[-0.5,0.5],[-0.5,0.5]]);    # position and velocity bound
U=np.array([[-80.0,80.0]]);             # torque bounds

# Data of the system 
dt=0.05;
M=1.5;
g=9.81;
l=1.5;
tau_max=40;
tau_min=-40;

# definition of the initial omega

omega=np.array([0.0]) # not yet used

# starting points in our L list
# defined as the extremes verteces of our X-region

tmp_L=list(itertools.product(X.tolist()[0],X.tolist()[1]))
L=tmp_L
#L=[(0.5,0.5)]
#L=np.array(tmp_L); 
L.append((-0.5,0.0))
L.append((0.5,0.0))
#L=[(0.5,0.5)]

print('\n initial starting points in the list : \n')
print(L,'\n');


###############################################################
#################### Starting the Algorithm ###################
###############################################################

# Non linearity for a simple pendulum system
# Simple joint with nl effect
def nl_effect_pendulum(q,dq):
    nl= -M*g*l*sin(q)*0.0+M*l**2*dq/dt; # Is it correct ???
    return nl;

'''
We check if the elements is L can reach the desired conditions in one time-step,
if they we put in P, if not we reject the points in R. we choose the point using
compute_torque().
The perfect exit condition for this loop is to empty the L checking the distance 
between two consecutive generated points, but is not yet implemented
'''
i=0
while (L !=[] and i<=300):
      
    nonlinear = nl_effect_pendulum(L[0][0],L[0][1]);
    print('non linear effect: ',nonlinear)
    
    r=compute_torque(X,U,L[0],dt,nonlinear,M)
    print('\n computed tau',r.x);
    
    i=i+1
    
    if (r.x >= tau_max or r.x <= tau_min):
        if (L[0] in R):
            R
        else:
            R.append(L[0])
            
        #remove_old_add_new(L[0],L,P,X)
        rejected_point(L[0],L,P,X)
                    
    else:
        if (L[0] in P):
            P
        else:
            P.append(L[0])
        
        #remove_old_add_new_if_P(L[0],L,R,X)
        Approved_point(L[0],L,R,X)
    
    print('####### \n iteration ',i,'\n ########')
        # if ( (0.0,0.0) in P):
        #     P.remove((0.0,0.0))
        # L.remove(L[0])
        # if (abs(L[i][0]) >= abs(L[i][1])):
        #     #for q in range(len(L)):
        #         # if (L[i][0]==abs(L[q][0]):
        #         #     q=q
        #         #     # must select properly
        #     b = np.array([[L[i][0]/2.0,L[i][1]]])
        #     L=np.concatenate((L,b))
        # L[i]=[0,0]

print('this is L \n',L,'\n')
print('this is R \n',R,'\n')
print('this is P \n',P,'\n')
print(i)
(f,ax) = plut.create_empty_figure(1)
eps=1E-2
ax.set_ylim([-0.5-eps,0.5+eps])
ax.set_xlim([-0.5-eps,0.5+eps])


if(L != [] ):
    LL=np.array(L)
    scatter(LL[:,0],LL[:,1],color="blue")

if(R != [] ):
    RR=np.array(R)
    scatter(RR[:,0],RR[:,1],color="red")
    
if(P != [] ):
    PP=np.array(P)
    scatter(PP[:,0],PP[:,1],color="green")
LW=1
ax.plot([-0.5, 0.5], [-0.5, -0.5], 'r--',linewidth=LW);
ax.plot([-0.5, 0.5], [0.5, 0.5], 'r--',linewidth=LW);
ax.plot([-0.5, -0.5], [-0.5, 0.5], 'r--',linewidth=LW);
ax.plot([0.5, 0.5], [-0.5, 0.5], 'r--',linewidth=LW);

# for i,txt in enumerate(P):
#     ax.annotate(i, P[i])

    
plt.show()
