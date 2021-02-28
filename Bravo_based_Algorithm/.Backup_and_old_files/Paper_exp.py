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


X=np.array([[-1.0,1.0],[-1.0,1.0]]);


tmp_L=list(itertools.product(X.tolist()[0],X.tolist()[1]))
L=tmp_L

def compute_torque(X,U,L_point):
 
    q=L_point[0]
    dq=L_point[1]
    qmax=X[0][1]
    qmin=X[0][0]
    EPS =1E-06
    print(q,dq,qmax,qmin)
    def cost_fun(tau): # FLEXIBILITY: can i add another term to taking account of the previous step?
        cost = dq +dt*(tau+nonlinear) 
        return cost**2

    # def pos_upper_bound(tau): # No continuous time bound satisfaction!       
    #     position =q+dt*dq+dt**2 * ((tau+nonlinear)/M)/2 ;
    #     return position+qmax

    def pos_lower_bound(tau):        
        position = q+dt*dq+dt**2 * ((tau+nonlinear)/M)/2 ;
        return -abs(position)+qmax
    
    def pos_middle_position(tau):
        position = q+dt/2*dq+(dt/2)**2 * ((tau+nonlinear)/M)/2 ;
        return -abs(position)+qmax
    
    r=minimize(cost_fun, [0], jac=False, method='slsqp',  # slsqp
                      options={'maxiter': 200, 'disp': True }, # maximum iteration number
                     constraints=(
                         {'type':'ineq','fun': pos_middle_position},
                         {'type':'ineq','fun': pos_lower_bound}
                                  ))
    
    print('\n \n',pos_lower_bound(r.x),cost_fun(r.x))
    
    return r

i=0
while (L !=[] and i<=300):
      
 
    
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