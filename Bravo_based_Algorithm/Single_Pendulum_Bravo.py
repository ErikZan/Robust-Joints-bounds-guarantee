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

from scipy.optimize import minimize

#### Data ####
m=1.5 
l=2.0
g=9.81
torque=[-30.0,30.0]
q_interval=[0.0,3.14/2]

#### Function ####
def max_acc(Xi,tau): # interval aritmetic ?
    
    [tau_min,tau_max] = tau
    [q_min,q_max] =Xi
    
    ddq_min=(tau_min-m*l*g*np.sin(q_max))/(m*l**2)
    ddq_max=(tau_max-m*l*g*np.sin(q_min))/(m*l**2)
    
    return [ddq_min,ddq_max]
    
####  ####
step_size=0.05
L=np.arange(q_interval[0],q_interval[1],step_size)
Acc_min=np.zeros(size(L))
Acc_max=np.zeros(size(L))

for i in range(size(L)-1):
    Acc_min[i]=max_acc((L[i],L[i+1]),torque)[0]
    Acc_max[i]=max_acc((L[i],L[i+1]),torque)[1]
    
    print(i,Acc_min[i],Acc_max[i])
    
#### Maximize are of viable squared area ###
q_from_optimization=np.zeros(size(L))
dq_from_optimization=np.zeros(size(L))

for i in range(size(L)-1):
    def area(q):
        qmin=L[i]
        qmax=L[i+1]
        acc=Acc_min[i]
        return -((q-qmin)*sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))))

    def q_limit(q):
        qmin=L[i]
        return q-qmin

    def q_limit_pos(q):
        qmax=L[i+1]
        return qmax-q

    r = minimize(area,L[i], jac=False, method='slsqp',  # slsqp
                        options={'maxiter': 200, 'disp': True },#, # maximum iteration number
                        constraints=(
                            {'type':'ineq','fun': q_limit},
                            {'type':'ineq','fun': q_limit_pos}
                                    ))
    print(r.x)
    (q_from_optimization[i],dq_from_optimization[i])=(r.x,sqrt(2*abs(r.x)*((L[i+1]-L[i])-(r.x-L[i]))))

for i in range(size(q_from_optimization)):
    print(q_from_optimization[i],dq_from_optimization[i])
    
    
for i in range(size(L)-2):
    def area(q):
        qmin=L[i]
        qmax=L[i+1]
        acc=Acc_min[i]
        return -((q-qmin)*sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))))

    def q_limit(q):
        qmin=L[i]
        return q-qmin

    def q_limit_pos(q):
        qmax=L[i+1]
        return qmax-q

    r = minimize(area,L[i], jac=False, method='slsqp',  # slsqp
                        options={'maxiter': 200, 'disp': True },#, # maximum iteration number
                        constraints=(
                            {'type':'ineq','fun': q_limit},
                            {'type':'ineq','fun': q_limit_pos}
                                    ))
    print(r.x)
    (q_from_optimization[i],dq_from_optimization[i])=(r.x,sqrt(2*abs(r.x)*((L[i+1]-L[i])-(r.x-L[i]))))
    L[i+1]=q_from_optimization[i]
    L[i+2]=q_from_optimization[i]+step_size
    
for i in range(size(q_from_optimization)):
    print(q_from_optimization[i],dq_from_optimization[i])
    
(f,ax) = plut.create_empty_figure(1)
LW=1

def square_drawer(qmin,qmax,dqmin,dqmax):
    ax.plot([qmin, qmax], [0, 0], 'r--',linewidth=LW);
    ax.plot([qmin, qmax], [dqmax, dqmax], 'r--',linewidth=LW);
    ax.plot([qmin, qmin], [0, dqmax], 'r--',linewidth=LW);
    ax.plot([qmax, qmax], [0, dqmax], 'r--',linewidth=LW);

scatter(q_from_optimization[:],dq_from_optimization[:],color="green")

for i in range(size(q_from_optimization)-1):
    square_drawer(q_from_optimization[i],q_from_optimization[i+1],dq_from_optimization[i],dq_from_optimization[i+1])

for i,txt in enumerate(Acc_min):
     ax.annotate(str(Acc_min[i]), xy=(q_from_optimization[i],dq_from_optimization[i]),size=10)

plt.show()
