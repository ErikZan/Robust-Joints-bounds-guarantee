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

#### Data ####
m=1.5 
l=2.0
g=9.81
torque=[-15.0,15.0]
X_all=[0.0,0.5,0.0,2.0]
#### Variables ####
L=[X_all]
R=[]
#### Functions ####

def max_acc(Xi,tau,plot_trigger='False'): # interval aritmetic ?
    ''' Given an interval/range and a dynamic model compute the minumum positive
    and maximum negative acceleration avaible in that range  
    '''
    [q_min,q_max,dq_min,dq_max]=Xi; #  al momento non abbiamo velocità nell'espressione
    
    
    def ddq_max(q):
        return np.sign(tau)*(tau-m*l*g*np.sin(q))/(m*l**2)
    
    def q_limit(q):
        return q-q_min

    def q_limit_pos(q):
        return q_max-q
    
    #bnds=Bounds([q_min],[q_max])
    
    r = minimize(ddq_max,np.sign(tau)*(tau-m*l*g*np.sin(q_max))/(m*l**2), jac=False, method='slsqp',  # slsqp
                        options={'maxiter': 200, 'disp': plot_trigger }, 
                        constraints=(
                            {'type':'ineq','fun': q_limit},
                            {'type':'ineq','fun': q_limit_pos}
                                    )
                                    )
    
    return r

def Minimize_area(X_now,acc):
    
    (qmin,qmax,dq_target)=(X_now[0],X_now[1],X_now[2])
    
    def area(q):
        '''
        Area defined as q*dq where dq is already in the form of 
        dq=sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))) 
        and limits are given only on positions
        '''
        return -((q-qmin)*(  sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))+1E-6))  +  dq_target**2  )

    def q_limit(q):
        return q-qmin

    def q_limit_pos(q):
        return qmax-q

    
    
    r = minimize(area,qmax, jac=False, method='slsqp', # slsqp
                        options={'maxiter': 200, 'disp': False },#, # maximum iteration number
                        constraints=(
                            {'type':'ineq','fun': q_limit},
                            {'type':'ineq','fun': q_limit_pos}
                                    ))
    #print(r.x,'\n')
    (q,dq)=(r.x[0],sqrt(2*abs(acc)*( (X_now[1]-r.x[0])  )+dq_target**2 ))
    print(q,dq,'\n')
    return (q,dq)

def new_areas(Xup,q,dq):
    area1=[Xup[0],q,dq,Xup[3]]
    area2=[q,Xup[1],dq,Xup[3]]
    area3=[q,Xup[1],Xup[2],dq]
    return (area1,area2,area3)



# Maximum decelerations

for i in range(100):
    print(L[0])
    accelartion=max_acc(L[0],torque[0])
    acc=accelartion.fun
    
    print('acceleration : ',acc)
    
    (q,dq)=Minimize_area(L[0],acc)
    (new_area1,new_area2,new_area3) = new_areas(L[0],q,dq)

    # L.append(new_area1)
    # L.append(new_area2)
    # a=L[0]
    
    L.append(new_area3)
    L.append(new_area1)
    #L.append(new_area2)
    
    # L.append(a)
    
    R.append([L[0][0],q,L[0][2],dq])
    L.remove(L[0])

# Alternative Maximum acceleration

# step_size=0.05
# LL=np.arange(X_all[0],X_all[1],step_size)
# AA=np.zeros(2*size(LL))

# for i in range(size(LL)-1):
#     accelartion=max_acc([LL[i],LL[i+1],0,0],torque[0])
#     acc=accelartion.fun
    
#     (q,dq)=Minimize_area([LL[i],LL[i+1]],acc)
#     AA[i]=q
#     AA[i+1]=dq
    
# Plot Stuff

(f,ax) = plut.create_empty_figure(1)
LW=4

def square_drawer(qmin,qmax,dqmin,dqmax,color='r--'):
    ax.plot([qmin, qmax], [dqmin, dqmin], color,linewidth=LW);
    ax.plot([qmin, qmax], [dqmax, dqmax], color,linewidth=LW);
    ax.plot([qmin, qmin], [dqmin, dqmax], color,linewidth=LW);
    ax.plot([qmax, qmax], [dqmin, dqmax], color,linewidth=LW)
    
square_drawer(X_all[0],X_all[1],X_all[2],X_all[3])


for s in range(int(size(L)/4)-1):
    square_drawer(L[s+1][0],L[s+1][1],L[s+1][2],L[s+1][3],'y--')
    
for s in range(int(size(R)/4)):
    square_drawer(R[s][0],R[s][1],R[s][2],R[s][3],'g--')
    
#expression = np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )
    
q = np.arange(X_all[0], X_all[1], 0.01);
dq_viab_posE =  np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'b--');
# (f,ax) = plut.create_empty_figure(1)    

# s=0
# while(s<=size(AA)-4):
#     square_drawer(AA[s],AA[s+2],0,AA[s+3],'g--')
#     s=s+2
    
plt.show()