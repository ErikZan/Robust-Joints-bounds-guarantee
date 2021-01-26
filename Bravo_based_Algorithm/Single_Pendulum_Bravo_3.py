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

#### Data ####
m=1.5 
l=2.0
g=9.81
torque=[-15.0,15.0]
X_all=[-0.0,0.5,0.0,2.0]
#### Variables ####
L=[X_all]
R=[]
L2=[]
R2=[]
L_neg=[[0.0,0.5,0.0,-2.0]]
R_neg=[]
L2_neg=[[0.0,0.5,0.0,-2.0]]
R2_neg=[]
Saved_acc=[]
Saved_acc_neg=[]

#### Functions ####

def max_acc_int(Xi,tau):
    
    tau=interval([-tau,tau])
    q=interval(Xi[0:2])
    
    a_min=(tau[0][0]-m*l*g*imath.sin(q)[0][0])/(m*l**2)
    a_max=(tau[0][1]-m*l*g*imath.sin(q)[0][1])/(m*l**2)
    
    print(q,tau,a_min,a_max)
    
    return (a_min,a_max)


def Minimize_area(X_now,acc,plot_trigger=False):
    
    (qmin,qmax,dq_target)=(X_now[0],X_now[1],X_now[2])
    
    def area(q):
        '''
        Area defined as q*dq where dq is already in the form of 
        dq=sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))) 
        and limits are given only on positions
        '''
        return -((q-qmin)*(  sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))+1E-6))  +  dq_target**2  ) # usata fino a poco fa
        #return -((q-qmin)*sqrt(  )
    
    def area2(q):
        '''
        Area defined as q*dq where dq is already in the form of 
        dq=sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))) 
        and limits are given only on positions
        '''
        return -((qmax-q)*(  sqrt(2*abs(acc)*(q-qmin)+1E-6))  +  dq_target**2  ) # usata fino a poco fa
        #return -((q-qmin)*sqrt(  )
    
    def q_limit(q):
        return q-qmin

    def q_limit_pos(q):
        return qmax-q

    if (acc>=0):
        r = minimize(area,qmax, jac=False, method='slsqp', # slsqp
                            options={'maxiter': 200, 'disp': plot_trigger },#, # maximum iteration number
                            constraints=(
                                {'type':'ineq','fun': q_limit},
                                {'type':'ineq','fun': q_limit_pos}
                                        ))
        (q,dq)=(r.x[0],np.sign(acc)*sqrt(2*abs(acc)*( (X_now[1]-r.x[0])  )+dq_target**2 ))
        print('Area coordinates',q,dq,'\n')
    else:
        r = minimize(area2,qmax, jac=False, method='slsqp', # slsqp
                            options={'maxiter': 200, 'disp': plot_trigger },#, # maximum iteration number
                            constraints=(
                                {'type':'ineq','fun': q_limit},
                                {'type':'ineq','fun': q_limit_pos}
                                        ))
        (q,dq)=(r.x[0],np.sign(acc)*sqrt(2*abs(acc)*( (r.x[0]-X_now[0])  )+dq_target**2 ))
        print('Area coordinates',q,dq,'\n')
    
    return (q,dq)

def q_area_finder(dq,qmax,acc,dq_target):
    
    q = qmax - (dq**2-dq_target**2)/(2*acc)
    
    return q

def new_areas(Xup,q,dq):
    area1=[Xup[0],q,dq,Xup[3]]
    area2=[q,Xup[1],dq,Xup[3]]
    area3=[q,Xup[1],Xup[2],dq]
    return (area1,area2,area3)

def new_areas_neg(Xup,q,dq):
    area1=[q,Xup[1],dq,Xup[3]]
    area2=[q,Xup[1],dq,Xup[3]] # not correct
    area3=[Xup[0],q,Xup[2],dq]
    return (area1,area2,area3)

# Maximum decelerations
print('#'*60)
print('############ Maximum Deceleration ############ ')
print('#'*60)

n=100
Q_trig=0
trigger=True
for i in range(n):
    print('Area in verifica:',L[0],'\n')
    accelartion=max_acc_int(L[0],torque[0])
    acc=-accelartion[0]   # .fun
    Saved_acc.append(acc)
    print('acceleration : ',acc,'\n')
    
    (q,dq)=Minimize_area(L[0],acc)
    (new_area1,new_area2,new_area3) = new_areas(L[0],q,dq)
    L.append(new_area1)
    L.append(new_area3)
    #L.append(new_area2)
    
    #### new area 2 ###
    # accelartion=max_acc_int(new_area3,torque[0])
    # acc=-accelartion[0]
    # q_area2=q_area_finder(dq,new_area3[1],acc,new_area3[2])
    # new_area2=[new_area3[0],q_area2,dq,new_area1[3]]
    # L.append(new_area2)
    
    #########
    
    R.append([L[0][0],q,L[0][2],dq])
    ####
    
    if (size(R)/4>=3 and (size(R)/4)%2 != 0 and Q_trig==0 ): # with Q_trig make only one timestep
       
        L2.append([R[-2][1],R[-1][1],R[-1][3],R[-2][3]])
        
        if (trigger==True ): # with Q_trig make only one area, the first one
            Q_trig=1
            L.append([R[-2][1],R[-1][1],R[-1][3],R[-2][3]])
        
        L.append([R[-2][1],R[-1][1],R[-1][3],R[-2][3]])
        accelartion=max_acc_int(L2[0],torque[0])
        acc=-accelartion[0]   # .fun
        Saved_acc.append(acc)
        (q,dq)=Minimize_area(L2[0],acc)
        if (q>=R[-1][1]):
            q=R[-1][1]
        R2.append([L2[0][0],q,L2[0][2],dq])
        L2.remove(L2[0])
        
    ####
    L.remove(L[0])

# Maximum acceleration
print('#'*60)
print('############ Maximum Acceleration ############')
print('#'*60)
Q_trig=0
for i in range(n):
    print('Area in verifica:',L_neg[0],'\n')
    accelartion=max_acc_int(L_neg[0],torque[1])
    acc=-accelartion[1]
    Saved_acc_neg.append(acc)
    print('acceleration : ',acc,'\n')
    
    (q,dq)=Minimize_area(L_neg[0],acc)
    (new_area1,new_area2,new_area3) = new_areas_neg(L_neg[0],q,dq)

    L_neg.append(new_area1)
    L_neg.append(new_area3)
    R_neg.append([q,L_neg[0][1],L_neg[0][2],dq])
    
    if (size(R_neg)/4>=3 and (size(R_neg)/4)%2 != 0 and Q_trig==0 ): # with Q_trig make only one timestep
       
        L2_neg.append([R_neg[-1][0],R_neg[-2][0],R_neg[-1][2],R_neg[-2][2]])
        
        if (trigger==True ): # with Q_trig make only one area, the first one
            Q_trig=1
            L_neg.append([R_neg[-1][0],R_neg[-2][0],R_neg[-1][2],R_neg[-2][2]])
        
        L_neg.append([R_neg[-1][0],R_neg[-2][0],R_neg[-1][2],R_neg[-2][2]])
        accelartion=max_acc_int(L2_neg[0],torque[1])
        acc=-accelartion[1]   # .fun
        Saved_acc_neg.append(acc)
        (q,dq)=Minimize_area(L2_neg[0],acc)
        
        R_neg.append([q,L2_neg[0][1],L2_neg[0][2],dq])
        L2_neg.remove(L2_neg[0])
    
    
    L_neg.remove(L_neg[0])
    
# Alternative Maximum acceleration
print('#'*60)
print('############ Maximum alternative Acceleration ############\n ')
print('#'*60)

step_size=0.05
LL=np.arange(X_all[0],X_all[1],step_size)
AA=np.zeros(2*size(LL))

for i in range(size(LL)-1):
    accelartion=max_acc_int([LL[i],LL[i+1],0.0,0.0],torque[0])
    acc=accelartion[0]
    print(i,acc)
    (q,dq)=Minimize_area([LL[i],LL[i+1],0.0,0.0],acc)
    AA[i]=q
    AA[i+1]=dq
    
# for i in range(size(LL)-1):
#     accelartion=max_acc([LL[i],LL[i+1],0.0,0.0],torque[1])
#     acc=accelartion.fun
#     print(i,acc)
#     (q,dq)=Minimize_area([LL[i],LL[i+1],0.0,0.0],acc)
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
    ax.annotate(str(Saved_acc[s]), xy=(R[s][0]+( R[s][1]-R[s][0])/2, R[s][2]+( R[s][3]-R[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
    
# for s in range(int(size(R2)/4)):
#     square_drawer(R2[s][0],R2[s][1],R2[s][2],R2[s][3],'b--')
#     ax.annotate(str(Saved_acc[s]), xy=(R2[s][0]+( R2[s][1]-R2[s][0])/2, R2[s][2]+( R2[s][3]-R2[s][2])/2 ),size=8)
    
#expression = np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )
    
q = np.arange(X_all[0], X_all[1]+0.01, 0.01);
dq_viab_posE =  np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'b--');

### same name for 

qmax=X_all[1]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'p--');

####

#(f,ax) = plut.create_empty_figure(1)
square_drawer(X_all[0],X_all[1],0,-X_all[3])

for s in range(int(size(L_neg)/4)-1):
    square_drawer(L_neg[s+1][0],L_neg[s+1][1],L_neg[s+1][2],L_neg[s+1][3],'y--')
    
for s in range(int(size(R_neg)/4)):
    square_drawer(R_neg[s][0],R_neg[s][1],R_neg[s][2],R_neg[s][3],'g--')
    ax.annotate(str(Saved_acc_neg[s]), xy=(R_neg[s][0]+( R_neg[s][1]-R_neg[s][0])/2, R_neg[s][2]+( R_neg[s][3]-R_neg[s][2])/2 ),size=8) # int((10*R_neg[s][3]-10*R_neg[s][2])/(0.5*R_neg[s][3]))

# same name for qmax evn if is it different
qmax=X_all[1]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax.plot(-q+qmax, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax.plot(-q+X_all[1], dq_viab_posE, 'p--');

# Acceleration values
# (f,ax) = plut.create_empty_figure(1)
# ax.plot(q,(torque[0]-m*l*g*np.sin(q))/(m*l**2),color='orange')
# ax.plot(q,(torque[1]-m*l*g*np.sin(q))/(m*l**2),color='blue')
# lege1=mpatches.Patch(color='orange',label='Deceleration');
# lege2=mpatches.Patch(color='blue',label='Acceleration');
# ax.legend(handles=[lege1,lege2], loc='upper center',bbox_to_anchor=(0.5, 1.0),
#                     bbox_transform=plt.gcf().transFigure,ncol=5,fontsize=30 );
(f,ax) = plut.create_empty_figure(1)
q = np.arange(X_all[0], X_all[1]+0.01, 0.01);
dq_viab_posE =  np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'b--');

### same name for 

qmax=X_all[1]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax.plot(q, dq_viab_posE, 'p--');
for s in range(int(size(R2)/4)):
    square_drawer(R2[s][0],R2[s][1],R2[s][2],R2[s][3],'b--')
    ax.annotate(str(Saved_acc[s]), xy=(R2[s][0]+( R2[s][1]-R2[s][0])/2, R2[s][2]+( R2[s][3]-R2[s][2])/2 ),size=8)
(f,ax) = plut.create_empty_figure(1)    

s=0
while(s<=size(AA)-4):
    square_drawer(AA[s],AA[s+2],0,AA[s+3],'g--')
    s=s+2
    
plt.show()