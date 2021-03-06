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
from Functions_2 import *
#### Data ####

m=1.5 #*1E-1
l=2.0
g=9.81
torque=[-15.0,15.0]
X_all=[0.0,0.5,0.0,2.0]

min_size=100
min_q=X_all[1]/min_size
min_dq=X_all[3]/min_size

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
area_container_1=[]
area_container_2=[]

#### Functions ####
def max_acc_int(Xi,tau):
    
    tau=interval([-tau,tau])
    q=interval(Xi[0:2])
    
    a_min=(tau[0][0]-m*l*g*imath.sin(q)[0][0])/(m*l**2)
    a_max=(tau[0][1]-m*l*g*imath.sin(q)[0][1])/(m*l**2)
    
    print(q,tau,a_min,a_max)
    
    return (a_min,a_max)
#### ####

UPDATE_SPACE=0
if (UPDATE_SPACE==True):
    division=100
    dt=0.001
    np.save('data.npy',np.array([torque,X_all,m,l,division,dt])) 
    np.save('data_neg.npy',np.array([torque,L_neg[0],m,l,division,dt]))
    os.system("python3 /home/erik/Desktop/Thesis/Github/Robust\ Joints\ bounds\ guarantee/Bravo_based_Algorithm/check_points.py " )


# Maximum decelerations
print('#'*60)
print('############ Maximum Deceleration ############ ')
print('#'*60)

L3 =[]
L1 =[]
i=0
print(L)
while ( L != []):
    print('Area in verifica:',L[0],'\n')
    i=i+1
    print('iterazione n ',i)
    accelartion=max_acc_int(L[0],torque[0])
    acc=-accelartion[0]   # .fun
    Saved_acc.append(acc)
    print('acceleration : ',acc,'\n')
    
    (q,dq)=Minimize_area(L[0],acc)
    (new_area1,new_area3) = new_areas_2(L[0],q,dq)
        
    # L.append(new_area1)   
    # L.append(new_area3)
    
    L1.append(new_area1)   
    L3.append(new_area3)
    R.append([L[0][0],q,L[0][2],dq])
    
    while ( L1 !=[]):
        
        while( L3 !=[]):
            print("we are in L3 ", i , L3[0])
            accelartion=max_acc_int(L3[0],torque[0])
            acc=-accelartion[0]   # .fun
            Saved_acc.append(acc)
            print('acceleration : ',acc,'\n')
            
            (q,dq)=Minimize_area(L3[0],acc)
            #(new_area1,new_area2,new_area3) = new_areas(L[0],q,dq)
            (new_area1,new_area3) = new_areas_2(L3[0],q,dq)
            
            L1.append(new_area1)   
            L3.append(new_area3)
            R.append([L3[0][0],q,L3[0][2],dq])
            
            if (abs(R[-1][1]-R[-1][0])<=min_q or abs(R[-1][3]-R[-1][2])<=min_dq):
                R.remove(R[-1])
                if (new_area3 in L3):
                    L1.remove(new_area1)
                    L3.remove(new_area3) 
                    Saved_acc.remove(acc)
        
            if (new_area3 in L3):
                if ((abs(new_area1[1]-new_area1[0])<=min_q or abs(new_area1[3]-new_area1[2])<=min_dq) or (abs(new_area3[1]-new_area3[0])<=min_q or abs(new_area3[3]-new_area3[2])<=min_dq)):
                    R.remove(R[-1])
                    L1.remove(new_area1)
                    L3.remove(new_area3) 
                    Saved_acc.remove(acc)
                
            if (L3 != []):    
                L3.remove(L3[0])
            L_reorder(L3)
            L_reorder(L1)  
               
        accelartion=max_acc_int(L1[0],torque[0])
        acc=-accelartion[0]   # .fun
        Saved_acc.append(acc)
        print('acceleration : ',acc,'\n')
        
        (q,dq)=Minimize_area(L1[0],acc)
        #(new_area1,new_area2,new_area3) = new_areas(L[0],q,dq)
        (new_area1,new_area3) = new_areas_2(L1[0],q,dq)
        
        L1.append(new_area1)   
        L3.append(new_area3)
        
        R.append([L1[0][0],q,L1[0][2],dq])
        i=i+1
        if (abs(R[-1][1]-R[-1][0])<=min_q or abs(R[-1][3]-R[-1][2])<=min_dq):
            R.remove(R[-1])
            if (new_area1 in L1):
                L1.remove(new_area1)
                L3.remove(new_area3) 
                Saved_acc.remove(acc)
    
        if (new_area1 in L1):
            if ((abs(new_area1[1]-new_area1[0])<=min_q or abs(new_area1[3]-new_area1[2])<=min_dq) or (abs(new_area3[1]-new_area3[0])<=min_q or abs(new_area3[3]-new_area3[2])<=min_dq)):
                R.remove(R[-1])
                L1.remove(new_area1)
                L3.remove(new_area3) 
                Saved_acc.remove(acc)
        
        if (L1 != []):    
            L1.remove(L1[0])      
        L_reorder(L3)
        L_reorder(L1)
        if (i>=10000):
            print('Not converging: exit at ',i,' iteration')
            break
        
    L.remove(L[0])

    
    #print(L_neg)
############    
# Plot Stuff
############


# Plot Functions
LW=4

def square_drawer(qmin,qmax,dqmin,dqmax,ax,color='r--'):
    ax.plot([qmin, qmax], [dqmin, dqmin], color,linewidth=LW);
    ax.plot([qmin, qmax], [dqmax, dqmax], color,linewidth=LW);
    ax.plot([qmin, qmin], [dqmin, dqmax], color,linewidth=LW);
    ax.plot([qmax, qmax], [dqmin, dqmax], color,linewidth=LW)
    return

def polygon_drawer(Q,ax,color='r--'):
    for i in range(int(size(Q)/4)-1):
        q1=Q[i][1]
        q2=Q[i+1][1]
        q3=Q[i][2]  
        q4=Q[i+1][2]
        
        if (Q[i][3]>=q4):
            q4=Q[i][3]
            
        if (Q[i-1][3]>=Q[1][2]):
            q3=Q[i-1][3]
       
        ax.plot( [q1,q2],[q3,q4], color,linewidth=LW )
                
    return


(f,ax1) = plut.create_empty_figure(1)
    
PP=np.load('q_viable.npy')
RR=np.load('q_not_viable.npy')
plot(PP[:,0],PP[:,1],color="green",alpha=0.25)      
plot(RR[:,0],RR[:,1],color="red",alpha=0.25) 


PP_neg=np.load('q_viable_neg.npy')
RR_neg=np.load('q_not_viable_neg.npy')
plot(PP_neg[:,0],PP_neg[:,1],color="green",alpha=0.25)      
plot(RR_neg[:,0],RR_neg[:,1],color="red",alpha=0.25) 

square_drawer(X_all[0],X_all[1],X_all[2],X_all[3],ax1)


# for s in range(int(size(L)/4)-1):
#     square_drawer(L[s+1][0],L[s+1][1],L[s+1][2],L[s+1][3],ax1,'y--')  # yellow positive square
    
    
for s in range(int(size(R)/4)):
    square_drawer(R[s][0],R[s][1],R[s][2],R[s][3],ax1,'g--')
#    ax1.annotate(str(Saved_acc[s]), xy=(R[s][0]+( R[s][1]-R[s][0])/2, R[s][2]+( R[s][3]-R[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
    ax1.annotate(str(s), xy=(R[s][0]+( R[s][1]-R[s][0])/2, R[s][2]+( R[s][3]-R[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
 
#expression = np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )

# Expression with q => It seems to tend to this curve! must verify why

q = np.arange(X_all[0], X_all[1]+0.001, 0.001);
# dq_viab_posE =  np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
# line_viabE, = ax1.plot(q, dq_viab_posE, 'b--');

### same name for 

qmax=X_all[1]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'p--');

####
#(f,ax) = plut.create_empty_figure(1)
square_drawer(X_all[0],X_all[1],0,-X_all[3],ax1)

# for s in range(int(size(L_neg)/4)-1):
#     square_drawer(L_neg[s+1][0],L_neg[s+1][1],L_neg[s+1][2],L_neg[s+1][3],ax1,'y--') # yellow negative square
    
# for s in range(int(size(R_neg)/4)):
#     square_drawer(R_neg[s][0],R_neg[s][1],R_neg[s][2],R_neg[s][3],ax1,'g--')
#     #ax1.annotate(str(Saved_acc_neg[s]), xy=(R_neg[s][0]+( R_neg[s][1]-R_neg[s][0])/2, R_neg[s][2]+( R_neg[s][3]-R_neg[s][2])/2 ),size=8) # int((10*R_neg[s][3]-10*R_neg[s][2])/(0.5*R_neg[s][3]))

# same name for qmax evn if is it different
qmax=X_all[1]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax1.plot(-q+qmax, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax1.plot(-q+X_all[1], dq_viab_posE, 'p--');



       
plt.show()
