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
from Functions_Bravo import *
#### Data ####

m=1.5 #*1E-1
l=2.0
g=9.81
torque=[-15.0,15.0]
X_all=[-0.0,0.5,0.0,2.0,'5']

#### Variables ####

L=[X_all]
R=[]
L2=[]
R2=[]
L_neg=[[0.0,0.5,0.0,-2.0,'5']]
R_neg=[]
L2_neg=[[0.0,0.5,0.0,-2.0]]
R2_neg=[]
Saved_acc=[]
Saved_acc_neg=[]
area_container_1=[]
area_container_2=[]


# Maximum decelerations
print('#'*60)
print('############ Maximum Deceleration ############ ')
print('#'*60)

n=80

how_many=0
for i in range(n):
    print('Area in verifica:',L[0],'\n')
    accelartion=max_acc_int(L[0],torque[0],m,l,g)
    acc=-accelartion[0]   # .fun
    Saved_acc.append(acc)
    print('acceleration : ',acc,'\n')
    
    (q,dq)=Minimize_area(L[0],acc)
    (new_area1,new_area2,new_area3) = new_areas_with_tag(L[0],q,dq)
    #(new_area1,new_area3) = new_areas_2(L[0],q,dq)
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
    
    R.append([L[0][0],q,L[0][2],dq,L[0][4]])
    for p in range(int(size(R)/5-1)):
        #print('this is ',+ (int(R[p][4])) -(int(R[-1][4])))
        if ((int(R[p][4])) -(int(R[-1][4])) == -2):
            print('Inside the cycle the area ########')
            print(int(R[-1][4]),int(R[p][4]))
            L.append( [ R[p][1], R[-1][1],R[-1][3] ,R[p][3],R[-1][4]+'2'] )
            how_many+=1
            
    ####
    
        
    ####
    L.remove(L[0])

# Maximum acceleration
print('#'*60)
print('############ Maximum Acceleration ############')
print('#'*60)
Q_trig=0
for i in range(n):
    if (L_neg==[]):
        print('interrupted at ',i,' for too small area')
        break
    print('Area in verifica:',L_neg[0],'\n')
    accelartion=max_acc_int(L_neg[0],torque[1],m,l,g)
    acc=-accelartion[1]
    Saved_acc_neg.append(acc)
    print('acceleration : ',acc,'\n')
    
    (q,dq)=Minimize_area(L_neg[0],acc)
    (new_area1,new_area2,new_area3) = new_areas_neg_with_tag(L_neg[0],q,dq)
    #(new_area1,new_area3) = new_areas_neg_2(L_neg[0],q,dq)
    
    L_neg.append(new_area1)
    L_neg.append(new_area3)
    
    print('test on sign',abs(new_area1[1]-new_area1[0]))
    
    if (abs(new_area1[1]-new_area1[0])<=5E-3 or abs(new_area1[2]-new_area1[3])<=1E-2):
        L_neg.remove(new_area1)
    if (abs(new_area3[1]-new_area3[0])<=5E-3 or abs(new_area3[2]-new_area3[3])<=1E-2):
        L_neg.remove(new_area3)
    
    
    
    R_neg.append([q,L_neg[0][1],L_neg[0][2],dq,L_neg[0][4]])
    
    for p in range(int(size(R_neg)/5-1)):
        #print('this is ',+ (int(R[p][4])) -(int(R[-1][4])))
        if ((int(R_neg[p][4])) -(int(R_neg[-1][4])) == -2):
            print('Inside the cycle the area ########')
            print(int(R_neg[-1][4]),int(R_neg[p][4]))
            L_neg.append( [ R_neg[-1][0], R_neg[p][0],R_neg[-1][3] ,R_neg[p][3],R_neg[-1][4]+'2'] )
            how_many+=1
    
    
    L_neg.remove(L_neg[0])
    
# Alternative Maximum acceleration
# Removed , see previous file

############    
# Plot Stuff
############


# Plot Functions
LW=4
# very unreliable method, implement the call of ax after color
def square_drawer(qmin,qmax,dqmin,dqmax,ax,color='r--'):
    ax.plot([qmin, qmax], [dqmin, dqmin], color,linewidth=LW);
    ax.plot([qmin, qmax], [dqmax, dqmax], color,linewidth=LW);
    ax.plot([qmin, qmin], [dqmin, dqmax], color,linewidth=LW);
    ax.plot([qmax, qmax], [dqmin, dqmax], color,linewidth=LW)
    return

def polygon_drawer(Q,ax,color='r--'):
    for i in range(int(size(Q)/5)-1):
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
square_drawer(X_all[0],X_all[1],X_all[2],X_all[3],ax1)


for s in range(int(size(L)/5)-1):
    square_drawer(L[s+1][0],L[s+1][1],L[s+1][2],L[s+1][3],ax1,'y--')
    
    
for s in range(int(size(R)/5)):
    square_drawer(R[s][0],R[s][1],R[s][2],R[s][3],ax1,'g--')
    ax1.annotate(str(Saved_acc[s]), xy=(R[s][0]+( R[s][1]-R[s][0])/2, R[s][2]+( R[s][3]-R[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
    
# for s in range(int(size(R2)/4)):
#     square_drawer(R2[s][0],R2[s][1],R2[s][2],R2[s][3],ax,'b--')
#     ax.annotate(str(Saved_acc[s]), xy=(R2[s][0]+( R2[s][1]-R2[s][0])/2, R2[s][2]+( R2[s][3]-R2[s][2])/2 ),size=8)
    
#expression = np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )

# Expression with q => It seems to tend to this curve! must verify why

q = np.arange(X_all[0], X_all[1]+0.01, 0.01);
dq_viab_posE =  np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'b--');

### same name for 

qmax=X_all[1]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'p--');

#### POlygon Draw ####
Q=R_reorder(R)
polygon_drawer(Q,ax1)
####              ####

####
#(f,ax) = plut.create_empty_figure(1)
square_drawer(X_all[0],X_all[1],0,-X_all[3],ax1)

for s in range(int(size(L_neg)/5)-1):
    square_drawer(L_neg[s+1][0],L_neg[s+1][1],L_neg[s+1][2],L_neg[s+1][3],ax1,'y--')
    
for s in range(int(size(R_neg)/5)):
    square_drawer(R_neg[s][0],R_neg[s][1],R_neg[s][2],R_neg[s][3],ax1,'g--')
    ax1.annotate(str(Saved_acc_neg[s]), xy=(R_neg[s][0]+( R_neg[s][1]-R_neg[s][0])/2, R_neg[s][2]+( R_neg[s][3]-R_neg[s][2])/2 ),size=8) # int((10*R_neg[s][3]-10*R_neg[s][2])/(0.5*R_neg[s][3]))

# same name for qmax evn if is it different
qmax=X_all[1]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax1.plot(-q+qmax, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax1.plot(-q+X_all[1], dq_viab_posE, 'p--');

# Acceleration values
# (f,ax) = plut.create_empty_figure(1)
# ax.plot(q,(torque[0]-m*l*g*np.sin(q))/(m*l**2),color='orange')
# ax.plot(q,(torque[1]-m*l*g*np.sin(q))/(m*l**2),color='blue')
# lege1=mpatches.Patch(color='orange',label='Deceleration');
# lege2=mpatches.Patch(color='blue',label='Acceleration');
# ax.legend(handles=[lege1,lege2], loc='upper center',bbox_to_anchor=(0.5, 1.0),
#                     bbox_transform=plt.gcf().transFigure,ncol=5,fontsize=30 );

# R2 area plotted are here for clarity

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
    square_drawer(R2[s][0],R2[s][1],R2[s][2],R2[s][3],ax,'b--')
    ax.annotate(str(Saved_acc[s]), xy=(R2[s][0]+( R2[s][1]-R2[s][0])/2, R2[s][2]+( R2[s][3]-R2[s][2])/2 ),size=8)
  
# methods with range  and not area 
# Removed see previous files

#######################################
############# Test #############
#######################################
# dt=0.01
# (q0,dq0)=(Q[30][0],Q[30][2])
# #(q0,dq0)=(R[1][1],R[1][3])
# print(q0,dq0)

# n=200

# q=np.zeros(n)
# dq=np.zeros(n)

# q[0]=q0
# dq[0]=dq0

# for i in range(n-1):
#     q[i+1]=q[i]+dt*dq[i]+dt**2*((torque[0]-m*l*g*np.sin(q[i]))/(m*l**2))/2
#     dq[i+1]=dq[i]+dt*((torque[0]-m*l*g*np.sin(q[i]))/(m*l**2))
    
#     if (dq[i+1]<=0):
#         break

# ax1.plot(q[:], dq[:], 'g--');



# def which_R_contains(R,q,dq,sign_select='Positive'):
    
#     if (sign_select=='Positive'):
        
#         for i in range(int(size(R)/5)):
#             if ( (q>=R[i][0] and q<=R[i][1]) and (dq>=R[i][2] and dq<=R[i][3]) ):
#                 k=i
#                 return k
            
#         k='q,dq not found'
#         print(k)
#         return k
    
#     if (sign_select=='Negative'):
#         for i in range(int(size(R)/5)):
#             if ( (q>=R[i][0] and q<=R[i][1]) and (dq>=R[i][3] and dq<=R[i][2]) ):
#                 k=i
#                 return k
            
#         k='q,dq not found'
#         print(k)
#         return k

# for i in range(n-1):
    
#     # Verify in which Rectangle q,dq are and apply relative acceleration
#     k=which_R_contains(R,q[i],dq[i])
#     print(q[i],dq[i])
    
#     if (dq[i]>=0):
#         q[i+1]=q[i]+dt*dq[i]+dt**2*(-Saved_acc[k])/2
#         dq[i+1]=dq[i]+dt*(-Saved_acc[k])
    
#     if (k=='q,dq not found'):
#         k=which_R_contains(R_neg,q[i],dq[i],sign_select='Negative')
#         print(q[i],dq[i])
    
#         q[i+1]=q[i]+dt*dq[i]+dt**2*(-Saved_acc_neg[k])/2
#         dq[i+1]=dq[i]+dt*(-Saved_acc_neg[k])
    

# #(f,ax) = plut.create_empty_figure(1)
# ax1.plot(q[:], dq[:], 'b--');

# Plot with the initial acceleration mantained for all the time

# switch=0
# for i in range(n-1):
    
#     # Verify in which Rectangle q,dq are and apply relative acceleration
#     if (switch==0):
#         k=which_R_contains(R,q[i],dq[i])
#         switch=1
        
#     print(q[i],dq[i])
    
#     q[i+1]=q[i]+dt*dq[i]+dt**2*(-Saved_acc[k])/2
#     dq[i+1]=dq[i]+dt*(-Saved_acc[k])
    
#     if(dq[i+1]) <=0:
#         break
# ax1.plot(q[:], dq[:], 'r--');
  
plt.show()
