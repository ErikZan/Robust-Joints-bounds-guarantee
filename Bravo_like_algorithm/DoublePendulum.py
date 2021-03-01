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
from itertools import cycle

cycol = cycle('bgrcmk')


#### Data ####

m1=1.5 #*1E-1
m2=1.5
l1=2.0
l2=1.0

g=9.81

torque=[-20.0,20.0]
X_all_1=[0.0,0.5,0.0,2.0]
X_all_2=[0.0,0.5,0.0,2.0]

min_size=20
min_q=X_all_1[1]/min_size
min_dq=X_all_2[3]/min_size

def max_acc_int(Xi1,Xi2,torque):
    
    tau1=torque[0]
    tau2=torque[1]
    
    q1=interval(Xi1[0:2])
    q2=interval(Xi1[2:4])
    dq1=interval(Xi2[0:2])
    dq2=interval(Xi2[2:4])
    
    a_min_1= -(-tau1+ m2*g*imath.sin(q2)*imath.cos(q1+q2)  -m2*imath.sin(q1-q2)*(l1*dq1*imath.cos(q1-q2)+l2*dq2**2) - (m1+m2)*g*imath.sin(q1) ) / (l1*(m1+m2*(imath.sin(q1-q2)*imath.sin(q1-q2)) ))
    a_max_1= -(-tau2+ m2*g*imath.sin(q2)*imath.cos(q1+q2)  -m2*imath.sin(q1-q2)*(l1*dq1*imath.cos(q1-q2)+l2*dq2**2) - (m1+m2)*g*imath.sin(q1) ) / (l1*(m1+m2*(imath.sin(q1-q2)*imath.sin(q1-q2)) ))
    
    a_min_2= (-tau1  +(m1+m2)*(l1*dq1**2*imath.sin(q1-q2) -g*imath.sin(q2)+g*imath.sin(q1)*imath.cos(q1-q2)+m2*l2*dq2**2*imath.sin(q1-q2)*imath.cos(q1-q2) )   ) / (l1*(m1+m2*(imath.sin(q1-q2)*imath.sin(q1-q2)))) 
    a_max_2= (-tau2  +(m1+m2)*(l1*dq1**2*imath.sin(q1-q2) -g*imath.sin(q2)+g*imath.sin(q1)*imath.cos(q1-q2)+m2*l2*dq2**2*imath.sin(q1-q2)*imath.cos(q1-q2) )   ) / (l1*(m1+m2*(imath.sin(q1-q2)*imath.sin(q1-q2)))) 
    
    print(q1,q2,tau1,tau2,a_min_1,a_max_1,a_min_2,a_max_2)
    
    return (a_min_1,a_max_1,a_min_2,a_max_2)

max_acc_int(X_all_1,X_all_2,torque)

L=[X_all_1]
L2=[X_all_2]

L13 =[]
L11 =[]
L23 =[]
L21 =[]
R=[]
i=0
print(L)
while ( L != []):
    i=i+1
    while ( L2 != []):
        accelartion=max_acc_int(L[0],L2[0],torque)
        acc=np.max(accelartion[0])   # .fun
        #Saved_acc.append(acc)
        print('acceleration : ',acc,'\n')
        
        (q,dq)=Minimize_area(L[0],acc)
        (new_area1,new_area3) = new_areas_2(L[0],q,dq)
            
        # L.append(new_area1)   
        # L.append(new_area3)
        
        L11.append(new_area1)   
        L13.append(new_area3)
        R.append([L[0][0],q,L[0][2],dq])
        
        acc=np.max(accelartion[2])
        (q,dq)=Minimize_area(L2[0],acc)
        (new_area1,new_area3) = new_areas_2(L2[0],q,dq)
            
        # L.append(new_area1)   
        # L.append(new_area3)
        
        L21.append(new_area1)   
        L23.append(new_area3)
    
        R.append([L2[0][0],q,L2[0][2],dq])

        while ( L11 !=[]):
            
            while( L13 !=[]):
                
                while( L21 !=[]):
                    
                    while( L23 !=[]): 
                        print("we are in L3 ", i , L23[0])
                        accelartion=max_acc_int(L13[0],L23[0],torque)
                        acc=np.max(accelartion[2])   
                        (q,dq)=Minimize_area(L23[0],acc)
                        (new_area1,new_area3) = new_areas_2(L23[0],q,dq)
                        
                        L21.append(new_area1)   
                        L23.append(new_area3)
                        R.append([L23[0][0],q,L23[0][2],dq])
                        
                        
                        if (abs(R[-1][1]-R[-1][0])<=min_q or abs(R[-1][3]-R[-1][2])<=min_dq):
                            R.remove(R[-1])
                            if (new_area3 in L23):
                                L21.remove(new_area1)
                                L23.remove(new_area3) 
                                #Saved_acc.remove(acc)
                    
                        if (new_area3 in L23):
                            if ((abs(new_area1[1]-new_area1[0])<=min_q or abs(new_area1[3]-new_area1[2])<=min_dq) or (abs(new_area3[1]-new_area3[0])<=min_q or abs(new_area3[3]-new_area3[2])<=min_dq)):
                                R.remove(R[-1])
                                L21.remove(new_area1)
                                L23.remove(new_area3) 
                                #Saved_acc.remove(acc)
                            
                        if (L23 != []):    
                            L23.remove(L23[0])
                        L_reorder(L23)
                        L_reorder(L21)  
                    
                    # if (R[-1][3] >= L1[0][2]+1E-5):
                    #     R_ext.append([ L1[0][0],L1[0][1],L1[0][2],R[-1][3] ] )
                    #     L1[0]= [L1[0][0], R[-1][1], R[-1][3] ,X_all[3]] 
                   
                    accelartion=max_acc_int(L13[0],L21[0],torque)
                    acc=np.max(accelartion[2])   
                    (q,dq)=Minimize_area(L21[0],acc)
                    (new_area1,new_area3) = new_areas_2(L21[0],q,dq)
                    
                    L21.append(new_area1)   
                    L23.append(new_area3)
                    R.append([L21[0][0],q,L21[0][2],dq])
                    
                    
                    if (abs(R[-1][1]-R[-1][0])<=min_q or abs(R[-1][3]-R[-1][2])<=min_dq):
                        R.remove(R[-1])
                        if (new_area3 in L21):
                            L21.remove(new_area1)
                            L23.remove(new_area3) 
                            #Saved_acc.remove(acc)
                
                    if (new_area3 in L21):
                        if ((abs(new_area1[1]-new_area1[0])<=min_q or abs(new_area1[3]-new_area1[2])<=min_dq) or (abs(new_area3[1]-new_area3[0])<=min_q or abs(new_area3[3]-new_area3[2])<=min_dq)):
                            R.remove(R[-1])
                            L21.remove(new_area1)
                            L23.remove(new_area3) 
                            #Saved_acc.remove(acc)
                        
                    if (L21 != []):    
                        L21.remove(L21[0])
                    L_reorder(L23)
                    L_reorder(L21)  
                    
                accelartion=max_acc_int(L13[0],L2[0],torque)
                acc=np.max(accelartion[0])    
                (q,dq)=Minimize_area(L13[0],acc)
                (new_area1,new_area3) = new_areas_2(L13[0],q,dq)
                
                L21.append(new_area1)   
                L23.append(new_area3)
                R.append([L13[0][0],q,L13[0][2],dq])
                
                
                if (abs(R[-1][1]-R[-1][0])<=min_q or abs(R[-1][3]-R[-1][2])<=min_dq):
                    R.remove(R[-1])
                    if (new_area3 in L13):
                        L21.remove(new_area1)
                        L23.remove(new_area3) 
                        #Saved_acc.remove(acc)
            
                if (new_area3 in L13):
                    if ((abs(new_area1[1]-new_area1[0])<=min_q or abs(new_area1[3]-new_area1[2])<=min_dq) or (abs(new_area3[1]-new_area3[0])<=min_q or abs(new_area3[3]-new_area3[2])<=min_dq)):
                        R.remove(R[-1])
                        L21.remove(new_area1)
                        L23.remove(new_area3) 
                        #Saved_acc.remove(acc)
                    
                if (L13 != []):    
                    L13.remove(L13[0])
                L_reorder(L23)
                L_reorder(L21)
                
            accelartion=max_acc_int(L11[0],L2[0],torque)
            acc=np.max(accelartion[0])    
            (q,dq)=Minimize_area(L11[0],acc)
            (new_area1,new_area3) = new_areas_2(L11[0],q,dq)
            
            L21.append(new_area1)   
            L23.append(new_area3)
            R.append([L11[0][0],q,L11[0][2],dq])
            
            
            if (abs(R[-1][1]-R[-1][0])<=min_q or abs(R[-1][3]-R[-1][2])<=min_dq):
                R.remove(R[-1])
                if (new_area3 in L11):
                    L11.remove(new_area1)
                    L13.remove(new_area3) 
                    #Saved_acc.remove(acc)
        
            if (new_area3 in L11):
                if ((abs(new_area1[1]-new_area1[0])<=min_q or abs(new_area1[3]-new_area1[2])<=min_dq) or (abs(new_area3[1]-new_area3[0])<=min_q or abs(new_area3[3]-new_area3[2])<=min_dq)):
                    R.remove(R[-1])
                    L11.remove(new_area1)
                    L13.remove(new_area3) 
                    #Saved_acc.remove(acc)
                
            if (L11 != []):    
                L11.remove(L11[0])
            L_reorder(L23)
            L_reorder(L21)
                            
            if (i>=10):
                print('Not converging: exit at ',i,' iteration')
                break
            
        
    L.remove(L[0])
    L.remove(L2[0])
    
    
    
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

square_drawer(X_all_1[0],X_all_1[1],X_all_1[2],X_all_1[3],ax1)


# for s in range(int(size(L)/4)-1):
#     square_drawer(L[s+1][0],L[s+1][1],L[s+1][2],L[s+1][3],ax1,'y--')  # yellow positive square
    
    
for s in range(int(size(R)/4)):
    square_drawer(R[s][0],R[s][1],R[s][2],R[s][3],ax1,'g--')
#    ax1.annotate(str(Saved_acc[s]), xy=(R[s][0]+( R[s][1]-R[s][0])/2, R[s][2]+( R[s][3]-R[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
    ax1.annotate(str(s), xy=(R[s][0]+( R[s][1]-R[s][0])/2, R[s][2]+( R[s][3]-R[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
    
for s in range(int(size(R_ext)/4)):
    square_drawer(R_ext[s][0],R_ext[s][1],R_ext[s][2],R_ext[s][3],ax1,'y--') 
    ax1.annotate(str(s)+'ext', xy=(R_ext[s][0]+( R_ext[s][1]-R_ext[s][0])/2, R_ext[s][2]+( R_ext[s][3]-R_ext[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))
    
#expression = np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )

# Expression with q => It seems to tend to this curve! must verify why

q = np.arange(X_all[0], X_all[1]+0.005, 0.005);
# dq_viab_posE =  np.sqrt(-2*m*(m*np.sin(q)*g*l*q -m*np.sin(q)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
# line_viabE, = ax1.plot(q, dq_viab_posE, 'b--');

### same name for 

qmax=X_all[1]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= np.sqrt(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[0]+X_all[1]*torque[0])  )/(m*l)
line_viabE, = ax1.plot(q, dq_viab_posE, 'p--');

#### maximum Aceleration Plot
#(f,ax) = plut.create_empty_figure(1)
square_drawer(X_all[0],X_all[1],0,-X_all[3],ax1)

# for s in range(int(size(L_neg)/4)-1):
#     square_drawer(L_neg[s+1][0],L_neg[s+1][1],L_neg[s+1][2],L_neg[s+1][3],ax1,'y--') # yellow negative square
    
for s in range(int(size(R_neg)/4)):
    square_drawer(R_neg[s][0],R_neg[s][1],R_neg[s][2],R_neg[s][3],ax1,'b--')
    #ax1.annotate(str(Saved_acc_neg[s]), xy=(R_neg[s][0]+( R_neg[s][1]-R_neg[s][0])/2, R_neg[s][2]+( R_neg[s][3]-R_neg[s][2])/2 ),size=8) # int((10*R_neg[s][3]-10*R_neg[s][2])/(0.5*R_neg[s][3]))
    ax1.annotate(str(s), xy=(R_neg[s][0]+( R_neg[s][1]-R_neg[s][0])/2, R_neg[s][2]+( R_neg[s][3]-R_neg[s][2])/2 ),size=8) # int((10*R[s][3]-10*R[s][2])/(0.5*R[s][3]))

# same name for qmax evn if is it different
qmax=X_all[1]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax1.plot(-q+qmax, dq_viab_posE, 'o--');


qmax=X_all[0]
dq_viab_posE= -np.sqrt(abs(-2*m*(m*np.sin(qmax)*g*l*q -m*np.sin(qmax)*g*l*X_all[1]-q*torque[1]+X_all[1]*torque[1]))  )/(m*l)
line_viabE, = ax1.plot(-q+X_all[1], dq_viab_posE, 'p--');

ax1.set_ylabel(r'$\dot{q}$ $[\frac{rad}{s}]$');
ax1.set_xlabel(r'$q$ $[rad]$');

lege1=mpatches.Patch(color='red',alpha=0.25,label='Not Viable');
lege2=mpatches.Patch(color='green',alpha=0.25,label='Viable');
lege3=mpatches.Patch(color='green',ls='--',label='Inner approx');
ax1.legend(handles=[lege1,lege2,lege3], loc='upper center',bbox_to_anchor=(0.5, 1.0),bbox_transform=plt.gcf().transFigure,ncol=5,fontsize=30 );
       
# lege1=mpatches.Patch(color='yellow',label='Integrated Polytope');
# lege3=mpatches.Patch(color='green',label='Viable Polytope');

# ax1.legend(handles=[lege1,lege3], loc='lower center',bbox_to_anchor=(0.5, 1.0),bbox_transform=plt.gcf().transFigure,ncol=5,fontsize=30 );
ax1.set_xlim([0.0,0.5])
ax1.set_ylim([0.0,2.0])
plt.show()