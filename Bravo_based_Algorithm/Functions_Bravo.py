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

#### Functions ####

def max_acc_int(Xi,tau,m,l,g):
    
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

def R_reorder(R):
    Q=R
    Q.sort(key = lambda x: x[1],reverse=True) 
    return Q

def q_area_finder(dq,qmax,acc,dq_target):
    
    q = qmax - (dq**2-dq_target**2)/(2*acc)
    
    return q

def new_areas(Xup,q,dq):
    area1=[Xup[0],q,dq,Xup[3]]
    area2=[q,Xup[1],dq,Xup[3]]
    area3=[q,Xup[1],Xup[2],dq]
    return (area1,area2,area3)

def new_areas_with_tag(Xup,q,dq):
    area1=[Xup[0],q,dq,Xup[3],Xup[4]+'1']
    area2=[q,Xup[1],dq,Xup[3],Xup[4]+'2']
    area3=[q,Xup[1],Xup[2],dq,Xup[4]+'3']
    return (area1,area2,area3)

def new_areas_neg(Xup,q,dq):
    area1=[q,Xup[1],dq,Xup[3]]
    area2=[q,Xup[1],dq,Xup[3]] # not correct
    area3=[Xup[0],q,Xup[2],dq]
    return (area1,area2,area3)

def new_areas_neg_with_tag(Xup,q,dq):
    area1=[q,Xup[1],dq,Xup[3],Xup[4]+'1']
    area2=[q,Xup[1],dq,Xup[3],Xup[4]+'2'] # not correct
    area3=[Xup[0],q,Xup[2],dq,Xup[4]+'3']
    return (area1,area2,area3)

def new_areas_2(Xup,q,dq):
    area1=[Xup[0],q,dq,Xup[3]]
    area3=[q,Xup[1],Xup[2],Xup[3]]
    return (area1,area3)    

def new_areas_neg_2(Xup,q,dq):
    area1=[q,Xup[1],dq,Xup[3]]
    area3=[Xup[0],q,Xup[2],Xup[3]]
    return (area1,area3)