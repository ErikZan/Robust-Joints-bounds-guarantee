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

# def Minimize_area(X_now,acc,plot_trigger=False):
    
#     (qmin,qmax,dq_target)=(X_now[0],X_now[1],X_now[2])
    
   
    
#     def area2(q):
#         '''
#         Area defined as q*dq where dq is already in the form of 
#         dq=sqrt(2*abs(acc)*((qmax-qmin)-(q-qmin))) 
#         and limits are given only on positions
#         '''
#         return -((qmax-q)*(  sqrt(2*abs(acc)*(q-qmin)+1E-6))  +  dq_target**2  ) # usata fino a poco fa
#         #return -((q-qmin)*sqrt(  )
    
#     def q_limit(q):
#         return q-qmin

#     def q_limit_pos(q):
#         return qmax-q


#     if (acc>=0):
#         q = (dq_target ** 2 * math.sqrt(2) * (-math.sqrt(2) * dq_target ** 2 / 6 + math.sqrt(2 * dq_target ** 4 + 12 * acc * qmax - 12 * acc * qmin) / 6) + 2 * acc * qmax + acc * qmin) / acc / 3
#         #q = (dq_target ** 2 * math.sqrt(2) * (-math.sqrt(2) * dq_target ** 2 / 6 - math.sqrt(2 * dq_target ** 4 + 12 * acc * qmax - 12 * acc * qmin) / 6) + 2 * acc * qmax + acc * qmin) / acc / 3

#         dq = np.sign(acc)*sqrt(2*abs(acc)*( (qmax-q)  )+dq_target**2 )
#     else:
#         r = minimize(area2,qmax, jac=False, method='slsqp', # slsqp
#                             options={'maxiter': 200, 'disp': plot_trigger },#, # maximum iteration number
#                             constraints=(
#                                 {'type':'ineq','fun': q_limit},
#                                 {'type':'ineq','fun': q_limit_pos}
#                                         ))
#         (q,dq)=(r.x[0],np.sign(acc)*sqrt(2*abs(acc)*( (r.x[0]-X_now[0])  )+dq_target**2 ))
#         print('Area coordinates',q,dq,'\n')
    
#     return (q,dq)

def R_reorder(R,option=True):
    Q=R
    Q.sort(key = lambda x: x[3],reverse=option) 
    return Q

def L_reorder(R,option=True):
    Q=R
    Q.sort(key = lambda x: x[0],reverse=option) 
    return Q

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

def new_areas_2(Xup,q,dq):
    area1=[Xup[0],q,dq,Xup[3]]
    area3=[q,Xup[1],Xup[2],Xup[3]]
    return (area1,area3)    

def new_areas_neg_2(Xup,q,dq):
    area1=[q,Xup[1],dq,Xup[3]]
    area3=[Xup[0],q,Xup[2],Xup[3]]
    return (area1,area3)


####

###
def Minimize_area_viab_extend(X_now,X_viab,acc,plot_trigger=False,classic=False):
    
    if (classic==False):
        qmin=X_now[0]
        (dq_target,qmax)=(X_viab[3],X_viab[1])
    
    if (classic==True ):
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


def R_analyzer(R,L):
    EPS=1E-6
    q_find=L[1]
    for i in range( int(size(R)/4) ):
        if ( (R[i][0] <= q_find+ EPS) and (R[i][0] >= q_find - EPS) ) :
            target=R[i]
            return target
    
    return []