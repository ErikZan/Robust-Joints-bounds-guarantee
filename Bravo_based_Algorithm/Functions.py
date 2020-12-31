# -*- coding: utf-8 -*-

from numpy.lib.financial import ppmt
import pinocchio as se3
from pinocchio.utils import *
import os
import numpy.matlib
from numpy.linalg import pinv,inv
from math import sqrt
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
from scipy.optimize import minimize

# Functions for the Bravo et al. algoriths


def compute_torque(X,U,L_point,dt,nonlinear,M):
    '''
    Compute the minimum torque necessary to STOP the joint in ONE time step
    while subject to the position limits. If the result exceed your torque limit
    discard the point.
    
    PROBLEMS:
    -Contemporary design suitable only for stopping the joint in one timestep:
    NOT FLEXIBLE -> to do
    -No continuos-time-guarantee of the satisfaction of the position limits:
    i.e. if It starts from the lower limits with negative velocity limits is 
    violated in continuos while not being warned from the algorithm:
    COMPUTE MAXIMUM IN POSITION -> added pos_middle_position() that mitigate 
    but does not solve the the problem
    '''
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
    
def remove_old_add_new(point,L,P,X):
    '''
    When a point has a torque that is to high to stop into the position limits we remove it from L
    and send it to R. Then we perform check on P if there is a point that share a similar a x or y position
    and we move toward it. If there is no point we create a new point initializate with a coordinate shared
    with the previous point and one on the x or y limit.
    '''   
    x0=float(point[0]);
    y0=float(point[1]);
    a=[];
    qmax=X[0][1]
    vmax=X[1][1]
    if (len(P)==0):
        P=[(0,0)]
        
    for i in range(len(P)): # check on P is there is a similar point
            
        if ( abs(y0) >= abs(x0) ):  # if velocity > position we reduce it  
                                    # Possible improvement is to rescale accordingly the bound
            if (x0==P[i][0]):
                    b=P[i][1]
                    a.append(b)
          
            if (y0>=0.0):
                if (a==[] and y0==0.0):
                    a=[-vmax]
                elif(y0>=0.0):
                    a=[0.0]
                max_v= max(a)
                
                new_point=(x0,(y0+max_v)/2.00)
            else:
                if (a==[] and y0==0.0):
                    a=[vmax]
                elif(y0<=0.0):
                    a=[0.0]
                max_v= min(a)
                
                new_point=(x0,(y0+max_v)/2.00)
                  
        else:
            print("else")
            if (y0==P[i][1]):
                    b=P[i][0]
                    a.append(b)
                                    
            if (x0>=0.0):
                if (a==[] and x0==0.0):
                    a=[-qmax]
                elif(x0>=0.0):
                    a=[0.0]
                max_v= max(a)
                new_point=((x0+max_v)/2.00,y0)
            else:
                if (a==[] and y0==0.0):
                    a=[qmax]
                elif(x0<=0.0):
                    a=[0.0]
                max_v= min(a)
                new_point=((x0+max_v)/2.00,y0)
                
    L.remove((x0,y0))   # We remove the point from L
    L.append(new_point) # and add the new one
    return 

def remove_old_add_new_if_P(point,L,R,X):
    '''
    Same as remove_old_add_new() but with for positively evaluated number. 
    We also add to L a new_point_2, but it is probably to check again if is it useful
    '''    
    x0=float(point[0]);
    y0=float(point[1]);
    a=[];
    qmax=X[0][1]
    vmax=X[1][1]
    for i in range(len(R)):
            
    
        if ( abs(y0) >= abs(x0) ):
            
            if (x0==R[i][0]):
                    b=R[i][1]
                    a.append(b)
                    
            # if (a==[]):
            #     a=[vmax]
                       
            if (y0>=0.0):
                if (a==[]):
                    a=[vmax]
                max_v= max(a)
                new_point=(x0,(y0+max_v)/2.00)
                new_point2=(np.sign(x0)*np.random.random(1)*vmax,(y0+max_v)/2.00)
                
            else:
                if (a==[]):
                    a=[-vmax]
                max_v= min(a)
                new_point=(x0,(y0+max_v)/2.00)
                new_point2=(np.sign(x0)*np.random.random(1)*vmax,(y0+max_v)/2.00)  
                
        elif ( abs(y0) <= abs(x0) ):
            
            if (y0==R[i][1]):
                    b=R[i][0]
                    a.append(b)
                    
            # if (a==[]):
            #     a=[-np.sign(x0)*qmax]
                            
            if (x0>=0.0):
                if (a==[]):
                    a=[vmax]
                max_v= max(a)
                new_point=((x0+max_v)/2.00,y0)
                new_point2=((x0+max_v)/2.00,np.sign(y0)*np.random.random(1)*qmax)
            else:
                if (a==[]):
                    a=[-vmax]
                max_v= min(a)
                new_point=((x0+max_v)/2.00,y0)
                new_point2=((x0+max_v)/2.00,np.sign(y0)*np.random.random(1)*qmax)
    
    L.remove((x0,y0))
    L.append(new_point)
    L.append(new_point2)
    return 

def rejected_point(point,L,P,X):
    x0=point[0];
    y0=point[1];
    Pp=[0.0];
  
    if ( abs(y0) >= abs(x0) ):
        
        for i in range(len(P)):
            if (x0==P[i][0]):
                    b=P[i][1]
                    Pp.append(b)
                    
        P_vec = [None]*len(Pp)           
        for i in range(len(Pp)):
            P_vec[i]=abs(y0-Pp[i])
            
        P_max=min(P_vec)
        ind=P_vec.index(P_max)
        P_eff=Pp[ind]
                        
        new_point= (x0,(y0+P_eff)/2.0)
    elif ( abs(x0) >= abs(y0) ):
        for i in range(len(P)):
            if (y0==P[i][1]):
                    b=P[i][0]
                    Pp.append(b)
                    
        P_vec = [None]*len(Pp)           
        for i in range(len(Pp)):
            P_vec[i]=abs(x0-Pp[i])
            
            
        P_max=min(P_vec)
        ind=P_vec.index(P_max)
        P_eff=Pp[ind]
                        
        new_point= ((x0+P_eff)/2.0,y0)    

    L.remove((x0,y0))
    L.append(new_point)
    return

def Approved_point(point,L,P,X):
    x0=point[0];
    y0=point[1];
    
    pos_y= X[1][1]
    pos_x= X[0][1]
    neg_y= X[1][0]
    neg_x= X[0][0]
  
    if ( abs(y0) >= abs(x0) ):
        if (y0>=0):
            Pp=[pos_y/1.5];
        else:
            Pp=[neg_y/1.5];
            
        for i in range(len(P)):
            if (x0==P[i][0]):
                    b=P[i][1]
                    Pp.append(b)
                    
        P_vec = [None]*len(Pp)           
        for i in range(len(Pp)):
            P_vec[i]=abs(y0-Pp[i])
            
        P_max=min(P_vec)
        ind=P_vec.index(P_max)
        P_eff=Pp[ind]
                        
        new_point= (x0,(y0+P_eff)/2.0)
    elif ( abs(x0) >= abs(y0) ):
        if (x0>=0):
            Pp=[pos_x/1.5];
        else:
            Pp=[neg_x/1.5];
            
        for i in range(len(P)):
            if (y0==P[i][1]):
                    b=P[i][0]
                    Pp.append(b)
                    
        P_vec = [None]*len(Pp)           
        for i in range(len(Pp)):
            P_vec[i]=abs(x0-Pp[i])
            
            
        P_max=min(P_vec)
        ind=P_vec.index(P_max)
        P_eff=Pp[ind]
                        
        new_point= ((x0+P_eff)/2.0,y0)    

    L.remove((x0,y0))
    L.append(new_point)
    return
######################################################################
# Here there are functions that are not in use and possibly not correct
# To be removed
###################################################################### 
# def remove_old_add_new(point,L,q):
    
#     x0=point[0];
#     y0=point[1];
#     a=[];
    
#     for i in range(len(L)):
#             if (x0==L[i][0]):
#                 b=[L[i][0]]
#                 a.append(b)
    
#     if ( abs(y0) >= abs(x0) ):
        
#         if (y0>0):
#             max_v= np.amax(a)
#             new_point=np.array([[x0,(y0-max_v)/2]])
#         else:
#             max_v= np.amin(a)
#             new_point=np.array([[x0,(y0-max_v)/2]])   
#     else:
        
#         if (y0>0):
#             max_v= np.amax(a)
#             new_point=np.array([[(x0-max_v)/2,y0]])
#         else:
#             max_v= np.amin(a)
#             new_point=np.array([[(x0-max_v)/2,y0]])
            
#     ind=np.where(L==point)
#     new_L = np.delete(L,q)
#     new_L =np.concatenate((L,new_point))
#     return new_L 
 
  
# def Compute_torque(X,U,L_point,dt,nonlinear,M):

#     min_torque=U[0][0];
#     max_torque=U[0][1];
#     tq_bnds= (min_torque,max_torque); # not useful
#     print('the point',L_point,'\n');
#     #print(tq_bnds);
#     qmax= X[0][1];
#     qmin= X[0][0];
#     eps=10E-8;
    
#     def obj(tau):
#         q=tau**2
#         return q
    
#     if (L_point[1]>=0):
#         def vel_cons(tau):
#             velocity= L_point[1]+dt*((tau-nonlinear)/M)
#             return -velocity
#     else:
#         def vel_cons(tau):
#             velocity = L_point[1]+dt*((tau-nonlinear)/M)
#             return velocity
        
#     # def vel_neg_cons(tau):
#     #     velocity= L_point[1]+dt*((tau-nonlinear)/M)
#     #     return -velocity
    
#     # def vel_pos_cons(tau):
#     #     velocity= L_point[1]+dt*((tau-nonlinear)/M)
#     #     return velocity
    
#     def pos_upper_bound(tau):
#         #velocity= L_point[1]+dt*((tau-nonlinear)/M)
#         position = L_point[0]+dt*L_point[1]+dt**2 * ((tau-nonlinear)/M)/2 ;
#         return -(position)+(qmax+eps)
    
#     # def vel_cons(tau):
#     #     velocity= L_point[1]+dt*((tau-nonlinear)/M)
#     #     return -velocity
    
#     def pos_lower_bound(tau):
#         #velocity= L_point[1]+dt*((tau-nonlinear)/M)
#         position = L_point[0]+dt*L_point[1]+dt**2 * ((tau-nonlinear)/M)/2 ;
#         return position-(qmin-eps)
    
    
#     r=minimize(obj, [0], jac=False, method='slsqp',  # slsqp
#                       options={'maxiter': 200, 'disp': True }, # maximum iteration number
#                      constraints=(
#                          {'type':'ineq','fun': vel_cons },
#                          #{'type':'ineq','fun': vel_neg_cons},
#                          #{'type':'ineq','fun': vel_pos_cons},
#                          {'type':'ineq','fun': pos_upper_bound},
#                          {'type':'ineq','fun': pos_lower_bound}
#                                   ))
#     return r

# def Compute_final_pos(X,U,L_point,dt,nonlinear,M):
    
#     min_torque=U[0][0];
#     max_torque=U[0][1];
#     tq_bnds= (min_torque,max_torque); # not useful
#     print('the point',L_point,'\n');
#     #print(tq_bnds);
#     qmax= X[0][1];
#     qmin= X[0][0];
#     eps=10E-8;
    
#     def obj(tau):
#         q=(L_point[1]+dt*((tau-nonlinear)/M))**2
#         return q
    
    
    
#     r=minimize(obj, [0], jac=False, method='trust-constr',  # slsqp
#                       options={'maxiter': 200, 'disp': True },bounds=tq_bnds # maximum iteration number
#                     #  constraints=(
#                     #      {'type':'ineq','fun': vel_cons },
#                     #      #{'type':'ineq','fun': vel_neg_cons},
#                     #      #{'type':'ineq','fun': vel_pos_cons},
#                     #      {'type':'ineq','fun': pos_upper_bound},
#                     #      {'type':'ineq','fun': pos_lower_bound}
#                                   )
#     return r

# def tau_to_stop(X,U,L_point,dt,nonlinear,M):
#     tau = -(L_point[1]*M/dt-nonlinear)
#     return tau