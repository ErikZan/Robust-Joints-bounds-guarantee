# -*- coding: utf-8 -*-

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
    NOT FLEXIBLE
    -No continuos-time-guarantee of the satisfaction of the position limits:
    i.e. if It starts from the lower limits with negative velocity limits is 
    violated in continuos while not being warned from the algorithm
    COMPUTE MAXIMUM IN POSITION
    '''
    q=L_point[0]
    dq=L_point[1]
    qmax=X[0][1]
    qmin=X[0][0]
    
    def cost_fun(tau): # FLEXIBILITY: can i add another term to taking account of the previous step?
        cost = dq +dt*(tau+nonlinear)
        return cost**2

    def pos_upper_bound(tau): # No continuous time bound satisfaction!       
        position =q+dt*dq+dt**2 * ((tau+nonlinear)/M)/2 ;
        return position+qmax

    def pos_lower_bound(tau):        
        position = q+dt*dq+dt**2 * ((tau+nonlinear)/M)/2 ;
        return position+qmin
    
    r=minimize(cost_fun, [0], jac=False, method='slsqp',  # slsqp
                      options={'maxiter': 200, 'disp': True }, # maximum iteration number
                     constraints=({'type':'ineq','fun': pos_upper_bound},
                         {'type':'ineq','fun': pos_lower_bound}
                                  ))
    return r
    



######################################################################
# Here there are functions that are not in use and possibly not correct
# To be removed
###################################################################### 
  
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