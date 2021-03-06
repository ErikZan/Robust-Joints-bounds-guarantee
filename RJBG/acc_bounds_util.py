# -*- coding: utf-8 -*-
"""

Utility functions to compute the acceleration bounds for a system
with bounded position, velocity and accelerations. These computations
are based on the viability theory. A state is viable if, starting
from that state, it is possible to respect all the bounds (i.e. position, 
velocity and acceleration) in the future.

"""
import numpy as np

EPS = 1e-6;    # tolerance used to check violations

''' Return 0 if the state is viable, otherwise it returns a measure
    of the violation of the violated inequality. '''
def isStateViable(q, dq, qMin, qMax, dqMax, ddqMax, verbose=False):
    if(q<qMin-EPS):
        if(verbose):
            print("State (%f,%f) not viable because q<qMin" % (q,dq));
        return qMin-q;
    if(q>qMax+EPS):
        if(verbose):
            print("State (%f,%f) not viable because q>qMax" % (q,dq));
        return q-qMax;
    if(abs(dq)>dqMax+EPS):
        if(verbose):
            print("State (%f,%f) not viable because |dq|>dqMax" % (q,dq));
        return abs(dq)-dqMax;
    dqMaxViab =   np.sqrt(max(0,2*ddqMax*(qMax-q)));
    if(dq>dqMaxViab+EPS):
        if(verbose):
            print("State (%f,%f) not viable because dq>dqMaxViab=%f" % (q,dq,dqMaxViab));
        return dq-dqMaxViab;
    dqMinViab = - np.sqrt(max(0,2*ddqMax*(q-qMin)));
    if(dq<dqMinViab+EPS):
        if(verbose):
            print("State (%f,%f) not viable because dq<dqMinViab=%f" % (q,dq,dqMinViab));
        return dqMinViab-dq;

    if(verbose):
        print("State (%f,%f) is viable because dq<dqMinViab=%f and dq>dqMaxViab=%f" % (q,dq,dqMinViab,dqMaxViab));
    return 0.0;
    

''' Compute acceleration limits imposed by position bounds.
'''
def computeAccLimitsFromPosLimits(q, dq, qMin, qMax, ddqMax, dt, verbose=True):
    two_dt_sq   = 2.0/(dt**2);
    ddqMax_q3 = two_dt_sq*(qMax-q-dt*dq);
    ddqMin_q3 = two_dt_sq*(qMin-q-dt*dq);
    minus_dq_over_dt = -dq/dt;
    if(dq<=0.0):
        ddqUB  = ddqMax_q3;
        if(ddqMin_q3 < minus_dq_over_dt):
            ddqLB  = ddqMin_q3
        elif(q!=qMin):
            ddqMin_q2 = dq**2/(2*(q-qMin));
            ddqLB  = max(ddqMin_q2,minus_dq_over_dt);
        else:
            # q=qMin -> you're gonna violate the position bound
            ddqLB = ddqMax;
    else:
        ddqLB  = ddqMin_q3;
        if(ddqMax_q3 > minus_dq_over_dt):
            ddqUB  = ddqMax_q3;
        elif(q!=qMax):
            ddqMax_q2 = -dq**2/(2*(qMax-q));
            ddqUB  = min(ddqMax_q2,minus_dq_over_dt);
        else:
            # q=qMax -> you're gonna violate the position bound
            ddqUB = ddqMax;
            
    return (ddqLB, ddqUB);
    
    
''' Acceleration limits imposed by viability.
    ddqMax is the maximum acceleration that will be necessary to stop the joint before
    hitting the position limits.
    
     -sqrt( 2*ddqMax*(q-qMin) ) < dq[t+1] < sqrt( 2*ddqMax*(qMax-q) )
    ddqMin[2] = (-np.sqrt(max(0.0, 2*MAX_ACC*(q[i]+DT*dq[i]-qMin))) - dq[i])/DT;
    ddqMax[2] = (np.sqrt(max(0.0, 2*MAX_ACC*(qMax-q[i]-DT*dq[i]))) - dq[i])/DT;    
'''
def computeAccLimitsFromViability(q, dq, qMin, qMax, ddqMax, dt, verbose=True):
    dt_square = dt**2;
    dt_dq = dt*dq;
    minus_dq_over_dt = -dq/dt;
    dt_two_dq = 2*dt_dq;
    two_ddqMax = 2*ddqMax;
    dt_ddqMax_dt = ddqMax*dt_square;
    dq_square = dq**2;
    q_plus_dt_dq = q + dt_dq;
    
    two_a = 2*dt_square;
    b = dt_two_dq + dt_ddqMax_dt;
    c = dq_square - two_ddqMax*(qMax - q_plus_dt_dq);
    delta = b**2 - 2*two_a*c;
    if(delta>=0.0):
        ddq_1 = (-b + np.sqrt(delta))/(two_a);
    else:
        ddq_1 = minus_dq_over_dt;
        if(verbose):
            print("Error: state (%f,%f) not viable because delta is negative: %f" % (q,dq,delta));
    
    b = dt_two_dq - dt_ddqMax_dt;
    c = dq_square - two_ddqMax*(q_plus_dt_dq - qMin);
    delta = b**2 - 2*two_a*c;
    if(delta >= 0.0):
        ddq_2 = (-b - np.sqrt(delta))/(two_a);
    else:
        ddq_2 = minus_dq_over_dt;
        if(verbose):
            print("Error: state (%f,%f) not viable because delta is negative: %f" % (q,dq,delta))
    ddqUB = max(ddq_1, minus_dq_over_dt);
    ddqLB = min(ddq_2, minus_dq_over_dt);
    return (ddqLB, ddqUB);
    
        
''' Given the current position and velocity, the bounds of position,
    velocity and acceleration and the control time step, compute the
    bounds of the acceleration such that all the bounds are respected
    at the next time step and can be respected in the future.
    ddqStop is the maximum acceleration that will be necessary to stop the joint before
    hitting the position limits, whereas ddqMax is the absolute maximum acceleration.
'''
def computeAccLimits(q, dq, qMin, qMax, dqMax, ddqMax, dt, verbose=True, ddqStop=None):
    viabViol = isStateViable(q, dq, qMin, qMax, dqMax, ddqMax);
    if(viabViol>EPS and verbose):
        print("WARNING: specified state (q=%f dq=%f) is not viable (violation %f)" % (q,dq,viabViol));
        
    if(ddqStop==None):
        ddqStop=ddqMax;
        
    ddqUB = np.zeros(4);
    ddqLB = np.zeros(4);
    
    # Acceleration limits imposed by position bounds
    (ddqLB[0], ddqUB[0]) = computeAccLimitsFromPosLimits(q, dq, qMin, qMax, ddqMax, dt, verbose);
    
    # Acceleration limits imposed by velocity bounds
    # dq[t+1] = dq + dt*ddq < dqMax
    # ddqMax = (dqMax-dq)/dt
    # ddqMin = (dqMin-dq)/dt = (-dqMax-dq)/dt
    ddqLB[1] = (-dqMax-dq)/dt;
    ddqUB[1] = (dqMax-dq)/dt;
    
    # Acceleration limits imposed by viability
    (ddqLB[2], ddqUB[2]) = computeAccLimitsFromViability(q, dq, qMin, qMax, ddqStop, dt, verbose);
     
    # Acceleration limits
    ddqLB[3] = -ddqMax;
    ddqUB[3] = ddqMax;
    
    # Take the most conservative limit for each joint
    ddqLBFinal = np.max(ddqLB);
    ddqUBFinal = np.min(ddqUB);
    
    # In case of conflict give priority to position bounds
    if(ddqUBFinal<ddqLBFinal):
        if(verbose and ddqUBFinal<ddqLBFinal-EPS):
            print("Conflict between acc bounds min %f>max %f (min-max=%f)" % (ddqLBFinal,ddqUBFinal, ddqLBFinal-ddqUBFinal));
            print("(q,dq) = ", q, dq);
            print("ddqLB (pos,vel,viab,acc): ", ddqLB);
            print("ddqUB (pos,vel,viab,acc): ", ddqUB);
        if(ddqUBFinal==ddqUB[0]):
            ddqLBFinal = ddqUBFinal;
        else:
            ddqUBFinal = ddqLBFinal;
        if(verbose and ddqUBFinal<ddqLBFinal-EPS):
            print("                     New bounds are ddqMin %f ddqMax %f" % (ddqLBFinal,ddqUBFinal));
        
    return (ddqLBFinal,ddqUBFinal);
    
def computeMultiAccLimits(q, dq, qMin, qMax, dqMax, ddqMax, dt, verbose=True, ddqStop=None):
    ddqLB = np.zeros(len(q));
    ddqUB = np.zeros(len(q));
    if(ddqStop==None):
        ddqStop = ddqMax;
    for i in range(len(q)):
        (ddqLB[i], ddqUB[i]) = computeAccLimits(q[i], dq[i], qMin[i], qMax[i], dqMax[i], ddqMax[i], dt, verbose, ddqStop[i]);
    return (ddqLB, ddqUB);
    
# Here starts the second modified part 

''' Return 0 if the state is viable, otherwise it returns a measure
    of the violation of the violated inequality. '''
def isStateViable_2(q, dq, qMin, qMax, dqMax, ddqMax,E, verbose=False):
    if(q<qMin-EPS):
        if(verbose):
            print("State (%f,%f) not viable because q<qMin" % (q,dq));
        return qMin-q;
    if(q>qMax+EPS):
        if(verbose):
            print("State (%f,%f) not viable because q>qMax" % (q,dq));
        return q-qMax;
    if(abs(dq)>dqMax+EPS):
        if(verbose):
            print("State (%f,%f) not viable because |dq|>dqMax" % (q,dq));
        return abs(dq)-dqMax;
    dqMaxViab =   np.sqrt(max(0,2*(ddqMax-E)*(qMax-q)));
    if(dq>dqMaxViab+EPS):
        if(verbose):
            print("State (%f,%f) not viable because dq>dqMaxViab=%f" % (q,dq,dqMaxViab));
        return dq-dqMaxViab;
    dqMinViab = - np.sqrt(max(0,2*(ddqMax-E)*(q-qMin)));
    if(dq<dqMinViab+EPS):
        if(verbose):
            print("State (%f,%f) not viable because dq<dqMinViab=%f" % (q,dq,dqMinViab));
        return dqMinViab-dq;

    if(verbose):
        print("State (%f,%f) is viable because dq<dqMinViab=%f and dq>dqMaxViab=%f" % (q,dq,dqMinViab,dqMaxViab));
    return 0.0;
    

''' Compute acceleration limits imposed by position bounds.
'''
def computeAccLimitsFromPosLimits_2(q, dq, qMin, qMax, ddqMax, dt,E, verbose=True):
    two_dt_sq   = 2.0/(dt**2);
    ddqMax_q3 = two_dt_sq*(qMax-q-dt*dq)-E;
    ddqMin_q3 = two_dt_sq*(qMin-q-dt*dq)+E;
    minus_dq_over_dt = -dq/dt;
    if(dq<=0.0):
        ddqUB  = ddqMax_q3;
        if(ddqMin_q3 < minus_dq_over_dt):
            ddqLB  = ddqMin_q3
        elif(q!=qMin):
            ddqMin_q2 = dq**2/(2*(q-qMin))+E;
            ddqLB  = max(ddqMin_q2,minus_dq_over_dt);
        else:
            # q=qMin -> you're gonna violate the position bound
            ddqLB = ddqMax+E;
    else:
        ddqLB  = ddqMin_q3;
        if(ddqMax_q3 > minus_dq_over_dt):
            ddqUB  = ddqMax_q3;
        elif(q!=qMax):
            ddqMax_q2 = -dq**2/(2*(qMax-q))-E;
            ddqUB  = min(ddqMax_q2,minus_dq_over_dt);
        else:
            # q=qMax -> you're gonna violate the position bound
            ddqUB = ddqMax-E;
            
    return (ddqLB, ddqUB);
    
    
''' Acceleration limits imposed by viability.
    ddqMax is the maximum acceleration that will be necessary to stop the joint before
    hitting the position limits.
    
     -sqrt( 2*ddqMax*(q-qMin) ) < dq[t+1] < sqrt( 2*ddqMax*(qMax-q) )
    ddqMin[2] = (-np.sqrt(max(0.0, 2*MAX_ACC*(q[i]+DT*dq[i]-qMin))) - dq[i])/DT;
    ddqMax[2] = (np.sqrt(max(0.0, 2*MAX_ACC*(qMax-q[i]-DT*dq[i]))) - dq[i])/DT;    
'''
def computeAccLimitsFromViability_2(q, dq, qMin, qMax, ddqMax, dt,E, verbose=True):
    dt_square = dt**2;
    dt_dq = dt*dq;
    minus_dq_over_dt = -dq/dt+E;
    dt_two_dq = 2*dt_dq;
    two_ddqMax = 2*(ddqMax-E);
    dt_ddqMax_dt = (ddqMax-E)*dt_square;
    dq_square = dq**2;
    q_plus_dt_dq = q + dt_dq;
    
    two_a = 2*dt_square;
    b = dt_two_dq + dt_ddqMax_dt;
    c = dq_square - two_ddqMax*(qMax - q_plus_dt_dq);
    delta = b**2 - 2*two_a*c;
    if(delta>=0.0):
        ddq_1 = (-b + np.sqrt(delta))/(two_a);
    else:
        ddq_1 = minus_dq_over_dt;
        if(verbose):
            print("Error: state (%f,%f) not viable because delta is negative: %f" % (q,dq,delta));
    
    b = dt_two_dq - dt_ddqMax_dt;
    c = dq_square - two_ddqMax*(q_plus_dt_dq - qMin);
    delta = b**2 - 2*two_a*c;
    if(delta >= 0.0):
        ddq_2 = (-b - np.sqrt(delta))/(two_a);
    else:
        ddq_2 = minus_dq_over_dt;
        if(verbose):
            print("Error: state (%f,%f) not viable because delta is negative: %f" % (q,dq,delta))
    ddqUB = max(ddq_1, minus_dq_over_dt);
    ddqLB = min(ddq_2, minus_dq_over_dt);
    return (ddqLB, ddqUB);
    
        
''' Given the current position and velocity, the bounds of position,
    velocity and acceleration and the control time step, compute the
    bounds of the acceleration such that all the bounds are respected
    at the next time step and can be respected in the future.
    ddqStop is the maximum acceleration that will be necessary to stop the joint before
    hitting the position limits, whereas ddqMax is the absolute maximum acceleration.
'''
def computeAccLimits_2(q, dq, qMin, qMax, dqMax, ddqMax, dt,E, verbose=True, ddqStop=None):
    viabViol = isStateViable(q, dq, qMin, qMax, dqMax, ddqMax);
    if(viabViol>EPS and verbose):
        print("WARNING: specified state (q=%f dq=%f) is not viable (violation %f)" % (q,dq,viabViol));
        
    if(ddqStop==None):
        ddqStop=ddqMax;
        
    ddqUB = np.zeros(4);
    ddqLB = np.zeros(4);
    
    # Acceleration limits imposed by position bounds
    (ddqLB[0], ddqUB[0]) = computeAccLimitsFromPosLimits_2(q, dq, qMin, qMax, ddqMax, dt, E,verbose);
    
    # Acceleration limits imposed by velocity bounds
    # dq[t+1] = dq + dt*ddq < dqMax
    # ddqMax = (dqMax-dq)/dt
    # ddqMin = (dqMin-dq)/dt = (-dqMax-dq)/dt
    ddqLB[1] = (-dqMax-dq)/dt+E;
    ddqUB[1] = (dqMax-dq)/dt-E;
    
    # Acceleration limits imposed by viability
    (ddqLB[2], ddqUB[2]) = computeAccLimitsFromViability_2(q, dq, qMin, qMax, ddqStop, dt,E, verbose);
     
    # Acceleration limits
    ddqLB[3] = -ddqMax+E;
    ddqUB[3] = ddqMax-E;
    
    # Take the most conservative limit for each joint
    ddqLBFinal = np.max(ddqLB);
    ddqUBFinal = np.min(ddqUB);
    
    # In case of conflict give priority to position bounds
    if(ddqUBFinal<ddqLBFinal):
        if(verbose and ddqUBFinal<ddqLBFinal-EPS):
            print("Conflict between acc bounds min %f>max %f (min-max=%f)" % (ddqLBFinal,ddqUBFinal, ddqLBFinal-ddqUBFinal));
            print("(q,dq) = ", q, dq);
            print("ddqLB (pos,vel,viab,acc): ", ddqLB);
            print("ddqUB (pos,vel,viab,acc): ", ddqUB);
        if(ddqUBFinal==ddqUB[0]):
            ddqLBFinal = ddqUBFinal;
        else:
            ddqUBFinal = ddqLBFinal;
        if(verbose and ddqUBFinal<ddqLBFinal-EPS):
            print("                     New bounds are ddqMin %f ddqMax %f" % (ddqLBFinal,ddqUBFinal));
        
    return (ddqLBFinal,ddqUBFinal);
    
def computeMultiAccLimits_2(q, dq, qMin, qMax, dqMax, ddqMax, dt, verbose=True, ddqStop=None):
    ddqLB = np.zeros(len(q));
    ddqUB = np.zeros(len(q));
    if(ddqStop==None):
        ddqStop = ddqMax;
    for i in range(len(q)):
        (ddqLB[i], ddqUB[i]) = computeAccLimits(q[i], dq[i], qMin[i], qMax[i], dqMax[i], ddqMax[i], dt, verbose, ddqStop[i]);
    return (ddqLB, ddqUB);
    