# -*- coding: utf-8 -*-
"""

Test the expressions for the acceleration limits that guarantees the feasibility
of the position and velocity limits in the future.
These expressions have been derived using the viability theory.

"""
import numpy as np
from numpy.random import random
from plot_utils import create_empty_figure
import plot_utils as plut
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import brewer2mpl
from acc_bounds_util import isStateViable
from acc_bounds_util import computeAccLimits
from acc_bounds_util import computeAccLimitsFromViability
from acc_bounds_util import computeAccLimitsFromPosLimits
from inequalities_probability import InequalitiesProbability
import sys
import math
import matplotlib.patches as mpatches

def computeControlTransitionMatrix(A, B, T):
    n = B.shape[0];
    m = B.shape[1];
    G = np.zeros((T*n, T*m));
    A_to_the_t_times_B = B;
    G[0:n,0:m] = B;
    for t in range(1,T):
        # copy the line above shifted right
        G[t*n:(t+1)*n, m:] = G[(t-1)*n:t*n, :-m];
        # compute A^(t-1)*B        
        A_to_the_t_times_B = np.dot(A, A_to_the_t_times_B);
        G[t*n:(t+1)*n, :m] = A_to_the_t_times_B;
        
    return G;

def computeControlTransitionMatrix_with_error(A, B, T, E):
    n = B.shape[0];
    m = B.shape[1];
    G = np.zeros((T*n, T*m));
    A_to_the_t_times_B = B;
    G[0:n,0:m] = B;
    for t in range(1,T):
        # copy the line above shifted right
        G[t*n:(t+1)*n, m:] = G[(t-1)*n:t*n, :-m];
        # compute A^(t-1)*B        
        A_to_the_t_times_B = np.dot(A, A_to_the_t_times_B) + E;
        G[t*n:(t+1)*n, :m] = A_to_the_t_times_B;
        
    return G;
    
def computeStateTransitionMatrix(A, T):
    n = A.shape[0];
    G = np.zeros((T*n, n));
    A_to_the_t = A;
    G[:n,:] = A;
    for t in range(1,T):
        # compute A^(t-1)
        A_to_the_t = np.dot(A, A_to_the_t);
        G[t*n:(t+1)*n, :] = A_to_the_t;
        
    return G;

def computeStateTransitionMatrix_with_error(A, T, E):
    n = A.shape[0];
    G = np.zeros((T*n, n));
    A_to_the_t = A;
    G[:n,:] = A;
    for t in range(1,T):
        # compute A^(t-1)
        A_to_the_t = np.dot(A, A_to_the_t)+E;
        G[t*n:(t+1)*n, :] = A_to_the_t;
        
    return G;

plut.DEFAULT_LINE_WIDTH = 5;
plut.BOUNDS_COLOR = 'r';
plut.SAVE_FIGURES = True; 
EPS = 1e-10;
LW = 2;
EPS = 1e-10;

TEST_RANDOM=0;
TEST_DISCRETE_VIABILITY=1;
TEST_MAX_ACC = 0;    # if true select always the maximum acceleration possible 
TEST_MIN_ACC = False;    # if true select always the minimum acceleration possible 
TEST_MED_ACC = False;    # if true select always the average of max and min acc
TEST_MED_POS_ACC = True;    # if true select always the average of max and min acc imposed by pos bounds (saturated if necessary)
PLOT_STATE_SPACE = True;
PLOT_STATE_SPACE_DECRE = False;
PLOT_STATE_SPACE_PADOIS = False;
PLOT_STATE_SPACE_PROBABILITY = False;
PLOT_SIMULATION_RESULTS = False;
qMax    = 2.0;
qMin    = -2.0;
MAX_VEL = 5.0;
MAX_ACC = 10.0;
q0      =  1.75;
dq0     =  4.50;
N_TESTS = 20;
DT = 0.010;
VIABILITY_MARGIN = 1e10; # minimum margin to leave between ddq and its bounds found through viability
error_trigger = 0.0
E=0.00*MAX_ACC;
''' State in which acc bounds from pos are stricter than acc bounds from viability '''
#q0 = -0.086850;
#dq0 = -0.093971;
#qMin = -0.087266;
#DT = 0.01

DT_SAFE = 1.01*DT # 1.01*DT;
Q_INTERVAL = 0.02; # for plotting the range of possible angles is sampled with this step
DQ_INTERVAL = 0.2; # for plotting the range of possible velocities is sampled with this step

# Starting from qMin and applying maximum acceleration until you reach the middle position (i.e. 0.5*(qMax+qMin))
# you get the maximum velocity you can reach without violating position and acceleration bounds.
# By solving the equation:
#   qMin + 0.5*t**2*MAX_ACC = 0.5*(qMin+qMax)
# we can find t=sqrt((qMax-qMin)/MAX_ACC) and then use it to compute the max velocity:
#   dq(t) = t*MAX_ACC = sqrt(MAX_ACC*(qMax-qMin));
max_vel_from_pos_acc_bounds = np.sqrt(MAX_ACC*(qMax-qMin));
print("INFO Max velocity is %f, max velocity forced by position and acceleration bounds is %f" % (MAX_VEL, max_vel_from_pos_acc_bounds));


if(TEST_MAX_ACC):
    print("INFO Test acceleration bounds applying always maximum acceleration")
elif(TEST_MIN_ACC):
    print("INFO Test acceleration bounds applying always minimum acceleration")
elif(TEST_MED_ACC):
    print("INFO Test acceleration bounds applying always average between min and max acceleration")
elif(TEST_MED_POS_ACC):
    print("INFO Test acceleration bounds applying always the average of max and min acc imposed by pos bounds (saturated if necessary)")
else:
    print("INFO Test acceleration bounds applying always random accelerations between max and min acceleration")
print("")
    
    
if(MAX_VEL/MAX_ACC < DT):
    print("INFO Pay attention because dq_max/ddq_max (%f) < dt"%(MAX_VEL/MAX_ACC))

q   = np.zeros(N_TESTS+1);
dq  = np.zeros(N_TESTS+1);
ddq = np.zeros(N_TESTS);
ddqLB = np.zeros(N_TESTS);
ddqUB = np.zeros(N_TESTS);
ddqUB_viab = np.zeros(N_TESTS);
ddqLB_viab = np.zeros(N_TESTS);

q[0]    = q0;
dq[0]   = dq0;
for i in range(N_TESTS):
    #print "\nTime ", i;
    
    if(q[i]>qMax+EPS or q[i]<qMin-EPS):
        print("ERROR Time %d, position limits violated qMin=%f, q=%f, qMax=%f" % (i,qMin,q[i],qMax));
    if(dq[i]>MAX_VEL or dq[i]<-MAX_VEL):
        print("ERROR Time %d, velocity limits violated MAX_VEL=%f, dq=%f" % (i,MAX_VEL,dq[i]));
    if(i>0):
        if(dq[i-1]>0.0 and dq[i]<0.0):
            q_zero_vel = q[i-1] + 0.5*dq[i-1]**2/ddq[i-1];
            if(q_zero_vel>qMax+EPS):
                print("ERROR Time %d, up pos lim viol qOld=%f qNew=%f q=%f qMax=%f dqOld=%f" % (i,q[i-1],q[i],q_zero_vel,qMax,dq[i-1]));
        if(dq[i-1]<0.0 and dq[i]>0.0):
            q_zero_vel = q[i-1] - 0.5*dq[i-1]**2/ddq[i-1];
            if(q_zero_vel<qMin-EPS):
                print("ERROR Time %d, lower position limits violated q=%f, qMin=%f" % (i,q_zero_vel,qMin));
        
    (ddqMinFinal, ddqMaxFinal) = computeAccLimits(q[i], dq[i], qMin, qMax, MAX_VEL, MAX_ACC, DT_SAFE);
    ddqLB[i] = ddqMinFinal;
    ddqUB[i] = ddqMaxFinal;
    
    # check viability of future state if you apply ddq=ddqMinFinal
    qNew = q[i] + DT*dq[i] + 0.5*(DT**2)*ddqMinFinal;
    dqNew = dq[i] + DT*ddqMinFinal;
    viabViol = isStateViable(qNew, dqNew, qMin, qMax, MAX_VEL, MAX_ACC);
    if(viabViol!=0.0):
        dqMinViab = - np.sqrt(max(0.0,2*MAX_ACC*(qNew-qMin)));
        print("ERROR Time %d Incoherence in viab constr. qNew %f dqNew %f dqNewMinViab %f ddqMinFinal %f" % (i,qNew,dqNew,dqMinViab,ddqMinFinal));
        qNew = q[i] + DT_SAFE*dq[i] + 0.5*(DT_SAFE**2)*ddqMinFinal;
        dqNew = dq[i] + DT_SAFE*ddqMinFinal;    
        dqMinViab = - np.sqrt(max(0.0,2*MAX_ACC*(qNew-qMin)));
        print("       Values computed with DT_SAFE qNew %f dqNew %f dqNewMinViab %f" % (qNew,dqNew,dqMinViab));
        # compute min/max position assuming constant acceleration
        t_min = -dq[i]/ddqMinFinal;
        qMinAccConst = q[i] + t_min*dq[i] + 0.5*(t_min**2)*ddqMinFinal;
        print("       Trajectory will reach its min %f in %f sec" % (qMinAccConst, t_min));
    
    # check viability of future state if you apply ddq=ddqMaxFinal
    qNew = q[i] + DT*dq[i] + 0.5*DT**2*ddqMaxFinal          + error_trigger*(0.5*DT**2*MAX_ACC*0.1*random(1) - 0.5*DT**2*MAX_ACC*0.1*random(1)); # adding error
    dqNew = dq[i] + DT*ddqMaxFinal                          + error_trigger*(DT*MAX_ACC*0.1*random(1) - DT*MAX_ACC*0.1*random(1)); # adding error
    viabViol = isStateViable(qNew, dqNew, qMin, qMax, MAX_VEL, MAX_ACC);
    if(viabViol!=0.0):
        dqMaxViab =   np.sqrt(max(0.0,2*MAX_ACC*(qMax-qNew)));
        print("ERROR Time %d Incoherence in viability constraints. dqNew %f dqNewMaxViab %f" % (i,dqNew, dqMaxViab));

    # check viability of current state
    viabViol = isStateViable(q[i], dq[i], qMin, qMax, MAX_VEL, MAX_ACC);
    if(viabViol!=0.0):
        print("ERROR Time %d not viable q=%f, dq=%f, violation=%f" % (i, q[i], dq[i], viabViol));
    
    # Check conflicts between max and min bound
    if(ddqMaxFinal<ddqMinFinal-EPS):
        print("ERROR Time %d infeasible constraints: min=%f, max=%f" % (i, ddqMinFinal, ddqMaxFinal));
        
    (ddqLB_viab[i],ddqUB_viab[i]) = computeAccLimitsFromViability(q[i], dq[i], qMin, qMax, MAX_ACC, DT);
    
    if(TEST_MAX_ACC):
        ddq[i] = ddqMaxFinal;
    elif(TEST_MIN_ACC):
        ddq[i] = ddqMinFinal;
    elif(TEST_MED_ACC):
        ddq[i] = 0.5*(ddqMaxFinal+ddqMinFinal);
    elif(TEST_MED_POS_ACC):
        ddq[i] = random(1)*(2*MAX_ACC) - MAX_ACC;
        if(ddqUB_viab[i]-ddq[i] < VIABILITY_MARGIN or ddq[i]-ddqLB_viab[i] < VIABILITY_MARGIN):
            print("WARNING Time %d enforcing viability margin" % i)
            ddq[i] = 0.5*(ddqUB_viab[i]+ddqLB_viab[i]);
#        if(min(MAX_ACC,ddqUB_viab[i]) - max(-MAX_ACC,ddqLB_viab[i])>10.0):
#            ddq[i] = random(1)*(min(MAX_ACC,ddqUB_viab[i]) - max(-MAX_ACC,ddqLB_viab[i]) - 10) + max(-MAX_ACC,ddqLB_viab[i]) + 5;
#        else:
#            print "Time %d viable acc range is %f" % (i, min(MAX_ACC,ddqUB_viab[i]) - max(-MAX_ACC,ddqLB_viab[i]));
#            ddq[i] = 0.5*(ddqUB_viab[i]+ddqLB_viab[i]);
    else:
        ddq[i] = random(1)*(ddqMaxFinal - ddqMinFinal) + ddqMinFinal;
    
    if(ddq[i]>MAX_ACC):
        ddq[i] = MAX_ACC;
    elif(ddq[i]<-MAX_ACC):
        ddq[i] = -MAX_ACC;
        
    if(q[i]>qMax):
        ddq[i] = -MAX_ACC;
    elif(q[i]<qMin):
        ddq[i] = +MAX_ACC;
    
        
    
    if(TEST_MAX_ACC):     # if true select always the maximum acceleration possible 
        ddq[i]+=E     
    elif(TEST_MIN_ACC):
        ddq[i]-=E   
    elif(TEST_RANDOM):
        ddq[i]+=E*(random(1)/2-random(1)/2);
    elif(TEST_DISCRETE_VIABILITY):
        if (dq[i]<=0):
            ddq[i]-=E; #*(random(1)/2-random(1)/2);
        else:
            ddq[i]+=E; #*(random(1)/2-random(1)/2);

    dq[i+1] = dq[i] + DT*ddq[i];
    q[i+1]  = q[i] + DT*dq[i] + 0.5*(DT**2)*ddq[i];
    
    
print("Minimum ddq margin from viability upper bound: %f" % np.min(ddqUB_viab-ddq));
print("Minimum ddq margin from viability lower bound: %f" % np.min(ddq-ddqLB_viab));

if(PLOT_STATE_SPACE_PROBABILITY):
    std_dev = MAX_ACC;
    cmap=brewer2mpl.get_map('OrRd', 'sequential', 4, reverse=False).mpl_colormap;
    (f,ax) = create_empty_figure(1,1, [0,0]);

    # plot viability constraints
    qMid = 0.5*(qMin+qMax);
    q_plot = np.arange(qMid, qMax+Q_INTERVAL, Q_INTERVAL);
    dq_plot = np.sqrt(np.max([np.zeros(q_plot.shape),2*MAX_ACC*(qMax-q_plot)],0));
    ax.plot(q_plot,dq_plot, 'r--');
    q_plot = np.arange(qMid, qMin-Q_INTERVAL, -Q_INTERVAL);
    dq_plot = -np.sqrt(np.max([np.zeros(q_plot.shape),2*MAX_ACC*(q_plot-qMin)],0));
    ax.plot(q_plot,dq_plot, 'r--');
    
    q_plot = np.arange(qMid, qMax+Q_INTERVAL, Q_INTERVAL);
    dq_plot = np.sqrt(np.max([np.zeros(q_plot.shape),1.4*MAX_ACC*(qMax-q_plot)],0));
    ax.plot(q_plot,dq_plot, 'b--');
    q_plot = np.arange(qMid, qMin-Q_INTERVAL, -Q_INTERVAL);
    dq_plot = -np.sqrt(np.max([np.zeros(q_plot.shape),1.4*MAX_ACC*(q_plot-qMin)],0));
    ax.plot(q_plot,dq_plot, 'b--');
    
    rq = np.arange(qMin, qMax+Q_INTERVAL, Q_INTERVAL);
    rdq = np.arange(-max_vel_from_pos_acc_bounds, max_vel_from_pos_acc_bounds+DQ_INTERVAL, DQ_INTERVAL);
    Q, DQ = np.meshgrid(rq, rdq);
    
    Z  = -np.ones(Q.shape)*MAX_ACC;
    T = 250;
    A = np.array([[1, DT], [0, 1]]);
    B = np.array([[0.5*DT**2], [DT]]);
    
    E = np.array([[-0.01], [-0.01]]);
    
    G     = computeControlTransitionMatrix(A, B, T);
    A_bar = computeStateTransitionMatrix(A, T);
    
    #G     = computeControlTransitionMatrix_with_error(A, B, T, E);
    #A_bar = computeStateTransitionMatrix_with_error(A, T, E);
    
    for i in range(len(rdq)):
        for j in range(len(rq)):
            if(isStateViable(Q[i,j], DQ[i,j], qMin, qMax, MAX_VEL, MAX_ACC)==0.0):
                if((DQ[i,j]>0.0 and Q[i,j]<=qMid - 0.5*(DQ[i,j]**2)/MAX_ACC) or (DQ[i,j]<0.0 and Q[i,j]<=qMid + 0.5 *(DQ[i,j]**2)/MAX_ACC)):
                    Z[i,j] = 100;
                else:
                    # compute probability that future positions dont violate bounds                    
                    x0 = np.array([Q[i,j], DQ[i,j]]);
                    a = MAX_ACC;
                    b = -2*x0[1];
                    c = qMid-x0[0]+x0[1]**2/(2*MAX_ACC);
                    delta = b**2 - 4*a*c;
                    t0 = (-b + np.sqrt(delta))/(2*a);
                    t_switch = int(np.ceil(t0/DT));  # time of switch # added int
                    ddq_plot = np.array(t_switch*[-MAX_ACC,]); 
                    
                    ineq_prob = InequalitiesProbability(np.array(t_switch*[std_dev,]));
                    G_pos = G[:2*t_switch:2,:t_switch];   # transition matrix from ddq to q
                    g = qMax - np.dot(A_bar[:2*t_switch,:], x0)[::2];
                    prob_ind = ineq_prob.computeIndividualProbabilities(ddq_plot, -G_pos, g);
                    prob = np.prod(prob_ind);
                    Z[i,j] = 100*prob;
    
    CF = ax.contourf(Q, DQ, Z, 10, cmap=cmap);
    f.colorbar(CF, ax=ax, shrink=0.9);
    ax.set_title('Prob that future positions dont violate bounds');
   
#    q_plot = np.zeros(T);
#    dq_plot = np.zeros(T);
#    ddq_plot = np.zeros(T);
#    q_plot[0] = qMin + 0.5*(qMax-qMin);
#    dq_plot[0] = 0.99*np.sqrt(2*MAX_ACC*(qMax-q_plot[0]));
#    for t in range(T-1):
#        if((dq_plot[t]>0.0 and q_plot[t]<=qMid - 0.5*(dq_plot[t]**2)/MAX_ACC) or 
#           (dq_plot[t]<0.0 and q_plot[t]<=qMid + 0.5 *(dq_plot[t]**2)/MAX_ACC)):
#            ddq_plot[t] = MAX_ACC;
#            print "Time %f starting to accelerate" % (t*DT);
#        else:
#            ddq_plot[t] = -MAX_ACC;
#        q_plot[t+1] = q_plot[t] + DT*dq_plot[t] + 0.5*(DT**2)*ddq_plot[t];
#        dq_plot[t+1] = dq_plot[t] + DT*ddq_plot[t];
#    ax.plot(q_plot,dq_plot,'k x');
#    
#    # compute probability that future positions dont violate bounds
#    x0 = np.array([q_plot[0], dq_plot[0]]);
#    
#    a = MAX_ACC;
#    b = -2*x0[1];
#    c = qMid-x0[0]+x0[1]**2/(2*MAX_ACC);
#    delta = b**2 - 4*a*c;
#    t0 = (-b + np.sqrt(delta))/(2*a);
#    t1 = (-b - np.sqrt(delta))/(2*a);
#    t_switch = max(t0, t1)/DT;
#    
#    x_plot = np.dot(A_bar,x0) + np.dot(G,ddq_plot);
#    q_plot_2 = x_plot[::2];
#    dq_plot_2 = x_plot[1::2];
#    ax.plot(q_plot_2, dq_plot_2, 'b o');
#    
#    ineq_prob = InequalitiesProbability(T, T, np.array(T*[std_dev,]));
#    G_pos = G[::2,:];   # transition matrix from ddq to q
#    g = qMax - np.dot(A_bar, x0)[::2];
#    prob_ind = ineq_prob.computeIndividualProbabilities(ddq_plot, -G_pos, g);
#    prob = np.prod(prob_ind);
#    pos_std_dev = np.sqrt(np.diag(ineq_prob.Sigma_eG));
#    ax.plot(q_plot_2 - 2*pos_std_dev, dq_plot_2, 'b--');
#    ax.plot(q_plot_2 + 2*pos_std_dev, dq_plot_2, 'b--');
#    
#    ax.xaxis.set_ticks([qMin,qMax]);
#    ax.yaxis.set_ticks([-max_vel_from_pos_acc_bounds, max_vel_from_pos_acc_bounds]);
#    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
#    ax.set_xlim([1.1*qMin, 1.1*qMax]);
#    ax.set_ylim([-1.1*max_vel_from_pos_acc_bounds, 1.1*max_vel_from_pos_acc_bounds]);
#    ax.set_xlabel('q');
#    ax.set_ylabel('dq');


if(PLOT_STATE_SPACE):    
    (f,ax) = create_empty_figure(1,1, [0,0]);

    # # plot viability constraints
    # qMid = 0.5*(qMin+qMax);
    # q_mid_2_max = np.arange(qMid, qMax+Q_INTERVAL, Q_INTERVAL);
    # dq_viab_pos = np.sqrt(np.max([np.zeros(q_mid_2_max.shape),2*MAX_ACC*(qMax-q_mid_2_max)],0));
    # line_viab, = ax.plot(q_mid_2_max, dq_viab_pos, 'r--');
    # q_min_2_mid = np.arange(qMid, qMin-Q_INTERVAL, -Q_INTERVAL);
    # dq_viab_neg = -np.sqrt(np.max([np.zeros(q_min_2_mid.shape),2*MAX_ACC*(q_min_2_mid-qMin)],0));
    # ax.plot(q_min_2_mid, dq_viab_neg, 'r--');
    
    # # plot implicit constraints
    # t_max = np.sqrt((qMax-qMin)/MAX_ACC);
    # t = np.arange(0, t_max, 0.001);
    # q_plot = qMax - 0.5*(t**2)*MAX_ACC;
    # dq_plot = -t*MAX_ACC;
    # line_impl, = ax.plot(q_plot,dq_plot, 'b--');
    # q_plot = qMin + 0.5*(t**2)*MAX_ACC;
    # dq_plot = t*MAX_ACC;
    # ax.plot(q_plot,dq_plot, 'b--');
    
    # dq_viab_neg[np.where(dq_viab_neg < -MAX_VEL)[0]] = -MAX_VEL;
    # dq_viab_pos[np.where(dq_viab_pos >  MAX_VEL)[0]] =  MAX_VEL;
    # ax.fill_between(q_min_2_mid, dq_viab_neg, -1.0*dq_viab_neg, alpha=0.25, linewidth=0, facecolor='green');
    # ax.fill_between(q_mid_2_max, -1.0*dq_viab_pos, dq_viab_pos, alpha=0.25, linewidth=0, facecolor='green');
    
    # plot velocity bounds
    line_vel, = ax.plot([qMin, qMax], [MAX_VEL, MAX_VEL], 'k--');
    ax.plot([qMin, qMax], [-MAX_VEL, -MAX_VEL], 'k--');
    
    # plot position bounds
    line_pos, = ax.plot([qMin, qMin], [-MAX_VEL, MAX_VEL], 'k--');
    ax.plot([qMax, qMax], [-MAX_VEL, MAX_VEL], 'k--');
    
#    if(PLOT_STATE_SPACE_DECRE):
#        dq_plot = np.arange(0, MAX_VEL, DQ_INTERVAL);
#        q_plot = np.zeros(dq_plot.shape);
#        for i in range(dq_plot.shape[0]):
#            n = 0.5 + dq_plot[i]/MAX_ACC;
#            n_floor = math.floor(n);
#            n_ceil = math.ceil(n);
#            p_floor = qMax - n_floor*DT*dq_plot[i] + 0.5*(n_floor**2 - n_floor)*MAX_ACC*DT*DT;
#            p_ceil  = qMax - n_ceil*DT*dq_plot[i] + 0.5*(n_ceil**2 - n_ceil)*MAX_ACC*DT*DT;
#            q_plot[i] = min(p_floor, p_ceil);
#        line_decre2, = ax.plot(q_plot,dq_plot, 'y--');
        
    # if(PLOT_STATE_SPACE_DECRE):
    #     q_plot = np.arange(qMin, qMax+Q_INTERVAL, Q_INTERVAL);
    #     dq_plot = np.zeros(q_plot.shape);
    #     for i in range(dq_plot.shape[0]):
    #         n = np.sqrt(max(0, 2*MAX_ACC*(qMax-q_plot[i]))) / (MAX_ACC*DT);
    #         n_floor = math.floor(n);
    #         n_ceil = math.ceil(n);
    #         if(n_floor==0):
    #             dq_plot[i] = 0;
    #         else:
    #             p_floor = (qMax-q_plot[i])/(n_floor*DT) + 0.5*(n_floor-1)*MAX_ACC*DT;
    #             p_ceil  = (qMax-q_plot[i])/(n_ceil*DT) + 0.5*(n_ceil-1)*MAX_ACC*DT;
    #             dq_plot[i] = min(p_floor, p_ceil);
    #     line_decre, = ax.plot(q_plot,dq_plot, 'y--');
        
    #     q_plot = np.arange(qMax, qMin-Q_INTERVAL, -Q_INTERVAL);
    #     dq_plot = np.zeros(q_plot.shape);
    #     for i in range(dq_plot.shape[0]):
    #         s = np.sqrt(max(0, 2*MAX_ACC*(q_plot[i]-qMin))) / (MAX_ACC*DT);
    #         dq_plot[i] = ((qMin-q_plot[i]) - 0.5*(s**2-s)*MAX_ACC*DT*DT) / ((s+1)*DT);
    #     #ax.plot(q_plot,dq_plot, 'g--');
     
    # plot Padois constraints
    # if(PLOT_STATE_SPACE_PADOIS):
    #     q_plot = np.arange(qMin, qMax+Q_INTERVAL, Q_INTERVAL);
    #     dq_plot = np.zeros(q_plot.shape);
    #     for i in range(dq_plot.shape[0]):
    #         s = np.sqrt(max(0, 2*MAX_ACC*(qMax-q_plot[i]))) / (MAX_ACC*DT);
    #         dq_plot[i] = ((qMax-q_plot[i]) + 0.5*(s**2-s)*MAX_ACC*DT*DT) / ((s+1)*DT);
    #     line_padois, = ax.plot(q_plot,dq_plot, 'g--');
        
    #     q_plot = np.arange(qMax, qMin-Q_INTERVAL, -Q_INTERVAL);
    #     dq_plot = np.zeros(q_plot.shape);
    #     for i in range(dq_plot.shape[0]):
    #         s = np.sqrt(max(0, 2*MAX_ACC*(q_plot[i]-qMin))) / (MAX_ACC*DT);
    #         dq_plot[i] = ((qMin-q_plot[i]) - 0.5*(s**2-s)*MAX_ACC*DT*DT) / ((s+1)*DT);
    #     ax.plot(q_plot,dq_plot, 'g--');
    
    #     leg = ax.legend([line_viab, line_padois, line_decre, line_impl, line_pos],
    #             ['Viability', 'Padois', 'Decre',
    #             'Implicit pos-acc', 
    #             'Pos-vel bounds'],
    #             bbox_to_anchor=(-0.1, 1.02, 1.2, .102), loc=2, ncol=3, mode="expand", borderaxespad=0.);
    # else:
    #     leg = ax.legend([line_viab, line_impl, line_pos],
    #             ['Viability', 
    #             'Implicit pos-acc', 
    #             'Pos-vel bounds'],
    #             bbox_to_anchor=(-0.1, 1.02, 1.2, .102), loc=2, ncol=3, mode="expand", borderaxespad=0.);
    # leg.get_frame().set_alpha(0.6);
    
    
    lege1=mpatches.Patch(color='blue',label='Implicit pos-acc');
    lege2=mpatches.Patch(color='red',label='Viability');
    lege3=mpatches.Patch(color='black',label='Pos-vel bounds');
    lege4=mpatches.Patch(color='yellow',label='Viability Robust');
    lege5=mpatches.Patch(color='green',label='Implicit pos-acc Robust');
    ax.legend(handles=[lege1,lege2,lege3], loc='upper center',bbox_to_anchor=(0.5, 1.0),
                bbox_transform=plt.gcf().transFigure,ncol=5,fontsize=30 );
    # plot trajectory
    ax.plot(q,dq,'k x');

    ax.xaxis.set_ticks([0]);
    ax.yaxis.set_ticks([]);
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
    ax.set_xlim([1.1*qMin, 1.1*qMax]);
    ax.set_ylim([-1.1*max_vel_from_pos_acc_bounds, 1.1*max_vel_from_pos_acc_bounds]);
    #ax.set_xlabel(r'$q$');
    #ax.set_ylabel(r'$\dot{q}$');
    
    # ax.set_xlim([qMin,qMax])
    ax.set_ylim([-MAX_VEL-0.025,MAX_VEL+0.025])
    
    ax.annotate(r'$q$', xy=(0.485, 0), ha='left', va='top', xycoords='axes fraction', fontsize=60)
    ax.annotate(r'$\dot{q}$', xy=(0, 0.61), ha='left', va='top', xycoords='axes fraction', fontsize=60)
    plt.savefig('/home/erik/Desktop/Viab_pres_2.png')
    

    
if(N_TESTS>2 and PLOT_SIMULATION_RESULTS):
#    mpl.rcParams['figure.figsize']      = 12, 6
    (f,ax) = create_empty_figure(3,1);
    t = np.arange(0, (N_TESTS+1)*DT, DT)[:N_TESTS+1]

    SMALL_RATIO = 10;
    DT_SMALL = DT/SMALL_RATIO;
    t_small = np.arange(0, N_TESTS*DT+DT_SMALL, DT_SMALL);
    q_small = np.zeros(t_small.shape);
    dq_small = np.zeros(t_small.shape);
    q_small[0] = q[0];
    dq_small[0] = dq[0];
    j = 0;
    for i in range(t_small.shape[0]-1):
        dq_small[i+1] = dq_small[i] + DT_SMALL*ddq[j];
        q_small[i+1]  = q_small[i] + DT_SMALL*dq_small[i] + 0.5*(DT_SMALL**2)*ddq[j];
        if((i+1)%SMALL_RATIO==0):
            j += 1;
    
    ax[0].plot(t_small, q_small, linewidth=LW);
    ax[0].plot([0, t[-1]], [qMax, qMax], '--', color=plut.BOUNDS_COLOR, alpha=plut.LINE_ALPHA);
    ax[0].plot([0, t[-1]], [qMin, qMin], '--', color=plut.BOUNDS_COLOR, alpha=plut.LINE_ALPHA);
    #ax[0].set_title('position');
    ax[0].set_ylabel(r'$q$ [rad]');
    ax[0].set_xlim([0, t[-1]]);
    ax[0].set_ylim([np.min(q_small)-0.1, np.max(q_small)+0.1]);
    ax[0].yaxis.set_ticks([np.min(q_small), np.max(q_small)]);
    ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
    
    #print 't', t.shape, 'dq', dq.shape;
    ax[1].plot(t, dq, linewidth=LW);
    ax[1].plot([0, t[-1]], [MAX_VEL, MAX_VEL], '--', color=plut.BOUNDS_COLOR, alpha=plut.LINE_ALPHA);
    ax[1].plot([0, t[-1]], [-MAX_VEL, -MAX_VEL], '--', color=plut.BOUNDS_COLOR, alpha=plut.LINE_ALPHA);
    ax[1].set_ylim([np.min(dq)-0.1*MAX_VEL, np.max(dq)+0.1*MAX_VEL]);
    if(np.max(dq)>0.0 and np.min(dq)<0):
        ax[1].yaxis.set_ticks([np.min(dq), 0, np.max(dq)]);
    else:        
        ax[1].yaxis.set_ticks([np.min(dq), np.max(dq)]);
    ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
    #ax[1].set_title('velocity');
    ax[1].set_ylabel(r'$\dot{q}$ [rad/s]');
    ax[1].set_xlim([0, t[-1]]);
    
    
    ax[2].step(t, np.hstack((ddq[0], ddq)), linewidth=LW);
    ax[2].plot([0, t[-1]], [MAX_ACC, MAX_ACC], '--', color=plut.BOUNDS_COLOR, alpha=plut.LINE_ALPHA);
    ax[2].plot([0, t[-1]], [-MAX_ACC, -MAX_ACC], '--', color=plut.BOUNDS_COLOR, alpha=plut.LINE_ALPHA);
    #ax[2].plot(t[:-1], ddqLB, "--", color='red', alpha=plut.LINE_ALPHA);
    #ax[2].plot(t[:-1], ddqUB, "--", color='red', alpha=plut.LINE_ALPHA);
    #ax[2].set_title('acceleration');
    ax[2].set_xlabel('Time [s]');
    ax[2].set_ylabel(r'$\ddot{q}$ [rad/s${}^2$]');
    ax[2].set_xlim([0, t[-1]]);
    ax[2].set_ylim([np.min(ddq)-0.1*MAX_ACC, np.max(ddq)+0.1*MAX_ACC]);
    ax[2].yaxis.set_ticks([np.min(ddq), np.max(ddq)]);
    ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
    
    plut.saveFigure('max_acc_traj_'+str(int(1e3*DT))+'_ms');
    
plt.show();
    