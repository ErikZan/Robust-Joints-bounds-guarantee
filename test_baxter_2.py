# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:52:00 2016

@author: adelpret
"""

import pinocchio as se3
from pinocchio.utils import *
import os
import numpy.matlib
from numpy.linalg import pinv,inv
from math import sqrt
from time import sleep

import plot_utils as plut
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import datetime
from plot_utils import create_empty_figure

from qp_solver import qpSolver
from acc_bounds_util_2e import computeMultiAccLimits_3,isBoundsTooStrict_Multi , DiscreteViabilityConstraints_Multi
from baxter_wrapper import BaxterWrapper, Q_MIN, Q_MAX, DQ_MAX, TAU_MAX, MODELPATH
                
def plot_bounded_joint_quantity(time, x, X_MIN, X_MAX, name, xlabel='', ylabel=''):
    mpl.rcParams['font.size']       = 30;
    mpl.rcParams['axes.labelsize']  = 30;
    f, ax = plut.create_empty_figure(4,2);
    ax = ax.reshape(8);
    for j in range(7):
        ax[j].plot(time, x[j,:].T, linewidth=LW);
        ax[j].plot([time[0], time[-1]], [X_MIN[j], X_MIN[j]], 'r--');
        ax[j].plot([time[0], time[-1]], [X_MAX[j], X_MAX[j]], 'r--');
        ax[j].set_title('Joint '+str(j));
        ax[j].set_ylabel(ylabel);
        ax[j].set_ylim([np.min(x[j,:]) - 0.1*(X_MAX[j]-X_MIN[j]), np.max(x[j,:]) + 0.1*(X_MAX[j]-X_MIN[j])])
    ax[6].set_xlabel(xlabel);
    ax[7].set_xlabel(xlabel);
    ax[0].set_title(name);
    #plut.saveFigure(TEST_NAME+'_'+name);
    plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'_'+name,1)
    return ax;

# Convertion from translation+quaternion to SE3 and reciprocally
q2m = lambda q: se3.SE3( se3.Quaternion(q[6,0],q[3,0],q[4,0],q[5,0]).matrix(), q[:3])
m2q = lambda M: np.concatenate([ M.translation,se3.Quaternion(M.rotation).coeffs() ])

''' PLOT-RELATED USER PARAMETERS '''
LW = 4;     # line width
PLOT_END_EFFECTOR_POS = True;
PLOT_END_EFFECTOR_ACC = True;
PLOT_JOINT_POS_VEL_ACC_TAU = 0;
Q_INTERVAL = 0.001; # the range of possible angles is sampled with this step for plotting
PLAY_TRAJECTORY_ONLINE = False;
PLAY_TRAJECTORY_AT_THE_END = True;
CAPTURE_IMAGES = True;
plut.SAVE_FIGURES = 1;
PLOT_SINGULAR = 1;
#plut.FIGURE_PATH = '/home/erik/Desktop/Thesis/figures/baxter/'; # old path => SaveFigurewithDire... usa GarbageFlder now
IMAGES_FILE_NAME = 'baxter_viab_dt_2x';
BACKGROUND_COLOR = (1.0, 1.0, 1.0, 1.0);

DATE_STAMP=datetime.datetime.now().strftime("%m_%d__%H_%M_%S")
GARBAGE_FOLDER='/home/erik/Desktop/FIGURES_T/Baxter/'+DATE_STAMP+'/'
os.makedirs(GARBAGE_FOLDER);
''' END OF PLOT-RELATED USER PARAMETERS '''

''' CONTROLLER USER PARAMETERS '''
CTRL_LAW = 'IK_QP'; #'IK_QP', 'IK'
ACC_BOUNDS_TYPE = 'VIAB'; #'VIAB', 'NAIVE'
CONSTRAIN_JOINT_TORQUES = False;
END_EFFECTOR_NAME = 'left_w2'; #'left_w2'; left_wrist
W_POSTURE = 1.0e-3; # 1e-3
T = 5.0;    # total simulation time
DT = 0.01;  # time step
DT_SAFE =1.01*DT; # 2*DT;
kp = 200; # 10 default , 100 improve performance with error
kp_post = 200;
kd = 2*sqrt(kp);
kd_post = 2*sqrt(kp_post);

#x_des = np.array([[0.3, 0.30, 1.23, 0.15730328, 0.14751489, 0.48883663,  0.845301]]).T;
x_des = np.array([[0.3, 0.3, 1.23, 0.15730328, 0.14751489, 0.48883663,  0.845301]]).T; #x_des = np.array([[0.5,  0.5 ,  1.5, -0.01354349,  0.0326968 , 0.38244455,  0.92330042]]).T;
#DDQ_MAX = 12.0*np.ones(14);
DDQ_MAX = np.array([ 12.0, 2.0, 30.0, 30.0, 30.0, 30.0, 30.0,     
                     12.0 ,2.0, 30.0, 30.0, 30.0, 30.0, 30.0]);
#DDQ_MAX = np.array([ 12.0, 2.0, 33.0, 54.0, 358.0, 485.0, 26257.0,     
#                     12.0 ,2.0, 33.0, 54.0, 358.0, 485.0, 26257.0]);
#DDQ_MIN = np.array([-12.0, -2.0, -33.0, -54.0, -358.0, -485.0, -26257.0,    
#                    -12.0, -2.0, -33.0, -54.0, -358.0, -485.0, -26257.0])
E = np.array([ 12.0, 2.0, 30.0, 30.0, 30.0, 30.0, 30.0,     
                     .0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0])*0.33 ; # DDQ_MAX[2]*0.3;
q0 = np.array([ 0. , -0.0,  0. ,  0.0,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 0. ,  0. ,  0. ]) ;# q0 = np.array([ 0. , -0.1,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 0. ,  0. ,  0. ])

# check if bounds are OK
#isBoundsTooStrict_Multi(Q_MIN,Q_MAX,DQ_MAX,DDQ_MAX,DT,E)
DiscreteViabilityConstraints_Multi(Q_MIN,Q_MAX,DQ_MAX,DDQ_MAX,DT,E)
Q_POSTURE = np.array(0.5*(Q_MIN+Q_MAX));
Q_POSTURE[8:] = 0.0;
''' END OF CONTROLLER USER PARAMETERS '''

TEST_NAME = 'baxter_'+ACC_BOUNDS_TYPE+'_dt_'+str(int(DT_SAFE/DT));

M_des = q2m(x_des);
robot = BaxterWrapper();
robot.initViewer(loadModel=False)
robot.loadViewerModel( "pinocchio"); #, MODELPATH)
robot.viewer.gui.setCameraTransform('python-pinocchio',[4.000033378601074, -5.577442721005355e-07, -8.179827659660077e-07, 0.5338324904441833, 0.5607414841651917, 0.4348451793193817, 0.45989298820495605]); #[3.5000033378601074, -7.042121978884097e-07, -5.638392508444667e-07, 0.5374045968055725, 0.5444704294204712, 0.4312002956867218, 0.47834569215774536])
robot.viewer.gui.setLightingMode('world/floor', 'OFF');
robot.viewer.gui.setVisibility('world/floor', 'OFF');
assert(robot.model.existFrame(END_EFFECTOR_NAME))
#robot.viewer.gui.setBackgroundColor(robot.windowID, BACKGROUND_COLOR);
#print robot.model

robot.display(q0);           # Display the robot in Gepetto-Viewer.

#IDEE = robot.model.getBodyId(END_EFFECTOR_NAME); # Access to the index of the end-effector
IDEE = robot.model.getJointId(END_EFFECTOR_NAME); # Access to the index of the end-effector

print('Initial e-e- pose: x(0)', m2q(robot.position(q0,IDEE)).T);
print('Desired e-e pose: x_des', x_des.T)

NQ = q0.shape[0];
if(CONSTRAIN_JOINT_TORQUES):
    solver = qpSolver('ik', NQ, NQ);
else:
    solver = qpSolver('ik', NQ, 0);

# spatial velocity to go from M to Mdes, expressed in frame M, is log(M.inverse()*Mdes)
NT = int(T/DT);
q = np.zeros((NQ,NT));
dq = np.zeros((NQ,NT));
ddq = np.zeros((NQ,NT-1));
tau = np.zeros((NQ,NT-1));
x = np.zeros((7,NT));
dx = np.zeros((6,NT));
ddx = np.zeros((6,NT));
ddx_des = np.zeros((6,NT));
ddq_lb = np.zeros((NQ,NT-1));
ddq_ub = np.zeros((NQ,NT-1));

''' initialize '''
q[:,0] = q0;
M = robot.position(q[:,0],IDEE);
J = robot.jacobian(q[:,0],IDEE); # always zero ??? why ???
dJdq = robot.dJdq(q[:,0], dq[:,0], IDEE);
x[:,0] = m2q(M);
GOAL_REACHED = False;
for t in range(NT-1):
    ddx_des[:,t] = kp*se3.log(M.inverse()*M_des).vector - kd*J@dq[:,t];
    ddq_post_des = kp_post*(Q_POSTURE - q[:,t]) - kd_post*dq[:,t];
    MM = robot.mass(q[:,t]);
    h = robot.bias(q[:,t], dq[:,t]);
    
#    ddx_des2[:3,t] = kp*M.rotation.T*(M_des.translation - M.translation) - kd*J[:3,:]*dq[:,t];
        
    if(GOAL_REACHED==False and np.linalg.norm(M_des.translation - M.translation) < 3e-6): #3e-3
        print("Final e-e configuration reached at time %f" % (t*DT));
        GOAL_REACHED = True;
        
    if(CTRL_LAW=='IK'):
       #ddq[:,t] = pinv(J, W_POSTURE)@(ddx_des[:,t] - dJdq.vector);     # ddq[:,t] = pinv(J, W_POSTURE)*(ddx_des[:,t] - dJdq.vector);     
       ddq[:,t] = J.T@inv(J@J.T+W_POSTURE*np.identity(J.shape[0]))@(ddx_des[:,t] - dJdq.vector)- kd_post*dq[:,t];    
    elif(CTRL_LAW=='IK_QP'):
        if(ACC_BOUNDS_TYPE=='VIAB'):
            (ddq_lb[:,t], ddq_ub[:,t]) = computeMultiAccLimits_3(q[:,t], dq[:,t], Q_MIN, Q_MAX, DQ_MAX, DDQ_MAX, DT_SAFE,E);
        elif(ACC_BOUNDS_TYPE=='NAIVE'):
            for j in range(NQ):
                ddq_ub[j,t] = min( DDQ_MAX[j], ( DQ_MAX[j]-dq[j,t])/DT_SAFE, 2.0*(Q_MAX[j]-q[j,t]-DT_SAFE*dq[j,t])/(DT_SAFE**2));
                ddq_lb[j,t] = max(-DDQ_MAX[j], (-DQ_MAX[j]-dq[j,t])/DT_SAFE, 2.0*(Q_MIN[j]-q[j,t]-DT_SAFE*dq[j,t])/(DT_SAFE**2));
                if(ddq_lb[j,t] > DDQ_MAX[j]):
                    ddq_lb[j,t] = DDQ_MAX[j];
                if(ddq_ub[j,t] < -DDQ_MAX[j]):
                    ddq_ub[j,t] = -DDQ_MAX[j];
        a = ddx_des[:,t] - dJdq.vector;
        hess = np.dot(J.T, J) + W_POSTURE*np.identity(NQ);
        grad = -np.dot(J.T, a) - W_POSTURE*ddq_post_des;
        if(CONSTRAIN_JOINT_TORQUES):
            b_lb = -TAU_MAX - h;
            b_ub =  TAU_MAX - h;
            ddq_des = solver.solve(hess, grad, ddq_lb[:,t], ddq_ub[:,t], 1.0*MM, 1.0*b_lb, 1.0*b_ub);
        else:
            ddq_des = solver.solve(hess, grad, ddq_lb[:,t], ddq_ub[:,t]);
        ddq[:,t] = ddq_des+E;
    else:
        print("Error unrecognized control law:", CTRL_LAW);

    ddx[:,t] = J@ddq[:,t] + dJdq.vector;
    ddx[:3,t] = M.rotation@ddx[:3,t];
    ddx[3:,t] = M.rotation@ddx[3:,t];
    ddx_des[:3,t] = M.rotation@ddx_des[:3,t];
    ddx_des[3:,t] = M.rotation@ddx_des[3:,t];        
    tau[:,t] = MM@ddq[:,t] + h;
    
    ''' Numerical Integration '''
    q[:,t+1] = q[:,t] + DT*dq[:,t] + 0.5*DT*DT*ddq[:,t];
    dq[:,t+1] = dq[:,t] + DT*ddq[:,t];
    if(PLAY_TRAJECTORY_ONLINE):
        robot.display(q[:,t+1]);
        # sleep(DT); ###### prima era commentata
    
    ''' store data '''
    M = robot.position(q[:,t+1],IDEE);
    J = robot.jacobian_2(q[:,t+1],IDEE);
    #print(robot.computeJointJacobian(q[:,t+1],IDEE));
    #print(robot.getJointJacobian(IDEE));
    dJdq = robot.dJdq(q[:,t+1], dq[:,t+1], IDEE);
    x[:,t+1] = m2q(M);
    dx[:,t+1] = J@dq[:,t+1];

print('Final e-e pose x(T))', m2q(M).T);
print('Difference between desired and measured e-e pose: M.inverse()*M_des', m2q(M.inverse()*M_des).T)

if(PLAY_TRAJECTORY_AT_THE_END):
    if(CAPTURE_IMAGES):
        robot.startCapture(IMAGES_FILE_NAME,path=GARBAGE_FOLDER+'/img/');
    robot.play(q, DT);
    if(CAPTURE_IMAGES):
        robot.stopCapture();

time = np.arange(0, NT*DT, DT);
if(PLOT_END_EFFECTOR_POS):
    f, ax = plut.create_empty_figure(3,1);
    for i in range(3):
        ax[i].plot(time, x[i,:].T, linewidth=LW);
        ax[i].plot([time[0], time[-1]], [x_des[i], x_des[i]], 'r--');
    ax[0].legend(['Measured', 'Desired']);
    ax[0].set_ylabel('X [m]');
    ax[1].set_ylabel('Y [m]');
    ax[2].set_ylabel('Z [m]');
    ax[2].set_xlabel('Time [s]');
    ax[0].yaxis.set_ticks([x[0,0], x[0,-1]]);
    ax[1].yaxis.set_ticks([x[1,0], x[1,-1]]);
    ax[2].yaxis.set_ticks([x[2,0], x[2,-1]]);
    ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'));
    ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'));
    ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'));
    ax[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'));
    plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'_ee',1)
    ax[0].set_title('End-effector position');

if(PLOT_END_EFFECTOR_ACC):
    f, ax = plut.create_empty_figure(3,1);
    for i in range(3):
        ax[i].plot(time, ddx[i,:].T, 'b');
        ax[i].plot(time, ddx_des[i,:].T, 'r--');
    ax[0].set_title('End-effector acceleration');

if(PLOT_JOINT_POS_VEL_ACC_TAU):
    plot_bounded_joint_quantity(time,        q,       Q_MIN,   Q_MAX, 'Joint positions', 'Time [s]', r'$q$ [rad]');
    plot_bounded_joint_quantity(time,       dq,   -1*DQ_MAX,  DQ_MAX, 'Joint velocities', 'Time [s]', r'$q$ [rad/s]');
    plot_bounded_joint_quantity(time[:-1], ddq,  -1*DDQ_MAX, DDQ_MAX, 'Joint accelerations', 'Time [s]', r'$\ddot{q}$ [rad/s${}^2$]');
    plot_bounded_joint_quantity(time[:-1], tau,  -1*TAU_MAX, TAU_MAX, 'Joint torques', 'Time [s]', r'$\tau$ [Nm]');

for j in range(7):
    qMax = Q_MAX[j];
    qMin = Q_MIN[j]
    qMid = 0.5*(qMin+qMax);
    qHalf = 0.5*(qMax-qMin);
    max_vel_from_pos_acc_bounds = np.sqrt(DDQ_MAX[j]*(qMax-qMin));
    
    if(qMax-np.max(q[j,:]) < 0.1*qHalf or 
        np.min(q[j,:])-qMin < 0.1*qHalf or 
        np.max(np.abs(dq[j,:])) > DQ_MAX[j] or
        np.max(np.abs(tau[j,:])) > TAU_MAX[j]):
                
        if(np.max(q[j,:]) > qMax):
            print("Joint %d hit up bound with vel %f" % (j, dq[j, np.where(q[j,:]>qMin)[0][0]]));
        elif(qMax-np.max(q[j,:]) < 0.1*qHalf):
            print("Joint %d got close to up bound, qMax=%f, max(q)=%f" % (j, qMax, np.max(q[j,:])));
            
        if(np.min(q[j,:]) < qMin):
            print("Joint %d hit low bound with vel %f" % (j, dq[j, np.where(q[j,:]<qMin)[0][0]]));
        elif(np.min(q[j,:])-qMin < 0.1*qHalf):
            print("Joint %d got close to low bound, min(q)=%f, qMin=%f" % (j, np.min(q[j,:]), qMin));
            
        if(np.max(np.abs(dq[j,:])) > 0.99*DQ_MAX[j]):
            print("Joint %d got close to max vel, max(dq)=%f, max_vel=%f" % (j, np.max(np.abs(dq[j,:])), DQ_MAX[j]));
        if(np.max(np.abs(tau[j,:])) > 0.99*TAU_MAX[j]):
            print("Joint %d got close to max torque, max(tau)=%f, max_tau=%f" % (j, np.max(np.abs(tau[j,:])), TAU_MAX[j]));

        # plot acceleration
        f, ax = plt.subplots(3, 2, sharex=True);
        ax = ax.reshape(6);
        plut.movePlotSpines(ax[5], [0, 0]);
        """ ax[5].plot(time[:-1], ddq[j,:], linewidth=LW); # ax[5].plot(time[:-1], ddq[j,:].A.squeeze(), linewidth=LW);
        ax[5].plot(time[:-1], ddq_lb[j,:], 'o--');
        ax[5].plot(time[:-1], ddq_ub[j,:], 'g--'); """
        ax[5].step(time[:-1], ddq[j,:], linewidth=LW); # ax[5].plot(time[:-1], ddq[j,:].A.squeeze(), linewidth=LW);
        ax[5].step(time[:-1], ddq_lb[j,:], 'o--');
        ax[5].step(time[:-1], ddq_ub[j,:], 'g--');
        #ax[5].step(time[:-1], (ddq_lb[j,:]),color='yellow', linewidth=LW);
        ax[5].set_ylabel(r'$\ddot{q}$ [rad/s${}^2$]');
        ax[5].set_xlabel('Time [s]');
        
        # plot velocity
        plut.movePlotSpines(ax[3], [0, 0]);
        ax[3].plot(time, dq[j,:], linewidth=LW); #.A.squeeze()
        ax[3].plot([time[0], time[-1]], [DQ_MAX[j], DQ_MAX[j]], 'r--');
        ax[3].plot([time[0], time[-1]], [-DQ_MAX[j], -DQ_MAX[j]], 'r--');
        ax[3].set_ylabel(r'$\dot{q}$ [rad/s]');
        ax[3].set_xlabel('Time [s]');
        
        # plot position
        plut.movePlotSpines(ax[1], [qMin, 0]);
        ax[1].plot(time, q[j,:], linewidth=LW); # .A.squeeze()
        ax[1].plot([time[0], time[-1]], [Q_MAX[j], Q_MAX[j]], 'r--');
        ax[1].plot([time[0], time[-1]], [Q_MIN[j], Q_MIN[j]], 'r--');
        ax[1].set_ylabel(r'$q$ [rad]');
        ax[1].set_xlabel('Time [s]');
        
        ax = plt.subplot(1, 2, 1);
        # plot viability constraints
        q_plot = np.arange(qMid, qMax+Q_INTERVAL, Q_INTERVAL);
        dq_plot = np.sqrt(np.max([np.zeros(q_plot.shape),2*DDQ_MAX[j]*(qMax-q_plot)],0));
        ind = np.where(dq_plot<= DQ_MAX[j])[0];
        ax.plot(q_plot[ind],dq_plot[ind], 'r--');
        q_plot = np.arange(qMid, qMin-Q_INTERVAL, -Q_INTERVAL);
        dq_plot = -np.sqrt(np.max([np.zeros(q_plot.shape),2*DDQ_MAX[j]*(q_plot-qMin)],0));
        ind = np.where(dq_plot >= -DQ_MAX[j])[0];
        ax.plot(q_plot[ind],dq_plot[ind], 'r--');
        ax.plot([qMin, qMax], [+DQ_MAX[j], +DQ_MAX[j]], 'k--');
        ax.plot([qMin, qMax], [-DQ_MAX[j], -DQ_MAX[j]], 'k--');
        ax.plot([qMin, qMin], [-DQ_MAX[j], +DQ_MAX[j]], 'k--');
        ax.plot([qMax, qMax], [-DQ_MAX[j], +DQ_MAX[j]], 'k--');
        # plot state-space trajectory
        ax.plot(q[j,:], dq[j,:], 'b-', linewidth=LW); # .A.squeeze() .A.squeeze()
        ax.xaxis.set_ticks([qMin, q[j,0], qMax]);
        ax.yaxis.set_ticks([-DQ_MAX[j], 0, DQ_MAX[j]]);
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'));
        ax.set_xlim([np.min(q[j,:]) -0.05*qHalf, np.max(q[j,:])+0.05*qHalf]);
        ax.set_ylim([np.min(dq[j,:])-0.05*DQ_MAX[j], np.max(dq[j,:])+0.05*DQ_MAX[j]]);
        ax.set_title('Joint '+str(j));
        ax.set_xlabel(r'$q$ [rad]');
        ax.set_ylabel(r'$\dot{q}$ [rad/s]');
        
        plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'_j'+str(j),1);
        # Plot without Joint space
        f, ax = plt.subplots(3, 1, sharex=True);
        ax = ax.reshape(3);
        plut.movePlotSpines(ax[2], [0, 0]);
        """ ax[5].plot(time[:-1], ddq[j,:], linewidth=LW); # ax[5].plot(time[:-1], ddq[j,:].A.squeeze(), linewidth=LW);
        ax[5].plot(time[:-1], ddq_lb[j,:], 'o--');
        ax[5].plot(time[:-1], ddq_ub[j,:], 'g--'); """
        ax[2].step(time[:-1], ddq[j,:], linewidth=LW); # ax[5].plot(time[:-1], ddq[j,:].A.squeeze(), linewidth=LW);
        ax[2].step(time[:-1], ddq_lb[j,:], 'o--',linewidth=LW);
        ax[2].step(time[:-1], ddq_ub[j,:], 'g--',linewidth=LW);
        #ax[5].step(time[:-1], (ddq_lb[j,:]),color='yellow', linewidth=LW);
        ax[2].set_ylabel(r'$\ddot{q}$ [rad/s${}^2$]');
        ax[2].set_xlabel('Time [s]');
        
        # plot velocity
        plut.movePlotSpines(ax[1], [0, 0]);
        ax[1].plot(time, dq[j,:], linewidth=LW); #.A.squeeze()
        ax[1].plot([time[0], time[-1]], [DQ_MAX[j], DQ_MAX[j]], 'r--');
        ax[1].plot([time[0], time[-1]], [-DQ_MAX[j], -DQ_MAX[j]], 'r--');
        ax[1].set_ylabel(r'$\dot{q}$ [rad/s]');
        ax[1].set_xlabel('Time [s]');
        
        # plot position
        plut.movePlotSpines(ax[2], [qMin, 0]);
        ax[0].plot(time, q[j,:], linewidth=LW); # .A.squeeze()
        ax[0].plot([time[0], time[-1]], [Q_MAX[j], Q_MAX[j]], 'r--');
        ax[0].plot([time[0], time[-1]], [Q_MIN[j], Q_MIN[j]], 'r--');
        ax[0].set_ylabel(r'$q$ [rad]');
        ax[0].set_xlabel('Time [s]');
        plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'pva_j'+str(j),1);
            
        # Plot singular
        if(PLOT_SINGULAR):
            (f, ax_acce) = create_empty_figure(1);
            """ ax[5].plot(time[:-1], ddq[j,:], linewidth=LW); # ax[5].plot(time[:-1], ddq[j,:].A.squeeze(), linewidth=LW);
            ax[5].plot(time[:-1], ddq_lb[j,:], 'o--');
            ax[5].plot(time[:-1], ddq_ub[j,:], 'g--'); """
            ax_acce.step(time[:-1], ddq[j,:], linewidth=LW); # ax[5].plot(time[:-1], ddq[j,:].A.squeeze(), linewidth=LW);
            ax_acce.step(time[:-1], ddq_lb[j,:], 'o--');
            ax_acce.step(time[:-1], ddq_ub[j,:], 'g--');
            #ax[5].step(time[:-1], (ddq_lb[j,:]),color='yellow', linewidth=LW);
            ax_acce.set_ylabel(r'$\ddot{q}$ [rad/s${}^2$]');
            ax_acce.set_xlabel('Time [s]');
            plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'pos_j'+str(j),1);
            
            # plot velocity
            (f, ax_vel) = create_empty_figure(1);
            ax_vel.plot(time, dq[j,:], linewidth=LW); #.A.squeeze()
            ax_vel.plot([time[0], time[-1]], [DQ_MAX[j], DQ_MAX[j]], 'r--');
            ax_vel.plot([time[0], time[-1]], [-DQ_MAX[j], -DQ_MAX[j]], 'r--');
            ax_vel.set_ylabel(r'$\dot{q}$ [rad/s]');
            ax_vel.set_xlabel('Time [s]');
            plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'vel_j'+str(j),1);
            
            # plot position
            (f, ax_pos) = create_empty_figure(1);
            ax_pos.plot(time, q[j,:], linewidth=LW); # .A.squeeze()
            ax_pos.plot([time[0], time[-1]], [Q_MAX[j], Q_MAX[j]], 'r--');
            ax_pos.plot([time[0], time[-1]], [Q_MIN[j], Q_MIN[j]], 'r--');
            ax_pos.set_ylabel(r'$q$ [rad]');
            ax_pos.set_xlabel('Time [s]');
            plut.saveFigureandParameterinDateFolder(GARBAGE_FOLDER,TEST_NAME+'acc_j'+str(j),1);
            
plt.show()
    
