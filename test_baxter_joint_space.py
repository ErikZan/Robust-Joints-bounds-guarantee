# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:52:00 2016

@author: adelpret
"""

import pinocchio as se3
from pinocchio.utils import *

import numpy.matlib
from numpy.linalg import pinv
from math import sqrt
from time import sleep

import plot_utils as plut
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

from qp_solver import qpSolver
from acc_bounds_util import computeMultiAccLimits
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
    plut.saveFigure(TEST_NAME+'_'+name);
    return ax;

# Convertion from translation+quaternion to SE3 and reciprocally
q2m = lambda q: se3.SE3( se3.Quaternion(q[6,0],q[3,0],q[4,0],q[5,0]).matrix(), q[:3])
m2q = lambda M: np.vstack([ M.translation,se3.Quaternion(M.rotation).coeffs() ])

''' PLOT-RELATED USER PARAMETERS '''
LW = 4;     # line width
LINE_ALPHA = 1.0;
LEGEND_ALPHA = 1.0;
line_styles     =["c-", "b--", "g-.", "k:", "m-"];
PLOT_JOINT_POS_VEL_ACC = False;
PLOT_STATE_SPACE = False;
Q_INTERVAL = 0.001; # the range of possible angles is sampled with this step for plotting
PLAY_TRAJECTORY_ONLINE = False;
PLAY_TRAJECTORY_AT_THE_END = False;
CAPTURE_IMAGES = False;
plut.SAVE_FIGURES = True;
plut.FIGURE_PATH = '/home/adelpret/repos/20160606_tro_acc_limits/paper_tro/figures/joint_space/';
IMAGES_FILE_NAME = 'baxter_viab_dt_2x';
''' END OF PLOT-RELATED USER PARAMETERS '''

''' CONTROLLER USER PARAMETERS '''
ACC_BOUNDS_TYPE = 'NAIVE'; #'VIAB', 'NAIVE'
T = 4.0;    # total simulation time
DT = 0.01;  # time step
#DT_SAFE = np.array([2, 5, 20])*DT;
DT_SAFE = np.array([1])*DT;
kp = 1000;
kd = 2*sqrt(kp);
DDQ_MAX = np.array([ 12.0, 2.0, 30.0, 30.0, 30.0, 30.0, 30.0,     
                     12.0 ,2.0, 30.0, 30.0, 30.0, 30.0, 30.0]);
q0 = np.matrix([[ 0. , -0.1,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 0. ,  0. ,  0. ]]).T
Q_DES = np.array(0.5*(Q_MIN+Q_MAX)).T;
Q_DES[0] = Q_MAX[0] + 0.5;
Q_DES[8:] = 0.0;
''' END OF CONTROLLER USER PARAMETERS '''

if(len(DT_SAFE)==1):
    line_styles[0] = 'b-';

TEST_NAME = ACC_BOUNDS_TYPE+'_dt_'+str((DT_SAFE/DT).astype(int))[2:-1].replace(' ', '_');

robot = BaxterWrapper();
robot.initDisplay(loadModel=False)
robot.loadDisplayModel("world/pinocchio", "pinocchio", MODELPATH)

robot.viewer.gui.setLightingMode('world/floor', 'OFF');
#print robot.model

robot.display(q0);           # Display the robot in Gepetto-Viewer.

NQ = q0.shape[0];

# spatial velocity to go from M to Mdes, expressed in frame M, is log(M.inverse()*Mdes)
NT = int(T/DT);
NDT = len(DT_SAFE);
q = np.zeros((NQ,NT,NDT));
dq = np.zeros((NQ,NT,NDT));
ddq = np.zeros((NQ,NT-1,NDT));
ddq_des = np.zeros((NQ,NT-1,NDT));
ddq_lb = np.zeros((NQ,NT-1,NDT));
ddq_ub = np.zeros((NQ,NT-1,NDT));

for nt in range(NDT):
    ''' initialize '''
    pos_bound_viol = NQ*[False,]    
    q[:,0,nt] = q0.A.squeeze();
    for t in range(NT-1):
        ddq_des[:,t,nt] = kp*(Q_DES - q[:,t,nt]) - kd*dq[:,t,nt];
        
        if(ACC_BOUNDS_TYPE=='VIAB'):
            (ddq_lb[:,t,nt], ddq_ub[:,t,nt]) = computeMultiAccLimits(q[:,t,nt], dq[:,t,nt], Q_MIN, Q_MAX, DQ_MAX, DDQ_MAX, DT_SAFE[nt]);
        elif(ACC_BOUNDS_TYPE=='NAIVE'):
            for j in range(NQ):
                ddq_ub[j,t,nt] = min( DDQ_MAX[j], ( DQ_MAX[j]-dq[j,t,nt])/DT_SAFE[nt], 2.0*(Q_MAX[j]-q[j,t,nt]-DT_SAFE[nt]*dq[j,t,nt])/(DT_SAFE[nt]**2));
                ddq_lb[j,t,nt] = max(-DDQ_MAX[j], (-DQ_MAX[j]-dq[j,t,nt])/DT_SAFE[nt], 2.0*(Q_MIN[j]-q[j,t,nt]-DT_SAFE[nt]*dq[j,t,nt])/(DT_SAFE[nt]**2));
                if(ddq_lb[j,t,nt] > DDQ_MAX[j]):
                    ddq_lb[j,t,nt] = DDQ_MAX[j];
                if(ddq_ub[j,t,nt] < -DDQ_MAX[j]):
                    ddq_ub[j,t,nt] = -DDQ_MAX[j];
        
        for j in range(NQ):
            if(ddq_des[j,t,nt] > ddq_ub[j,t,nt]):
                ddq[j,t,nt] = ddq_ub[j,t,nt];
            elif(ddq_des[j,t,nt] < ddq_lb[j,t,nt]):
                ddq[j,t,nt] = ddq_lb[j,t,nt];
            else:
                ddq[j,t,nt] = ddq_des[j,t,nt];
    
        ''' check position bounds '''  
        for j in range(NQ):
            if(pos_bound_viol[j]==False):
                if(q[j,t,nt] > Q_MAX[j]):
                    print("With dt %.3f joint %d hit upper bound with vel %f" % (DT_SAFE[nt], j, dq[j,t,nt]));
                    pos_bound_viol[j]=True;
                elif(q[j,t,nt] < Q_MIN[j]):
                    print("With dt %.3f joint %d hit lower bound with vel %f" % (DT_SAFE[nt], j, dq[j,t,nt]));
                    pos_bound_viol[j]=True;
            elif(q[j,t,nt] <= Q_MAX[j] and q[j,t,nt]>=Q_MIN[j]):
                pos_bound_viol[j]=False;
        
        ''' Numerical Integration '''
        q[:,t+1,nt] = q[:,t,nt] + DT*dq[:,t,nt] + 0.5*DT*DT*ddq[:,t,nt];
        dq[:,t+1,nt] = dq[:,t,nt] + DT*ddq[:,t,nt];
        if(PLAY_TRAJECTORY_ONLINE):
            robot.display(q[:,t+1,nt]);
    #       sleep(DT);

if(PLAY_TRAJECTORY_AT_THE_END):
    if(CAPTURE_IMAGES):
        robot.startCapture(IMAGES_FILE_NAME);
    robot.play(q[:,:,nt], DT);
    if(CAPTURE_IMAGES):
        robot.stopCapture();

time = np.arange(0, NT*DT, DT);

if(PLOT_JOINT_POS_VEL_ACC):
    plot_bounded_joint_quantity(time,        q,       Q_MIN,   Q_MAX, 'Joint positions', 'Time [s]', r'$q$ [rad]');
    plot_bounded_joint_quantity(time,       dq,   -1*DQ_MAX,  DQ_MAX, 'Joint velocities', 'Time [s]', r'$q$ [rad/s]');
    plot_bounded_joint_quantity(time[:-1], ddq,  -1*DDQ_MAX, DDQ_MAX, 'Joint accelerations', 'Time [s]', r'$\ddot{q}$ [rad/s${}^2$]');

for j in range(7):
    qMax = Q_MAX[j];
    qMin = Q_MIN[j];
    qMid = 0.5*(qMin+qMax);
    qHalf = 0.5*(qMax-qMin);
    max_vel_from_pos_acc_bounds = np.sqrt(DDQ_MAX[j]*(qMax-qMin));
    
    if( qMax-np.max(q[j,:]) < 0.1*qHalf or 
        np.min(q[j,:])-qMin < 0.1*qHalf or 
        np.max(np.abs(dq[j,:])) > DQ_MAX[j]):
                
        if(np.max(q[j,:]) > qMax):
            print("Joint %d hit up bound" % (j));
        elif(qMax-np.max(q[j,:]) < 0.1*qHalf):
            print("Joint %d got close to up bound, qMax=%f, max(q)=%f" % (j, qMax, np.max(q[j,:])));
            
        if(np.min(q[j,:]) < qMin):
            print("Joint %d hit low bound" % (j));
        elif(np.min(q[j,:])-qMin < 0.1*qHalf):
            print("Joint %d got close to low bound, min(q)=%f, qMin=%f" % (j, np.min(q[j,:]), qMin));
            
        if(np.max(np.abs(dq[j,:])) > 0.99*DQ_MAX[j]):
            print("Joint %d got close to max vel, max(dq)=%f, max_vel=%f" % (j, np.max(np.abs(dq[j,:])), DQ_MAX[j]));
                
        f, ax = plt.subplots(3, 1, sharex=True);
        
        # plot position
        ax_pos = ax[0];
        ax_vel = ax[1];
        ax_acc = ax[2];
#        plut.movePlotSpines(ax[0], [qMin, 0]);
        ax_pos.plot([time[0], time[-1]], [Q_MAX[j], Q_MAX[j]], 'r--');
        ax_pos.plot([time[0], time[-1]], [Q_MIN[j], Q_MIN[j]], 'r--');
        ax_pos.set_ylabel(r'$q$ [rad]');
        ax_pos.yaxis.set_ticks([Q_MIN[j], Q_MAX[j]]);
        ax_pos.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
        ax_pos.set_ylim([np.min(q[j,:,:]), 1.1*np.max(q[j,:,:])]);
        
        # plot velocity
#        plut.movePlotSpines(ax_vel, [0, 0]);
        ax_vel.plot([time[0], time[-1]], [DQ_MAX[j], DQ_MAX[j]], 'r--');
        ax_vel.plot([time[0], time[-1]], [-DQ_MAX[j], -DQ_MAX[j]], 'r--');
        ax_vel.set_ylabel(r'$\dot{q}$ [rad/s]');
        ax_vel.yaxis.set_ticks([DQ_MAX[j], DQ_MAX[j]]);
        ax_vel.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
        ax_vel.set_ylim([-1.1*DQ_MAX[j], 1.1*DQ_MAX[j]]);

        # plot acceleration
#        plut.movePlotSpines(ax_acc, [0, 0]);        
        ax_acc.plot([time[0], time[-1]], [ DDQ_MAX[j],  DDQ_MAX[j]], 'r--');
        ax_acc.plot([time[0], time[-1]], [-DDQ_MAX[j], -DDQ_MAX[j]], 'r--');
        ax_acc.set_ylabel(r'$\ddot{q}$ [rad/s${}^2$]');
        ax_acc.yaxis.set_ticks([-DDQ_MAX[j], DDQ_MAX[j]]);
        ax_acc.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
        ax_acc.set_ylim([-1.1*DDQ_MAX[j], 1.1*DDQ_MAX[j]]);
        
        ax[2].set_xlabel('Time [s]');
        for nt in range(NDT):
            ax_pos.plot(time, q[j,:,nt].squeeze(), line_styles[nt], linewidth=LW, alpha=LINE_ALPHA**nt, label=r'$\delta t=$'+str(int(DT_SAFE[nt]/DT))+'x');
            ax_vel.plot(time, dq[j,:,nt].squeeze(), line_styles[nt], linewidth=LW, alpha=LINE_ALPHA**nt);
            ax_acc.plot(time[:-1], ddq[j,:,nt].squeeze(), line_styles[nt], linewidth=LW, alpha=LINE_ALPHA**nt);
#            ax_acc.plot(time[:-1], ddq_des[j,:].A.squeeze(), 'b:', linewidth=LW);
#            if(NDT==1):
#                ax_acc.plot(time[:-1], ddq_lb[j,:,nt], 'r--');
#                ax_acc.plot(time[:-1], ddq_ub[j,:,nt], 'r--');
        
        if(NDT>1):
            leg = ax_pos.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=NDT, mode="expand", borderaxespad=0.)
#            leg = ax_pos.legend(loc='best');
            leg.get_frame().set_alpha(LEGEND_ALPHA)
        
        if(NDT==1 and PLOT_STATE_SPACE):
            f, ax = plt.subplots(1, 1, sharex=True);
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
            ax.plot(q[j,:].A.squeeze(), dq[j,:].A.squeeze(), 'b-', linewidth=LW);
            ax.xaxis.set_ticks([qMin, q[j,0], qMax]);
            ax.yaxis.set_ticks([-DQ_MAX[j], 0, DQ_MAX[j]]);
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'));
            ax.set_xlim([np.min(q[j,:]) -0.05*qHalf, np.max(q[j,:])+0.05*qHalf]);
            ax.set_ylim([np.min(dq[j,:])-0.05*DQ_MAX[j], np.max(dq[j,:])+0.05*DQ_MAX[j]]);
    #        ax.set_title('Joint '+str(j));
            ax.set_xlabel(r'$q$ [rad]');
            ax.set_ylabel(r'$\dot{q}$ [rad/s]');
        
        plut.saveFigure(TEST_NAME+'_j'+str(j));
        

plt.show()
    