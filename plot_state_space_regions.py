# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:54:01 2015

Test the expressions for the acceleration limits that guarantees the feasibility
of the position and velocity limits in the future.
These expressions have been derived using the viability theory.
@author: adelpret
"""
import numpy as np
from plot_utils import create_empty_figure
import plot_utils as plut
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import brewer2mpl
from acc_bounds_util_2e import isStateViable_2
from acc_bounds_util_2e import computeAccLimits_2
from acc_bounds_util_2e import computeAccLimitsFromViability_2
from acc_bounds_util_2e import computeAccLimitsFromPosLimits_2

plut.SAVE_FIGURES = True; 
EPS = 1e-10;

qMax    = 0.5;
qMin    = -0.5;
MAX_VEL = 2.0;
MAX_ACC = 5.0;
DT = 0.1;

qMax    = 2.0;
qMin    = -2.0;
MAX_VEL = 5.0;
MAX_ACC = 10.0;
E = 0.1*MAX_ACC;  

DT_SAFE = 1.01*DT;
Q_INTERVAL = 0.002; # for plotting the range of possible angles is sampled with this step
DQ_INTERVAL = 0.02; # for plotting the range of possible velocities is sampled with this step

# Starting from qMin and applying maximum acceleration until you reach the middle position (i.e. 0.5*(qMax+qMin))
# you get the maximum velocity you can reach without violating position and acceleration bounds.
# By solving the equation:
#   qMin + 0.5*t**2*MAX_ACC = 0.5*(qMin+qMax)
# we can find t=sqrt((qMax-qMin)/MAX_ACC) and then use it to compute the max velocity:
#   dq(t) = t*MAX_ACC = sqrt(MAX_ACC*(qMax-qMin));
max_vel_from_pos_acc_bounds = np.sqrt((MAX_ACC-E)*(qMax-qMin));
print("INFO Max velocity is %f, max velocity forced by position and acceleration bounds is %f" % (MAX_VEL, max_vel_from_pos_acc_bounds));
  

cmap=brewer2mpl.get_map('OrRd', 'sequential', 4, reverse=False).mpl_colormap
rq = np.arange(qMin, qMax+Q_INTERVAL, Q_INTERVAL);
rdq = np.arange(-max_vel_from_pos_acc_bounds, max_vel_from_pos_acc_bounds+DQ_INTERVAL, DQ_INTERVAL);
Q, DQ = np.meshgrid(rq, rdq);

Z_up  = -np.ones(Q.shape)*(MAX_ACC-E);
Z_low = np.ones(Q.shape)*(MAX_ACC-E);
Z_viab_up  = -np.ones(Q.shape)*(MAX_ACC-E);
Z_viab_low = np.ones(Q.shape)*(MAX_ACC-E);
Z_pos_up  = -np.ones(Q.shape)*(MAX_ACC-E);
Z_pos_low = np.ones(Q.shape)*(MAX_ACC-E);
Z_vel_up  = -np.ones(Q.shape)*(MAX_ACC-E);
Z_vel_low = np.ones(Q.shape)*(MAX_ACC-E);
Z_regions = np.zeros(Q.shape);
for i in range(len(rdq)):
    for j in range(len(rq)):
        (Z_pos_low[i,j], Z_pos_up[i,j]) = computeAccLimitsFromPosLimits_2(Q[i,j], DQ[i,j], qMin, qMax, MAX_ACC, DT,E);
        Z_vel_low[i,j] = (-MAX_VEL-DQ[i,j])/DT;
        Z_vel_up[i,j] = (MAX_VEL-DQ[i,j])/DT;
        if(isStateViable_2(Q[i,j], DQ[i,j], qMin, qMax, MAX_VEL, MAX_ACC,E)==0.0):
            (Z_viab_low[i,j], Z_viab_up[i,j]) = computeAccLimitsFromViability_2(Q[i,j], DQ[i,j], qMin, qMax, MAX_ACC, DT,E);
            (Z_low[i,j], Z_up[i,j]) = computeAccLimits_2(Q[i,j], DQ[i,j], qMin, qMax, MAX_VEL, MAX_ACC, DT,E);
            if(Z_pos_up[i,j] == Z_up[i,j]):
                Z_regions[i,j] = 2.3;
                #print "Position bound is constraining at ", Q[i,j], DQ[i,j];
            elif(Z_vel_up[i,j] == Z_up[i,j]):
                Z_regions[i,j] = 1;
            elif(MAX_ACC - E == Z_up[i,j]):
                Z_regions[i,j] = 3;
            elif(Z_viab_up[i,j] == Z_up[i,j]):
                Z_regions[i,j] = 4;
            else:
                print("Error acc ub unidentified")
            
print("Min value of ddqMaxPos-ddqMaxViab = %f (should be >=0 if pos bound are redundant)" % np.min(Z_pos_up-Z_viab_up));
print("Max value of ddqMinPos-ddqMinViab = %f (should be <=0 if pos bound are redundant)" % np.max(Z_pos_low-Z_viab_low));

f, ax = create_empty_figure();
CF = ax.contourf(Q, DQ, Z_regions, 10, cmap=plt.get_cmap('Paired_r'));
#f.colorbar(CF, ax=ax, shrink=0.9);
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'));
#ax.set_xlabel(r'$q$');
#ax.set_ylabel(r'$\dot{q}$');
ax.xaxis.set_ticks([qMin, qMax]);
ax.yaxis.set_ticks([-MAX_VEL, 0, MAX_VEL]);
ax.set_ylim([-MAX_VEL, MAX_VEL]);
#ax.set_title('Acc regions');
plut.saveFigure('state_space_regions');
    
plt.show();
    