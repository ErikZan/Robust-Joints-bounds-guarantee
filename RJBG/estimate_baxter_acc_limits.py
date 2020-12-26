# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:58:48 2016

Estimate the acceleration limits of the Baxter robot through sampling.

@author: adelpret
"""

import pinocchio as se3
from baxter_wrapper import BaxterWrapper
from pinocchio.utils import *
import numpy.matlib
from numpy.random import uniform
import plot_utils as plut
import matplotlib.pyplot as plt
from baxter_wrapper import Q_MIN, Q_MAX, DQ_MAX, TAU_MAX
from time import sleep

''' USER PARAMETERS '''
N_SAMPLING = 10**6;
DISPLAY_ROBOT_CONFIGURATIONS = False;
PAUSE_WHEN_WEIRD_STUFF_HAPPENS = False;
''' END OF USER PARAMETERS '''

print("Gonna sample %d random joint configuration to compute a conservative estimation of the joint acceleraiton limits"%(N_SAMPLING));
np.set_printoptions(precision=2, suppress=True);

# Sample the joint space of the robot
robot = BaxterWrapper();
robot.initDisplay(loadModel=True)
#robot.loadDisplayModel("world/pinocchio", "pinocchio", MODELPATH)
robot.viewer.gui.setLightingMode('world/floor', 'OFF');
#print robot.model
q = robot.Q_INIT;
robot.display(q);           # Display the robot in Gepetto-Viewer.

NQ = q.shape[0];
dq0 = np.matlib.zeros(NQ);
ddqMax = np.zeros(NQ);
ddqMin = np.zeros(NQ);
ddqMaxFinal = np.zeros(NQ) + 1e10;
ddqMinFinal = np.zeros(NQ) - 1e10;
for i in range(N_SAMPLING):
    if(i*10%N_SAMPLING==0):
        print("%.0f %%"%(i*100/N_SAMPLING));
    q = uniform(Q_MIN, Q_MAX, NQ);
    if(DISPLAY_ROBOT_CONFIGURATIONS):
        robot.display(q);
    M = robot.mass(q);
    h = robot.bias(q, dq0);
    for j in range(NQ):
        ddqMax[j] = ( TAU_MAX[j] - h[j]) / M[j,j];
        if(ddqMax[j]>0.0):
            if(ddqMax[j] < ddqMaxFinal[j]):
                ddqMaxFinal[j] = ddqMax[j];
        else:
            print("\nWARNING ddqMax of joint %d is negative: %f"%(j,ddqMax[j]));
            print("This is the random configuration where this happened", q);
            if(PAUSE_WHEN_WEIRD_STUFF_HAPPENS):
                sleep(3);

        ddqMin[j] = (-TAU_MAX[j] - h[j]) / M[j,j];            
        if(ddqMin[j]<0.0):
            if(ddqMin[j] > ddqMinFinal[j]):
                ddqMinFinal[j] = ddqMin[j];
        else:
            print("\nWARNING ddqMin is positive %f, joint %d, h %f, tauMax %f"%(ddqMin[j], j, h[j], TAU_MAX[j]));
            print("This is the random configuration where this happened", q);
            qErr = np.copy(q);
            if(PAUSE_WHEN_WEIRD_STUFF_HAPPENS):
                sleep(3);
            
print("DDQ_MAX:\n", ddqMaxFinal);
print("DDQ_MIN:\n", ddqMinFinal);
