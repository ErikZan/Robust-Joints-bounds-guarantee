# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:58:48 2016

Estimate bounds on the dynamics of the Baxter robot through sampling.

@author: adelpret
"""

import pinocchio as se3
from baxter_wrapper import BaxterWrapper
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import zero as mat_zeros
from time import sleep
from pinocchio.utils import rand, mprint
from pinocchio.dcrba import DCRBA, Coriolis, DRNEA
import numpy as np
from numpy.linalg import norm
import os 
import time
import sys

# RESULT:
# [ 0,  1,  2,  5,  6,  7,  8,  10,  11,  12,  15,  16,  17,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  52,  53,  54,  55,  56,  57,  59,  60,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  77,  78,  80,  81,  82,  84,  85,  86,  87,  88,  91,  92,  94,  96,  97,  98,  100,  101,  102,  103,  104,  105,  106,  108,  109,  110,  111,  ]

mat_diag = lambda M: np.asmatrix(np.diag(M)).reshape((M.shape[0],1))

np.random.seed(0)

''' USER PARAMETERS '''
N_SAMPLING = 1000;
ACTIVE_COLLISION_PAIRS = [0,  1,  2,  5,  6,  7,  8,  10,  11,  12,  15,  16,  17,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  52,  53,  54,  55,  56,  57,  59,  60,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  77,  78,  80,  81,  82,  84,  85,  86,  87,  88,  91,  92,  94,  96,  97,  98,  100,  101,  102,  103,  104,  105,  106,  108,  109,  110,  111];
''' END OF USER PARAMETERS '''

print("Gonna sample %d random joint configuration to figure out with the help of the user which collision pairs are valid"%(N_SAMPLING));
np.set_printoptions(precision=2, suppress=True);

# Sample the joint space of the robot
model_path = os.getcwd()+'/../data/baxter'
robot = RobotWrapper(model_path+'/baxter_description/urdf/baxter_2_dof.urdf', [ model_path, ]);
robot.addAllCollisionPairs();
NON_ACTIVE_COLLISION_PAIRS = [];
for i in range(len(robot.collision_data.activeCollisionPairs)):
    if(i not in ACTIVE_COLLISION_PAIRS):
        NON_ACTIVE_COLLISION_PAIRS += [i];
robot.deactivateCollisionPairs(NON_ACTIVE_COLLISION_PAIRS);

robot.initDisplay(loadModel=True)
robot.viewer.gui.setLightingMode('world/floor', 'OFF');
#print robot.model
Q_MIN   = robot.model.lowerPositionLimit;
Q_MAX   = robot.model.upperPositionLimit;
q = se3.randomConfiguration(robot.model, Q_MIN, Q_MAX);
robot.display(q);           # Display the robot in Gepetto-Viewer.
nq = robot.nq;

rightAnswerCounter = 0;
for i in range(N_SAMPLING):
    if(rightAnswerCounter > 10):
        ans = input("The model has been right for %d times in a row. Do you wanna stop?"%rightAnswerCounter);
        if 'y' in ans:
            print("Stopping script");
            print("Active collision pairs are: [", end=' ')
            for (cp, active) in enumerate(robot.collision_data.activeCollisionPairs):
                if(active):
                    print("%d, "%cp, end=' ')
            print("]");
            break;
            
    q = se3.randomConfiguration(robot.model, Q_MIN, Q_MAX);
    if(robot.isInCollision(q)):
        print("\n *** Robot is in collision at sample %d ***" % i);
        robot.display(q);
        collPairList = robot.findAllCollisionPairs();
        ans = input("Do you think the robot is in collision [y/n]? ");
        if('n' in ans):
            rightAnswerCounter = 0;
            print("Deactivating all these collisions")                
            for cp in collPairList:
                print("    collision %d:"%cp[0], cp[1], end=' ')
                robot.collision_data.deactivateCollisionPair(cp[0]);
        else:
            rightAnswerCounter += 1;
    else:
        print("\n *** Robot NOT IN COLLISION at sample %d ***" % i);
        robot.display(q);
        collPairList = robot.findAllCollisionPairs(False);            
        ans = input("Do you think the robot is in collision [y/n]? ");
        if('y' in ans):
            rightAnswerCounter = 0;
            print("Re-activating all these collisions:")
            for cp in collPairList:
                print("    collision %d:"%cp[0], cp[1], end=' ')
                robot.collision_data.activateCollisionPair(cp[0], True);
        else:
            rightAnswerCounter += 1;

