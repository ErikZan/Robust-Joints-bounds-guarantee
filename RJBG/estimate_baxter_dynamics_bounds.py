# -*- coding: utf-8 -*-
"""
Estimate bounds on the dynamics of the Baxter robot through sampling.
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
import matplotlib.pyplot as plt
import os 
import time
import sys

def createListOfLists(size1, size2):
    l = size1*[None,];
    for i in range(size1):
        l[i] = size2*[None,];
    return l;

mat_diag = lambda M: np.asmatrix(np.diag(M)).reshape((M.shape[0],1))

np.random.seed(0)

''' USER PARAMETERS '''
N_SAMPLING = 1000;
DISPLAY_ROBOT_CONFIGURATIONS = False;
DISPLAY_EXTREMUM_POSTURES = False;
CHECK_COLLISIONS = True;
# baxter 1 arm
ACTIVE_COLLISION_PAIRS = [0,  1,  2,  5,  6,  7,  8,  10,  11,  12,  15,  16,  17,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  52,  53,  54,  55,  56,  57,  59,  60,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  77,  78,  80,  81,  82,  84,  85,  86,  87,  88,  91,  92,  94,  96,  97,  98,  100,  101,  102,  103,  104,  105,  106,  108,  109,  110,  111];
# baxter 2 dof
ACTIVE_COLLISION_PAIRS = [0,  1,  5,  6,  10,  15,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  47,  48,  49,  52,  54,  55,  56,  57,  59,  60,  62,  63,  64,  65,  66,  67,  68,  69,  71,  72,  74,  75,  77,  78,  80,  81,  82,  84,  86,  87,  88];
MOTOR_INERTIA = 4.59e-6;
GEAR_RATIO = 66;
DDQ_MAX = np.array([30.0, 60.0]);
DDQ_MAX = np.array([10.0, 30.0, 20.0, 20.0, 30.0, 30.0, 30.0]);
''' END OF USER PARAMETERS '''

print("Gonna sample %d random joint configuration to compute bounds on the Mass matrix and bias forces"%(N_SAMPLING));
np.set_printoptions(precision=2, suppress=True);

# Sample the joint space of the robot
model_path = os.getcwd()+'/../data/baxter'
#robot = RobotWrapper(model_path+'/baxter_description/urdf/baxter_2_dof.urdf', [ model_path, ]);
robot = RobotWrapper(model_path+'/baxter_description/urdf/baxter_1_arm.urdf', [ model_path, ]);
#model_path = os.getcwd()+'/../data'
#robot = RobotWrapper(model_path+'/pr2_description/urdf/pr2.urdf', [ model_path, ]);
robot.addAllCollisionPairs();
NON_ACTIVE_COLLISION_PAIRS = [];
for i in range(len(robot.collision_data.activeCollisionPairs)):
    if(i not in ACTIVE_COLLISION_PAIRS):
        NON_ACTIVE_COLLISION_PAIRS += [i];
robot.deactivateCollisionPairs(NON_ACTIVE_COLLISION_PAIRS);

dcrba = DCRBA(robot);
coriolis = Coriolis(robot);
drnea = DRNEA(robot);


robot.initDisplay(loadModel=True)
#robot.loadDisplayModel("world/pinocchio", "pinocchio", model_path)
robot.viewer.gui.setLightingMode('world/floor', 'OFF');
#print robot.model
Q_MIN   = robot.model.lowerPositionLimit;
Q_MAX   = robot.model.upperPositionLimit;
DQ_MAX  = robot.model.velocityLimit;
TAU_MAX = robot.model.effortLimit;
q = se3.randomConfiguration(robot.model, Q_MIN, Q_MAX);
robot.display(q);           # Display the robot in Gepetto-Viewer.

nq = robot.nq;
nv = robot.nv;
v = mat_zeros(robot.nv);
dv = mat_zeros(robot.nv);
ddq_ub_min = mat_zeros(nq) + 1e10;
ddq_lb_max = mat_zeros(nq) - 1e10;
M_max = mat_zeros((nv,nv)) - 1e10;
M_min = mat_zeros((nv,nv)) + 1e10;
h_max = mat_zeros(nv) - 1e10;
h_min = mat_zeros(nv) + 1e10;
dh_dq_max = mat_zeros((nv,nq)) - 1e10;
dh_dq_min = mat_zeros((nv,nq)) + 1e10;
dh_dv_max = mat_zeros((nv,nv)) - 1e10;
dh_dv_min = mat_zeros((nv,nv)) + 1e10;

q_h_max = mat_zeros((nq,nv));
q_h_min = mat_zeros((nq,nv));
q_M_max = createListOfLists(nv,nv);
q_M_min = createListOfLists(nv,nv);
Md = mat_zeros((nv,nv));
for i in range(nv):
    Md[i,i] = (GEAR_RATIO**2)*MOTOR_INERTIA;

# acceleration upper bounds as a function of the joint position
ddq_ub_of_q = np.zeros((nq, N_SAMPLING, 2));

for i in range(N_SAMPLING):
    q = se3.randomConfiguration(robot.model, Q_MIN, Q_MAX);
    while(CHECK_COLLISIONS and robot.isInCollision(q)):
        print("* Robot is in collision at sample %d" % i);
        q = se3.randomConfiguration(robot.model, Q_MIN, Q_MAX);

    if(DISPLAY_ROBOT_CONFIGURATIONS):
        robot.display(q);
    
    v = np.multiply(DQ_MAX, 2.0*rand(nv)-1.0);
    M = robot.mass(q) + Md;
    h = robot.biais(q, v);
#    dM = dcrba(q);          # d/dq M(q)  so that d/dqi M = Mp[:,:,i] (symmetric), then dtau = tensordot(Mp,dq,[2,0])
    dh_dv = coriolis(q,v);     # d/dvq RNEA(q,vq) = C(q,vq)
    dh_dq = drnea(q,v,dv);     # d/dq RNEA(q,vq,aq)
    ddqUb = np.divide(TAU_MAX - h, mat_diag(M));
    ddqLb = np.divide(-TAU_MAX - h, mat_diag(M));
    for j in range(nq):
        ddq_ub_of_q[j,i,0] = q[j];
        ddq_ub_of_q[j,i,1] = TAU_MAX[j] - h[j];
        for k in range(nv):
            if(k!=j):
                ddq_ub_of_q[j,i,1] -= abs(M[j,k]*DDQ_MAX[k]);
        ddq_ub_of_q[j,i,1]  /= M[j,j];

    for j in np.where(h>h_max)[0]:
        q_h_max[:,j] = q.copy();
    for j in np.where(h<h_min)[0]:
        q_h_min[:,j] = q.copy();
    for (j,k) in zip(np.where(M>M_max)[0], np.where(M>M_max)[1]):
        q_M_max[j][k] = q.copy();
    for (j,k) in zip(np.where(M<M_min)[0], np.where(M<M_min)[1]):
        q_M_min[j][k] = q.copy();
    for j in np.where(ddqUb<0.0)[0]:
        print("\nWARNING ddqMax of joint %d is negative: %f"%(j,ddqUb[j]));
    for j in np.where(ddqLb>0.0)[0]:
        print("\nWARNING ddqMin of joint %d is positive: %f"%(j,ddqLb[j]));

    M_max = np.maximum(M, M_max);
    M_min = np.minimum(M, M_min);
    h_max = np.maximum(h, h_max);
    h_min = np.minimum(h, h_min);
    dh_dq_max = np.maximum(dh_dq, dh_dq_max);
    dh_dq_min = np.minimum(dh_dq, dh_dq_min);
    dh_dv_max = np.maximum(dh_dv, dh_dv_max);
    dh_dv_min = np.minimum(dh_dv, dh_dv_min);    
    ddq_ub_min = np.minimum(ddqUb, ddq_ub_min);
    ddq_lb_max = np.maximum(ddqLb, ddq_lb_max);

            
print("max(dq)\n", DQ_MAX.T);
print("max(tau)\n", TAU_MAX.T);
print("max(M)\n", M_max);
print("min(M)\n", M_min);
print("max(h)\n", h_max.T);
print("min(h)\n", h_min.T);
print("max(dh_dq)\n", (dh_dq_max));
print("min(dh_dq)\n", (dh_dq_min));
print("max(dh_dv)\n", (dh_dv_max));
print("min(dh_dv)\n", (dh_dv_min));
print("min(DDQ_MAX):\n", ddq_ub_min.T);
print("max(DDQ_MIN):\n", ddq_lb_max.T);

for j in range(nq):
    plt.figure();
    plt.plot(ddq_ub_of_q[j,:,0], ddq_ub_of_q[j,:,1], 'x ');
    plt.title('Joint '+str(j));
    plt.xlabel('q');
    plt.ylabel('ddq upper bound');
plt.show();
    

if(DISPLAY_EXTREMUM_POSTURES):
    dt = 0.01;
    T = 1.0;
    N = int(T/dt);
    for j in range(nq):
#        print "This is joint %d" % j;
#        q = Q_MIN + 0.5*(Q_MAX-Q_MIN);
#        q[j] = Q_MIN[j];
#        for t in range(N):
#            robot.display(q);
#            q[j] += (Q_MAX[j]-Q_MIN[j])/N;
#            time.sleep(dt);
        robot.display(q_h_max[:,j]);
        input("Posture of max h for joint %d, that is %.1f. Press Enter to continue" % (j,h_max[j]));
        robot.display(q_h_min[:,j]);
        input("Posture of min h for joint %d, that is %.1f. Press Enter to continue" % (j,h_min[j]));

    for i in range(nq):
        for j in range(i,nq):
            robot.display(q_M_max[i][j]);
            input("Posture of max M[%d,%d], that is %.1f. Press Enter to continue" % (i,j,M_max[i,j]));
            robot.display(q_M_min[i][j]);
            input("Posture of min M[%d,%d], that is %.1f. Press Enter to continue" % (i,j,M_min[i,j]));
