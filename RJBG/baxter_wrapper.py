#from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as se3
import numpy as np
import numpy.matlib
from time import sleep
from time import time
import os

# Change following your setup.
MODELPATH = ['/home/erik/baxter_wrapper/baxter_common-master/',]

NQ_OFFSET = 1
NV_OFFSET = 1
"""
Q_MIN = -2.0*np.ones(14);
Q_MAX = 2.0*np.ones(14);
DQ_MAX = -2.0*np.ones(14);
"""

'''Q_MIN   = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059,
                    -1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]).T;
Q_MAX   = np.array([ 1.7016,  1.047,  3.0541,  2.618, 3.059,  2.094,   3.059,
                     1.7016,  1.047,  3.0541,  2.618, 3.059,  2.094,   3.059]).T;
DQ_MAX  = np.array([ 2.0,     2.0,    2.0,     2.0,   4.0,    4.0,     4.0,
                     2.0,     2.0,    2.0,     2.0,   4.0,    4.0,     4.0]).T;'''
Q_MIN   = np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059,
                    -1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]).T;
Q_MAX   = np.array([ 1.7016,  1.047,  3.0541,  2.618, 3.059,  2.094,   3.059,
                     1.7016,  1.047,  3.0541,  2.618, 3.059,  2.094,   3.059]).T;
DQ_MAX  = np.array([ 2.0,     2.0,    2.0,     2.0,   4.0,    4.0,     4.0,
                     2.0,     2.0,    2.0,     2.0,   4.0,    4.0,     4.0]).T ; # *10.0
                     
TAU_MAX = np.array([50.0,    60.0,   50.0,    50.0,  15.0,   15.0,    15.0,
                    50.0,    60.0,   50.0,    50.0,  15.0,   15.0,    15.0]).T;
# I increased TAU_MAX for joint 1 from 50 Nm to 60 Nm because sometimes just to 
# compensate gravity at joint 1 you need more than 50 Nm

class BaxterWrapper(RobotWrapper):    
    PLAYER_FRAME_RATE = 20;
    Q_INIT = [];
    q_def = [];
    v_def = [];

    def __init__(self, filename = MODELPATH[0]+'baxter_description/urdf/baxter.urdf'):
        RobotWrapper.initFromURDF(self, filename, MODELPATH);
        
        self.Q_INIT = np.zeros(self.nq-NQ_OFFSET);
        self.q_def = np.zeros(self.nq);
        self.v_def = np.zeros(self.nv);
        
#        self.q0 = np.matrix( [
#        0.0, 0.0, 0.648702, 0.0, 0.0 , 0.0, 1.0,                             # Free flyer 0-6
#        0.0, 0.0, 0.0, 0.0,                                                  # CHEST HEAD 7-10
#        0.261799388,  0.174532925, 0.0, -0.523598776, 0.0, 0.0, 0.174532925, # LARM       11-17
#        0.261799388, -0.174532925, 0.0, -0.523598776, 0.0, 0.0, 0.174532925, # RARM       18-24
#        0.0, 0.0, -0.453785606, 0.872664626, -0.41887902, 0.0,               # LLEG       25-30
#        0.0, 0.0, -0.453785606, 0.872664626, -0.41887902, 0.0,               # RLEG       31-36
#        ] ).T

        self.ff = list(range(7))
        self.head = list(range(7,7))
        self.l_arm = list(range(8,14))
        self.r_arm = list(range(15,21))

        self.opCorrespondances = { "lh": "left_w2",
                                   "rh": "right_w2",
                                   }

        for op,name in list(self.opCorrespondances.items()):
            idx = self.__dict__[op] = self.index(name)
            #self.__dict__['_M'+op] = types.MethodType(lambda s,q: s.position(q,idx),self)

#    @property
#    def nq(self):
#        return RobotWrapper.nq-NQ_OFFSET
#
#    @property
#    def nv(self):
#        return RobotWrapper.nv-NV_OFFSET

    def mass(self,q):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        return RobotWrapper.mass(self, self.q_def)[NV_OFFSET:, NV_OFFSET:];

    def bias(self,q,v):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        self.v_def[NV_OFFSET:] = v.reshape(self.nv-NV_OFFSET);
        return RobotWrapper.nle(self, self.q_def, self.v_def)[NV_OFFSET:];
#    def gravity(self,q):
#        return se3.rnea(self.model,self.data,q,self.v0,self.v0)

    def position(self,q,index):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        return RobotWrapper.placement(self, self.q_def, index);
        
    def velocity(self,q,v,index):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        self.v_def[NV_OFFSET:] = v.reshape(self.nv-NV_OFFSET);
        return RobotWrapper.velocity(self, self.q_def, self.v_def, index);
        
    def jacobian(self,q,index):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        RobotWrapper.computeJointJacobian(self,self.q_def, index);
        return RobotWrapper.getJointJacobian(self, index)[:,NV_OFFSET:];
    
    def jacobian_2(self,q,index):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        A = RobotWrapper.computeJointJacobian(self,self.q_def, index);
        return A[:,NV_OFFSET:];
    
        
    def dJdq(self, q, v, index):
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        self.v_def[NV_OFFSET:] = v.reshape(self.nv-NV_OFFSET);
        se3.forwardKinematics(self.model, self.data, self.q_def, self.v_def, self.v0);
        dJdq = self.data.a[index]; 
        dJdq.linear -= np.cross(dJdq.linear.squeeze(), dJdq.angular.squeeze());
        return dJdq;


    # Display in gepetto-view the robot at configuration q, by placing all the bodies.
    def display(self,q, refresh=True): 
        self.q_def[NQ_OFFSET:] = q.reshape(self.nq-NQ_OFFSET);
        return RobotWrapper.display(self, self.q_def); #, refresh);
        
    def play(self, q, dt, slow_down_factor=1, print_time_every=-1.0, robotName='hrp2'):
        trajRate = 1.0/dt
        rate = int(slow_down_factor*trajRate/self.PLAYER_FRAME_RATE);
        lastRefreshTime = time();
        timePeriod = 1.0/self.PLAYER_FRAME_RATE;
        for t in range(0,q.shape[1],rate):
            self.display(q[:,t], refresh=False);
            timeLeft = timePeriod - (time()-lastRefreshTime);
            if(timeLeft>0.0):
                sleep(timeLeft);
            self.viewer.gui.refresh();
            lastRefreshTime = time();
            if(print_time_every>0.0 and t*dt%print_time_every==0.0):
                print("%.1f"%(t*dt));
    '''            
    def startCapture(self, filename, extension='jpeg', path='/home/erik/Desktop/Thesis/figures/baxter/'):
        if(not os.path.exists(path)):
            os.makedirs(path);
        self.viewer.gui.startCapture(self.windowID, path+filename, extension);'''

    def startCapture(self, filename, extension='jpeg', path='/home/erik/Desktop/Thesis/figures/baxter/'):
        if(not os.path.exists(path)):
            os.makedirs(path);
        self.viewer.gui.startCapture('python-pinocchio', path+filename, extension);
    def stopCapture(self):
        self.viewer.gui.stopCapture('python-pinocchio');

__all__ = [ 'BaxterWrapper' ]
