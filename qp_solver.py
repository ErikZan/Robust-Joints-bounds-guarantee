import numpy as np
from qpoases import PyQProblem as SQProblem
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyReturnValue
import time

EPS = 1e-6

class qpSolver (object):
    """
    Solver for the following QP:
      minimize      0.5 x' H x + g' x
      subject to    b_lb <= B x <= b_ub
                    x_lb <= x   <= x_ub
    """
    
    NO_WARM_START = False;
       
    name = "";  # solver name
    n = 0;      # number of variables
    m_in = 0;   # number of equalities/inequalities
    
    x = [];    # last solution
    
    Hess = [];     # Hessian
    grad = [];     # gradient
    B = [];
    b_ub = [];
    b_lb = [];
    x_lb = [];
    x_ub = [];
    
    maxIter=0;              # max number of iterations
    verb=0;                 # verbosity level of the solver (0=min, 2=max)
    iter = 0;               # current iteration number
    computationTime = 0.0;  # total computation time
    qpTime = 0.0;           # time taken to solve the QP(s) only
    
    qpOasesSolver = [];
    options = [];           # qp oases solver's options
    
    def __init__ (self, name, n, m_in, maxIter=1000, verb=1):
        self.name       = name;
        self.iter       = 0;
        self.maxIter    = maxIter;
        self.verb       = verb;
        
        self.m_in       = m_in;
        self.n          = n;
        self.iter       = 0;
        self.qpOasesSolver  = SQProblem(self.n,self.m_in); #, HessianType.SEMIDEF);
        self.options        = Options();
        if(self.verb<=1):
            self.options.printLevel  = PrintLevel.NONE;
        elif(self.verb==2):
            self.options.printLevel  = PrintLevel.LOW;
        elif(self.verb==3):
            self.options.printLevel  = PrintLevel.MEDIUM;
        elif(self.verb>4):
            self.options.printLevel  = PrintLevel.DEBUG_ITER;
            print("set high print level")
        self.options.enableRegularisation = True;
        self.qpOasesSolver.setOptions(self.options);
        self.initialized = False;
        
        self.Hess = np.identity(self.n);
        self.grad = np.zeros(self.n);
        self.x_lb = np.array(self.n*[-1e10,]);
        self.x_ub = np.array(self.n*[1e10,]);
        
        self.B = np.zeros((self.m_in,self.n));
        self.b_ub = np.zeros(self.m_in) + 1e10;
        self.b_lb = np.zeros(self.m_in) - 1e10;
        
        self.x = np.zeros(self.n);

    ''' Solve the specified quadratic program. If some problem data are not supplied, the last specified
        values are used. By default upper bounds are set to 1e10 and lower bounds to -1e10, while B=0.
    '''
    def solve(self, H, g, x_lb=None, x_ub=None, B=None, b_lb=None, b_ub=None, maxIter=None, maxTime=100.0):
        start = time.time();
        
        # copy all data to avoid problem with memory alignement with qpOases
        if(type(H)==np.matrix):
            self.Hess = H.A.squeeze();
        else:
            self.Hess = np.copy(H);

        if(type(g)==np.matrix):
            self.grad = g.A.squeeze();
        else:
            self.grad = np.copy(g);
            
        if(x_lb is not None):
            if(type(x_lb)==np.matrix):
                self.x_lb = x_lb.A.squeeze();
            else:
                self.x_lb = np.copy(x_lb);
                
        if(x_ub is not None):
            if(type(x_ub)==np.matrix):
                self.x_ub = x_ub.A.squeeze();
            else:
                self.x_ub = np.copy(x_ub);
                
        if(b_lb is not None):
            if(type(b_lb)==np.matrix):
                self.b_lb = b_lb.A.squeeze();
            else:
                self.b_lb = np.copy(b_lb);
                
        if(b_ub is not None):
            if(type(b_ub)==np.matrix):
                self.b_ub = b_ub.A.squeeze();
            else:
                self.b_ub = np.copy(b_ub);
                
        if(B is not None):
            if(type(B)==np.matrix):
                self.B = B.A.squeeze();
            else:
                self.B = np.copy(B);
        
        if(maxIter==None):
            maxIter = self.maxIter;
        maxActiveSetIter    = np.array([maxIter]);
        maxComputationTime  = np.array(maxTime);        
        self.imode = self.qpOasesSolver.init(self.Hess, self.grad, self.B, self.x_lb, self.x_ub, self.b_lb, 
                                             self.b_ub, maxActiveSetIter, maxComputationTime);
        self.qpTime = maxComputationTime;
        self.iter   = 1+maxActiveSetIter[0];
        self.qpOasesSolver.getPrimalSolution(self.x);
        self.print_qp_oases_error_message(self.imode, self.name);
        
        Bx = np.dot(self.B, self.x);
        for i in range(self.m_in):
            if(Bx[i] > b_ub[i] + EPS):
                print("Constraint %d upper bound violated, B*x = %f, b_ub=%f" %(i,Bx[i],b_ub[i]));
            if(Bx[i] < b_lb[i] - EPS):
                print("Constraint %d lower bound violated, B*x = %f, b_lb=%f" %(i,Bx[i],b_lb[i]));
        
        # termination conditions
        if(self.qpTime>=maxTime):
            if(self.verb>0):
                print("[%s] Max time reached %f after %d iters" % (self.name, self.qpTime, self.iter));
            self.imode = 9;
            
        self.computationTime = time.time()-start;
        
        return self.x;
        
    def print_qp_oases_error_message(self, imode, solver_name):
        if(imode!=0 and self.verb>=0):
            if(imode==PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY):
                print("[%s] ERROR Qp oases HOTSTART_STOPPED_INFEASIBILITY" % solver_name); # 60
            elif(imode==PyReturnValue.MAX_NWSR_REACHED):
                print("[%s] ERROR Qp oases RET_MAX_NWSR_REACHED" % solver_name); # 63
            elif(imode==PyReturnValue.STEPDIRECTION_FAILED_CHOLESKY):
                print("[%s] ERROR Qp oases STEPDIRECTION_FAILED_CHOLESKY" % solver_name); # 68
            elif(imode==PyReturnValue.HOTSTART_FAILED_AS_QP_NOT_INITIALISED):
                print("[%s] ERROR Qp oases HOTSTART_FAILED_AS_QP_NOT_INITIALISED" % solver_name); # 53
            elif(imode==PyReturnValue.INIT_FAILED_INFEASIBILITY):
                print("[%s] ERROR Qp oases INIT_FAILED_INFEASIBILITY" % solver_name); # 37
#                    RET_INIT_FAILED_HOTSTART = 36
            elif(imode==PyReturnValue.UNKNOWN_BUG):
                print("[%s] ERROR Qp oases UNKNOWN_BUG" % solver_name); # 9
            else:
                print("[%s] ERROR Qp oases %d " % (solver_name, imode));
    