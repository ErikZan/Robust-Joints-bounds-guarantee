import numpy as np
from scipy.stats import mvn
from scipy.stats import norm

class InequalitiesProbability(object):
    """
    Utility class to compute the probability that a system of linear 
    inequalities is satisfied when the variable is subject to a zero mean
    Gaussian noise:
        G*(x+e)+g >= 0
    """
    m_in = 0;       # number of inequalities (i.e. rows of G)
#    mu = [];        # mean value (always 0) of e
    Sigma = [];     # diagonal covariance matrix of random variable e
    Sigma_eG = [];  # covariance matrix of random variable G*e
    
    def __init__(self, std_dev):
#        self.m_in   = nb_ineq;
#        self.mu     = np.zeros(self.m_in);
        self.Sigma  = np.diag(std_dev**2);
        return;
        
    def computeProbability(self, x, G, g):
        # compute probability of violating the constraints
        self.m_in       = G.shape[0];
        self.Sigma_eG   = np.dot(np.dot(G, self.Sigma), G.transpose());
        low             = np.array(-100*np.sqrt(np.diag(self.Sigma_eG)));
        ineq_margins    = np.dot(G,x)+g;
        p,i             = mvn.mvnun(low, ineq_margins, np.zeros(self.m_in), self.Sigma_eG);
        return p;
        
    def computeIndividualProbabilities(self, x, G, g):
        self.m_in       = G.shape[0];
        self.Sigma_eG   = np.dot(np.dot(G, self.Sigma), G.transpose());
        gaussDistr      = norm(np.zeros(self.m_in), np.sqrt(np.diag(self.Sigma_eG)));
        R               = gaussDistr.cdf(np.dot(G,x)+g);
        return R;
        
    # Equivalent to computeIndividualprobabilities but it uses mvn rather than norm
    def computeIndividualProbabilities2(self, x, G, g):
        self.m_in       = G.shape[0];
        R               = np.zeros(self.m_in);
        for i in range(self.m_in):
            Sigma_eG        = np.dot(np.dot(G[i,:], self.Sigma), G[i,:].transpose());
            low             = np.array([-100*np.sqrt(Sigma_eG)]);
            ineq_margins    = np.dot(G[i,:],x)+g[i];
            p,dummy         = mvn.mvnun(low, ineq_margins, 0.0, Sigma_eG);
            R[i]            = p;
#            print "Ineq %d, p(%f<x<%f) = %.2f,\t sigma %f" % (i, low[0], ineq_margins, p, Sigma_eG);
        return R;
