from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import logodds
from .transforms import rate_matrix

import theano

class ForwardS(ArrayStepShared):
    """
    Use forward sampling (equation 10) to sample a realization of S_t, t=1,...,T_n
    given Q, B, and X constant.
    """
    def __init__(self, vars, X, observed_jumps, model=None):

        model = modelcontext(model)
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)

        super(ForwardS, self).__init__(vars, shared)
        
        self.observed_jumps = observed_jumps
        self.step_sizes = np.sort(np.unique(observed_jumps))

        Q = rate_matrix.backward(self.shared['Q_ratematrix'])
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        #when we add last layer X will be evaluated the same way as Q, B0, B
        self.X = X

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate
        #them
        self.get_params = evaluate_symbolic_shared(Q, B0, B)

    def compute_pS(self,Q,M):
        pS = np.zeros((len(self.step_sizes), M, M))

        lambdas, U = np.linalg.eig(Q)
        for tau in self.step_sizes:
            exp_tD = np.diag(np.exp(lambdas*tau))
            U_inv = np.linalg.inv(U)
            pS_tau = np.dot(np.dot(U, exp_tD), U_inv)

            tau_ind = np.where(self.step_sizes == tau)[0][0]
            pS[tau_ind,:,:] = pS_tau

        return pS

    def computeBeta(self, Q, B0, B):
        M = self.M = Q.shape[0]
        X = self.X
        K = self.K = X.shape[0]
        Tn = self.Tn = X.shape[1]
        pS = self.pS = self.compute_pS(Q,M)
        observed_jumps = self.observed_jumps
        
        Beta = np.zeros((M,Tn))
        Beta[:,Tn-1] = 1
        for t in np.arange(Tn-1, 0, -1):
            tau_ind = np.where(self.step_sizes==observed_jumps[t-1])[0][0]
            for i in range(M):
                #j represents sum
                for j in range(M):
                    prod_psi = 1.0
                    for k in range(K):
                        if i != j:
                            if X[k,t] == 0 and X[k,t-1] == 0:
                                psi = 1-B[k,j]
                            elif X[k,t] == 1 and X[k,t-1] == 0:
                                psi = B[k,j]
                            else:
                                psi = 1.0
                        else:
                            if X[k,t] == 1 and X[k,t-1] == 0:
                                psi = 0.0
                            else:
                                psi = 1.0

                        prod_psi *= psi
                    Beta[i,t-1] += Beta[j,t]*pS[tau_ind,i,j]*prod_psi

        return Beta
        
    def astep(self, S_current):
        
        #paramaters are now usable
        Q,B0,B=self.get_params()
        
        Beta = self.computeBeta(Q, B0, B)

        S_next = S_current
        
        return S_next

def evaluate_symbolic_shared(Q, B0, B):
    f = theano.function([], [Q, B0, B])
    return f