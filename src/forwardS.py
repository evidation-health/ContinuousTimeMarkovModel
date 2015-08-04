from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import stick_breaking, logodds
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

        pi = stick_breaking.backward(self.shared['pi_stickbreaking'])
        Q = rate_matrix.backward(self.shared['Q_ratematrix'])
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        #when we add last layer X will be evaluated the same way as Q, B0, B
        self.X = X

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate
        #them
        self.get_params = evaluate_symbolic_shared(pi, Q, B0, B)

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
        M = self.M
        X = self.X
        K = self.K
        Tn = self.Tn = X.shape[1]
        pS = self.pS = self.compute_pS(Q,M)
        observed_jumps = self.observed_jumps
        
        beta = np.zeros((M,Tn))
        beta[:,Tn-1] = 1
        #print "t\ti\tj\tbeta\tpS_ij\tprod"
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
                    #print t-1,"\t",i,"\t",j,"\t",beta[j,t],"\t",pS[tau_ind,i,j],"\t",prod_psi
                    beta[i,t-1] += beta[j,t]*pS[tau_ind,i,j]*prod_psi
                #print "###beta:", beta[i,t-1]

        return beta
        
    def astep(self, q0):
        
        S = q0

        #paramaters are now usable
        pi,Q,B0,B=self.get_params()

        #calculate pS0(i) | X, pi, B0
        M = self.M = Q.shape[0]
        X = self.X
        K = self.K = X.shape[0]
        pS0 = np.zeros(M)
        for i in range(M):
            prod_psi = 1.0
            for k in range(K):
                if X[k,0] == 1:
                    prod_psi *= B0[k,i]
                else:
                    prod_psi *= 1-B0[k,i]
            pS0[i] = pi[i] * prod_psi

        #draw S0
        cdf = np.cumsum(pS0)
        r = np.random.uniform() * cdf[-1]
        drawn_state = np.searchsorted(cdf, r)
        S[0] = drawn_state

        #calculate pS_ij | X, Q, B
        #note: pS is probability of jump conditional on Q
        #whereas pS_ij is also conditional on everything else in the model
        #and is what we're looking for
        beta = self.computeBeta(Q, B0, B)
        Tn = self.Tn
        observed_jumps = self.observed_jumps
        pS = self.pS
        pS_ij = np.zeros(M)
        for t in range(0,Tn-1):
            i = S[t]
            tau = observed_jumps[t]
            tau_ind = np.where(self.step_sizes == tau)[0][0]

            for j in range(M):
                if S[t] > j:
                    pS_ij[j] = 0.0
                else:
                    prod_psi = 1.0
                    for k in range(K):
                        if i != j:
                            if X[k,t+1] == 0 and X[k,t] == 0:
                                psi = 1-B[k,j]
                            elif X[k,t+1] == 1 and X[k,t] == 0:
                                psi = B[k,j]
                            else:
                                psi = 1.0
                        else:
                            if X[k,t+1] == 1 and X[k,t] == 0:
                                psi = 0.0
                            else:
                                psi = 1.0
                        prod_psi *= psi
                    #if prod_psi == 0.0:
                    #    print "#########","t:", t, "i:", i, "j:", j
                    pS_ij[j] = beta[j,t+1]/beta[i,t] * pS[tau_ind,i,j] \
                                * prod_psi

            if t==0:
                print "\n\n~~~~","pS_ij:",pS_ij
            if t==2:
                print "~~~~","pS_ij:",pS_ij

            #sample S[t+1] from pS_ij
            cdf = np.cumsum(pS_ij)
            r = np.random.uniform() * cdf[-1]
            drawn_state = np.searchsorted(cdf, r)
            S[t+1] = drawn_state

        return S

def evaluate_symbolic_shared(pi, Q, B0, B):
    f = theano.function([], [pi, Q, B0, B])
    return f