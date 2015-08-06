from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import stick_breaking, logodds
from .transforms import rate_matrix

import theano

import time

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
        self.M = Q.shape[0]
        K = self.K = X.shape[0]

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate them
        self.get_params = evaluate_symbolic_shared(pi, Q, B0, B)
        self.pi, self.Q, self.B0, self.B=self.get_params()
        self.M = self.Q.shape[0]
        self.Tn = self.X.shape[1]

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

    #calculate p( X_t | S_t=j, S_{t-1}=i )
    def compute_pXt_GIVEN_St_St1(self,t,i,j):
        K = self.K
        X = self.X
        B = self.B

        pXt_GIVEN_St_St1 = 1.0
        for k in range(K):
            if i != j:
                if X[k,t] == 0 and X[k,t-1] == 0:
                    pXt_GIVEN_St_St1 = 1-B[k,j]
                elif X[k,t] == 1 and X[k,t-1] == 0:
                    pXt_GIVEN_St_St1 = B[k,j]
                else:
                    pXt_GIVEN_St_St1 = 1.0
            else:
                if X[k,t] == 1 and X[k,t-1] == 0:
                    pXt_GIVEN_St_St1 = 0.0
                else:
                    pXt_GIVEN_St_St1 = 1.0

        return pXt_GIVEN_St_St1

    def computeBeta(self, Q, B0, B):
        M = self.M
        X = self.X
        Tn = self.Tn = X.shape[1]
        pS = self.pS = self.compute_pS(Q,M)
        observed_jumps = self.observed_jumps
        
        beta = np.zeros((M,Tn))
        beta[:,Tn-1] = 1
        for t in np.arange(Tn-1, 0, -1):
            tau_ind = np.where(self.step_sizes==observed_jumps[t-1])[0][0]
            for i in range(M):
                for j in range(M):
                    pXt_GIVEN_St_St1 = self.compute_pXt_GIVEN_St_St1(t,i,j)
                    beta[i,t-1] += beta[j,t]*pS[tau_ind,i,j]*pXt_GIVEN_St_St1

        return beta
    
    def drawState(self, pS):
        cdf = np.cumsum(pS)
        r = np.random.uniform() * cdf[-1]
        drawn_state = np.searchsorted(cdf, r)
        return drawn_state

    def compute_S0_GIVEN_X0(self, n_change_points_left):
        M = self.M
        K = self.K
        pi = self.pi
        B0 = self.B0
        pS0 = np.zeros(M)
        X = self.X
        for i in range(M):
            if (M-1) - i < n_change_points_left:
                pS0[i] = 0.0
                continue
            pX0 = 1.0
            for k in range(K):
                if X[k,0] == 1:
                    pX0 *= B0[k,i]
                else:
                    pX0 *= 1-B0[k,i]
            pS0[i] = pi[i] * pX0

        return pS0

    def compute_pSt_GIVEN_St1(self, i, t, n_change_points_left):
        M = self.M
        beta = self.computeBeta(self.Q, self.B0, self.B)
        pS = self.pS

        pSt_GIVEN_St1 = np.zeros(M)

        tau = self.observed_jumps[t]
        tau_ind = np.where(self.step_sizes == tau)[0][0]
        pSt_GIVEN_St1[range(i)] = 0.0
        for j in range(i,M):
            if (M-1) - j < n_change_points_left:
                pSt_GIVEN_St1[j] = 0.0
                continue

            pXt_GIVEN_St_St1 = self.compute_pXt_GIVEN_St_St1(t+1,i,j)
            if pXt_GIVEN_St_St1 == 0.0:
                n_change_points_left -= 1
            pSt_GIVEN_St1[j] = beta[j,t+1]/beta[i,t] * pS[tau_ind,i,j] * pXt_GIVEN_St_St1

        return pSt_GIVEN_St1, n_change_points_left

    def astep(self, q0):
        #X change points are the points in time where at least 
        #one comorbidity gets turned on. it's important to track
        #these because we have to make sure constrains on sampling
        #S are upheld. Namely S only goes up in time, and S changes
        #whenever there is an X change point. If we don't keep
        #track of how many change points are left we can't enforce
        #both of these constraints.
        M = self.M
        X = self.X
        Tn = self.Tn
        S = np.zeros(Tn, dtype=np.int8)
        n_change_points_left = len(np.where(np.sum(np.diff(X), axis=0) > 0)[0])

        #calculate pS0(i) | X, pi, B0
        pS0_GIVEN_X0 = self.compute_S0_GIVEN_X0(n_change_points_left)
        S[0] = self.drawState(pS0_GIVEN_X0)

        #calculate p(S_t=i | S_{t=1}=j, X, Q, B)
        #note: pS is probability of jump conditional on Q
        #whereas pS_ij is also conditional on everything else in the model
        #and is what we're looking for
        for t in range(0,Tn-1):
            i = S[t].astype(np.int)
            pSt_GIVEN_St1, n_change_points_left = self.compute_pSt_GIVEN_St1(i, t, n_change_points_left)
            S[t+1] = self.drawState(pSt_GIVEN_St1)

        return S

def evaluate_symbolic_shared(pi, Q, B0, B):
    f = theano.function([], [pi, Q, B0, B])
    return f