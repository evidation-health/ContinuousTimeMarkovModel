from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import stick_breaking, logodds
from .transforms import rate_matrix, rate_matrix_one_way
from scipy import linalg

import theano

import time

class ForwardS(ArrayStepShared):
    """
    Use forward sampling (equation 10) to sample a realization of S_t, t=1,...,T_n
    given Q, B, and X constant.
    """
    def __init__(self, vars, N, T, max_obs, observed_jumps, model=None):
        self.N = N
        self.T = T
        self.max_obs = max_obs

        model = modelcontext(model)
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)

        super(ForwardS, self).__init__(vars, shared)
        
        self.observed_jumps = observed_jumps
        step_sizes = np.sort(np.unique(observed_jumps))
        self.step_sizes = step_sizes = step_sizes[step_sizes > 0]

        pi = stick_breaking.backward(self.shared['pi_stickbreaking'])
        lower = model.free_RVs[1].distribution.dist.lower
        upper = model.free_RVs[1].distribution.dist.upper
        Q = rate_matrix_one_way(lower, upper).backward(self.shared['Q_ratematrixoneway'])
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        X = self.shared['X']

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate them
        self.get_params = evaluate_symbolic_shared(pi, Q, B0, B, X)

    def compute_pS(self,Q,M):
        pS = np.zeros((len(self.step_sizes), M, M))

        for tau in self.step_sizes:
            pS_tau = linalg.expm(tau*Q)
            tau_ind = np.where(self.step_sizes == tau)[0][0]
            pS[tau_ind,:,:] = pS_tau

        return pS

    def computeBeta(self, Q, B0, B):
        M = self.M
        X = self.X
        T = self.T
        pS = self.pS = self.compute_pS(Q,M)
        observed_jumps = self.observed_jumps
        
        beta = np.ones((M,self.max_obs,self.N))
        for n in xrange(self.N):
            for t in np.arange(T[n]-1, 0, -1):
                tau_ind = np.where(self.step_sizes==observed_jumps[n,t-1])[0][0]
                was_changed = X[:,t,n] != X[:,t-1,n]
                pXt_GIVEN_St_St1 = np.prod(B[was_changed,:], axis=0) * np.prod(1-B[~was_changed,:], axis=0)
                pXt_GIVEN_St_St1 = np.tile([pXt_GIVEN_St_St1], (M,1))
                np.fill_diagonal(pXt_GIVEN_St_St1,np.float(not np.any(was_changed)))
                beta[:,t-1,n] = np.sum(beta[:,t,n]*pS[tau_ind,:,:]*pXt_GIVEN_St_St1, axis=1)

        return beta
    
    def drawState(self, pS):
        cdf = np.cumsum(pS, axis=1)
        r = np.random.uniform(size=self.N) * cdf[:,-1]
        drawn_state = np.zeros(self.N)
        for n in range(self.N):
            drawn_state[n] = np.searchsorted(cdf[n,:], r[n])
        return drawn_state

    def drawStateSingle(self, pS):
        cdf = np.cumsum(pS)
        r = np.random.uniform() * cdf[-1]
        drawn_state = np.searchsorted(cdf, r)
        return drawn_state

    def compute_S0_GIVEN_X0(self):
        N = self.N
        M = self.M
        K = self.K
        pi = self.pi
        B0 = self.B0
        X = self.X

        pS0 = np.zeros((N,M))
        for n in xrange(N):
            on = X[:,0,n] == 1
            off = np.invert(on)
            pX0 = np.prod(1-B0[off,:],axis=0) * np.prod(B0[on,:],axis=0)
            pX_GIVEN_S0 = self.beta[:,0,n]
            pS0[n,:] = pi * pX_GIVEN_S0 * pX0

        return pS0

    def astep(self, q0):
        #X change points are the points in time where at least 
        #one comorbidity gets turned on. it's important to track
        #these because we have to make sure constrains on sampling
        #S are upheld. Namely S only goes up in time, and S changes
        #whenever there is an X change point. If we don't keep
        #track of how many change points are left we can't enforce
        #both of these constraints.
        self.pi, self.Q, self.B0, self.B, self.X=self.get_params()

        K = self.K = self.X.shape[0]
        M = self.M = self.Q.shape[0]
        T = self.T
        X = self.X
        S = np.zeros((self.N,self.max_obs), dtype=np.int8) - 1

        #calculate pS0(i) | X, pi, B0
        beta = self.beta = self.computeBeta(self.Q, self.B0, self.B)
        pS0_GIVEN_X0 = self.compute_S0_GIVEN_X0()
        S[:,0] = self.drawState(pS0_GIVEN_X0)

        #calculate p(S_t=i | S_{t=1}=j, X, Q, B)
        #note: pS is probability of jump conditional on Q
        #whereas pS_ij is also conditional on everything else in the model
        #and is what we're looking for
        B = self.B
        observed_jumps = self.observed_jumps
        pS = self.pS
        
        for n in xrange(self.N):
            for t in xrange(0,T[n]-1):
                #import pdb; pdb.set_trace()
                i = S[n,t].astype(np.int)

                was_changed = X[:,t+1,n] != X[:,t,n]

                pXt_GIVEN_St_St1 = np.prod(B[was_changed,:], axis=0) * np.prod(1-B[~was_changed,:], axis=0)
                if np.any(was_changed):
                    pXt_GIVEN_St_St1[i] = 0.0
                else:
                    pXt_GIVEN_St_St1 = 1.0

                tau_ind = np.where(self.step_sizes==observed_jumps[n,t])[0][0]
                
                #don't divide by beta_t it's just a constant anyway
                pSt_GIVEN_St1 = beta[:,t+1,n] * pS[tau_ind,i,:] * pXt_GIVEN_St_St1

                #make sure not to go backward or forward too far
                #pSt_GIVEN_St1[0:i] = 0.0
                
                S[n,t+1] = self.drawStateSingle(pSt_GIVEN_St1)

        return S
        #return q0

def evaluate_symbolic_shared(pi, Q, B0, B, X):
    f = theano.function([], [pi, Q, B0, B, X])
    return f
