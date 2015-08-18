from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import logodds

from theano import function

class ForwardX(ArrayStepShared):
    """
    Use forward sampling (equation 10) to sample a realization of S_t, t=1,...,T_n
    given Q, B, and X constant.
    """
    def __init__(self, vars, N, T, D, O, max_obs, model=None):
        self.N = N
        self.T = T
        self.D = D
        self.O = O
        self.max_obs = max_obs

        model = modelcontext(model)
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)

        super(ForwardX, self).__init__(vars, shared)

        S = self.shared['S']
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        
        Z = logodds.backward(self.shared['Z_logodds'])
        L = logodds.backward(self.shared['L_logodds'])

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate them
        self.get_params = evaluate_symbolic_shared(S, B0, B, Z, L)

    def compute_pOt_GIVEN_Xt(self, n, k, t):
        not_k_idx = np.arange(self.K) != k
        pos_O_idx = np.in1d(np.arange(self.D), self.O[:,t,n])

        pOt_GIVEN_Xt = np.zeros(2)
        
        #Z's corresponding to positive and negative claims respectively
        Z_pos = self.Z[:,pos_O_idx]
        Z_neg = self.Z[:,~pos_O_idx]
        other_k = np.prod(1-self.X_prev[not_k_idx,t,n]* \
                    Z_pos[not_k_idx,:].T, axis=1)

        pOt_GIVEN_Xt = np.zeros(2)
        pOt_GIVEN_Xt[0] = np.prod(1-(1-self.L[pos_O_idx])*other_k)
        pOt_GIVEN_Xt[1] = np.prod(1-Z_neg[k,:]) * \
                            np.prod(1-(1-self.L[pos_O_idx])*(1-Z_pos[k,:]) * \
                                other_k)
        return pOt_GIVEN_Xt

    #computes beta for a single comorbidity k
    def computeBeta(self, n, k):
        beta = np.ones((2,self.max_obs))
        for t in np.arange(self.T[n]-1, 0, -1):
            
            Psi = np.zeros((2,2))
            Psi[1,1] = 1.0
            if self.S[n,t] == self.S[n,t-1]:
                Psi[0,0] = 1.0
                Psi[0,1] = 0.0
            else:
                Psi[0,0] = self.B[k,self.S[n,t]]
                Psi[0,1] = 1-self.B[k,self.S[n,t]]

            pOt_GIVEN_Xt = self.compute_pOt_GIVEN_Xt(n,k,t)

            beta[:,t-1] = np.sum(beta[:,t]*Psi*pOt_GIVEN_Xt, axis=1)

        return beta

    def compute_pX0_GIVEN_O0(self, k , n, beta):
        pOt_GIVEN_X0 = self.compute_pOt_GIVEN_Xt(n,k,0)

        pX0_GIVEN_O0 = beta[:,0] * pOt_GIVEN_X0 * \
                        np.array([self.B0[k,self.S[n,0]],1-self.B0[k,self.S[n,0]]])
        return pX0_GIVEN_O0

    def drawStateSingle(self, pX):
        cdf = np.cumsum(pX)
        r = np.random.uniform() * cdf[-1]
        drawn_state = np.searchsorted(cdf, r)
        return drawn_state

    def astep(self, X_previous):
        self.S, self.B0, self.B, self.Z, self.L = self.get_params()
        self.K = self.B.shape[0]
        self.X_prev = np.reshape(X_previous, (self.K,self.max_obs,self.N))

        X = np.zeros((self.K,self.max_obs,self.N), dtype=np.int8) - 1

        for n in xrange(self.N):
            for k in xrange(self.K):
                beta = self.computeBeta(n,k)
                pX0_GIVEN_O0 = self.compute_pX0_GIVEN_O0(k,n,beta)
                X[k,0,n] = self.drawStateSingle(pX0_GIVEN_O0)
                for t in xrange(0,self.T[n]-1):
                    Psi = np.zeros(2)
                    if self.S[n,t+1] == self.S[n,t]:
                        if X[k,t,n] != 1:
                            Psi[0] = 1.0
                        else:
                            Psi[1] = 1.0
                    else:
                        Psi[0] = self.B[k,self.S[n,t+1]]
                        Psi[1] = 1-self.B[k,self.S[n,t+1]]

                    pOt1_GIVEN_Xt1 = self.compute_pOt_GIVEN_Xt(n,k,t+1)
                    
                    pXt1 = beta[:,t+1] * Psi * pOt1_GIVEN_Xt1
                    X[k,t+1,n] = self.drawStateSingle(pXt1)

        return X

def evaluate_symbolic_shared(S,B0,B,Z,L):
    f = function([], [S,B0,B,Z,L])
    return f