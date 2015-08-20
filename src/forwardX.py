from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import logodds

from theano import function

import ContinuousTimeMarkovModel.src.cython.compute_prod_other_k as compute_prod_other_k

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

        self.pos_O_idx = np.zeros((D,max_obs,N), dtype=np.bool_)
        for n in xrange(N):
            for t in xrange(0,self.T[n]-1):
                self.pos_O_idx[:,t,n] = np.in1d(np.arange(self.D), self.O[:,t,n])

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

    def sampleState(self, pX):
        tot_prob = np.sum(pX, axis=0)
        r = np.random.uniform(self.K) * tot_prob
        drawn_state = np.greater_equal(r, pX[0,:])
        return drawn_state.astype(np.int8)

    def astep(self, X):
        self.S, self.B0, self.B, self.Z, self.L = self.get_params()
        self.K = self.B.shape[0]
        self.X = np.reshape(X, (self.K,self.max_obs,self.N))

        X_new = np.zeros((self.K,self.max_obs,self.N), dtype=np.int8) - 1

        #note we keep Psi and pOt_GIVEN_Xt because they are used
        #in the computation of beta ADN then again in the sampling forward of X
        beta = np.ones((2,self.K,self.max_obs))
        Psi = np.zeros((2,2,self.K,self.max_obs))
        pOt_GIVEN_Xt = np.zeros((2,self.K,self.max_obs))
        for n in xrange(self.N):
            Xn = self.X[:,:,n]
            pos_O_idx_n = self.pos_O_idx[:,:,n]
            pos_O_idx_n[0] = True
            pos_O_idx_n[1] = True
            
            #(1)compute beta a.k.a. the backwards variables a.k.a. 
            #likelihood of X given the entire time series of observations
            for t in np.arange(self.T[n]-1, -1, -1):
                #(A) Compute Psi which is the probability of jumping to state X_{t+1}=j
                #given you're in state X_{t}=i and S_{t}=m. Note the probability of
                #getting the comorbidity once you already have it is one. The prob.
                #of not having the comorbidity once you've already had it is zero
                Psi[1,1,:,t] = 1.0

                #if you did NOT change state the probability of getting a new
                #comorbidity is zero. If you did change state the new state
                #has comordbity onsets associated with it i.e. B
                if self.S[n,t] == self.S[n,t-1]:
                    Psi[0,0,:,t] = 1.0
                    Psi[0,1,:,t] = 0.0
                else:
                    Psi[0,0,:,t] = self.B[:,self.S[n,t]]
                    Psi[0,1,:,t] = 1-self.B[:,self.S[n,t]]

                #(B) Compute pOt_GIVEN_Xt i.e. the likelihood of X_t,k given Ot and all
                #other X_t,l where l =/= k
                pos_O_idx_n_t = pos_O_idx_n[:,t]
                Z_pos = self.Z[:,pos_O_idx_n_t]
                
                #compute prod_other_k which is product term in eq. 13 over k' \neq k
                #we compute it for all k, i.e. the kth row is the product of all k's
                #except that k. we use Cython here
                XZ_t = (Xn[:,t]*Z_pos.T).T
                n_pos_O = np.sum(pos_O_idx_n_t)
                prod_other_k = np.zeros((self.K, n_pos_O))

                prod_other_k = compute_prod_other_k.compute(XZ_t, n_pos_O, self.K)
                
                pOt_GIVEN_Xt[0,:,t] = np.prod(1-self.L[pos_O_idx_n_t]* \
                             prod_other_k, axis=1)
                pOt_GIVEN_Xt[1,:,t] = np.prod(1-self.Z[:,np.logical_not(pos_O_idx_n_t)], axis=1) * \
                        np.prod(1 - (1-self.L[pos_O_idx_n_t])* \
                            (1-Z_pos)*prod_other_k, axis=1)

                #(C) Now actually set the beta (finally)
                #we want this loop to go down to zero so we compute pOt_GIVEN_Xt[0]
                #which we need in section (2) to sample the initial X, but obviously
                #we won't want to go down to beta[-1] so we skip this part. Just
                # a little trick to not have to repeat that code to get pOt_GIVEN_Xt[0]
                if t < 0:
                    break
                beta[:,:,t-1] = np.sum(beta[:,:,t] * Psi[:,:,:,t] * \
                    pOt_GIVEN_Xt[:,:,t], axis=1)

            #(2)sample X_new
            #(A) Sample starting comorbidities
            pX0_GIVEN_O0 = beta[:,:,0] * \
                np.array([self.B0[:,self.S[n,0]],1-self.B0[:,self.S[n,0]]]) * \
                pOt_GIVEN_Xt[:,:,0]
            X_new[:,0,n] = self.sampleState(pX0_GIVEN_O0)

            #(B) Sample rest of X's through time
            for t in xrange(0,self.T[n]-1):
                Xnt = X_new[:,t,n]
                pXt_next = beta[:,:,t+1] * Psi[Xnt,:,0,t+1].T * pOt_GIVEN_Xt[:,:,t+1]
                X_new[:,t+1,n] = self.sampleState(pXt_next)

        return X_new

def evaluate_symbolic_shared(S,B0,B,Z,L):
    f = function([], [S,B0,B,Z,L])
    return f