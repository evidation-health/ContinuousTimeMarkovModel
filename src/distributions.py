from pymc3 import Continuous
from pymc3.distributions.discrete import Categorical, Binomial
from .transforms import rate_matrix, rate_matrix_one_way
import numpy as np
import theano.tensor as TT
from theano.tensor.nlinalg import eig, matrix_inverse
from theano.compile.sharedvalue import shared
import theano.tensor.slinalg
from theano.tensor.extra_ops import bincount

class DiscreteObsMJP_unif_prior(Continuous):

    def __init__(self, M, lower=0, upper=100, *args, **kwargs):
        self.lower = lower
        self.upper = upper
        super(DiscreteObsMJP_unif_prior, self).__init__(transform=rate_matrix_one_way(lower=lower, upper=upper), 
            *args, **kwargs)
        Q = np.ones((M, M), np.float64) - 0.5
        self.mode = Q

    def logp(self, value):
        return TT.as_tensor_variable(0.0)

class DiscreteObsMJP(Continuous):

    def __init__(self, pi, Q, M, N, observed_jumps, T, *args, **kwargs):
        super(DiscreteObsMJP, self).__init__(dtype='int32',*args, **kwargs)
        self.pi = pi
        self.Q = Q
        self.M = M
        self.observed_jumps = observed_jumps
        self.T = T
        step_sizes = np.sort(np.unique(observed_jumps))
        self.step_sizes = step_sizes = step_sizes[step_sizes > 0]
        max_obs = observed_jumps.shape[1]+1
        self.mode = np.ones((N,max_obs), dtype=np.int32)

        #pad observed jumps with -1 for later use in computeC
        self.N = N
        obs_jumps = np.insert(observed_jumps, max_obs-1, -1, axis=1)
        for n in range(N):
        	obs_jumps[n,T[n]-1:] = -1

        #convert observed jumps to their appropriate array index
		obs_jump_ind = obs_jumps.copy()
		for index, step in enumerate(step_sizes):
		    obs_jump_ind[obs_jumps == step] = index
		self.obs_jump_ind = obs_jump_ind

    def computeC(self,S):
    	M = self.M
    	n_step_sizes = len(self.step_sizes)

    	obs_jump_ind = TT.as_tensor_variable(self.obs_jump_ind, 'obs_jump_ind')
    	tau_ind = TT.flatten(obs_jump_ind)[:-1]*M*M
    	keep_jumps = (tau_ind >= 0).nonzero()
    	
    	jump_from_ind = TT.flatten(S)[:-1]*M
    	jump_to_ind = TT.flatten(S)[1:]

    	flat_ind = (tau_ind + jump_from_ind + jump_to_ind)[keep_jumps]
    	flat_ind_counts = bincount(flat_ind, minlength=n_step_sizes*M*M)

    	C = flat_ind_counts.reshape(shape=np.array([n_step_sizes,M,M]))
        
        return C
        
    def logp(self, S):
    	l = 0.0

    	#add prior
    	pi = self.pi
    	l += TT.sum(TT.log(pi[S[:,0]]))

    	#add likelihood
        Q = self.Q
        step_sizes = self.step_sizes

        C = self.computeC(S)

        n_step_sizes = len(self.step_sizes)
        for i in range(0, n_step_sizes):
            tau = step_sizes[i]
            P = TT.slinalg.expm(tau*Q)
            
            #compute likelihood in terms of P(tau)
            l += TT.sum(C[i,:,:]*TT.log(P))
            
        return l

from theano.compile.ops import as_op

X_theano_type = TT.TensorType('int8', [False, False, False])
@as_op(itypes=[TT.dscalar, TT.bscalar, TT.dmatrix, TT.dmatrix, X_theano_type, TT.imatrix, TT.lvector], otypes=[TT.dscalar])
def logp_numpy(l,N,B0,B,X,S,T):
	for n in xrange(N):
		pX0 = np.prod(B0[X[:,0,n] == 1, S[n,0]]) * np.prod(1-B0[X[:,0,n] != 1, S[n,0]])
		l += np.log(pX0)

		for t in range(1,T[n]):
			if S[n,t] != S[n,t-1]:
				turned_on = ((X[:,t-1,n] == 0) & (X[:,t,n] == 1))
				stayed_off = ((X[:,t-1,n] == 0) & (X[:,t,n] == 0))
				l += np.log(np.prod(B[turned_on, S[n,t]]))
				l += np.log(np.prod(1-B[stayed_off, S[n,t]]))

		return l
		#for t in range(1,T[n]):

class Comorbidities(Continuous):
    def __init__(self, S, B0, B, T, shape, *args, **kwargs):
        super(Comorbidities, self).__init__(shape = shape, dtype='int8',*args, **kwargs)
        X = np.ones(shape, dtype='int8')
        self.K = shape[0]
        self.max_obs = shape[1]
        self.N = shape[2]
        self.T = T
        self.S = S
        self.B0 = B0
        self.B = B
        self.mode = X

    def logp(self, X):
        K = self.K
        max_obs = self.max_obs
        N = self.N
        T = self.T
        S = self.S
        B0 = self.B0
        B = self.B

        l = np.float64(0.0)
        #import pdb; pdb.set_trace()
        l = logp_numpy(TT.as_tensor_variable(l),TT.as_tensor_variable(N),B0,B,X,S,TT.as_tensor_variable(T))        
        '''
        for n in xrange(N):
        	#likelihood of X0
        	
			
        	l += TT.sum(Binomial.dist(n=1, p=B0[:,S[n,0]]).logp(X[:,0,n]))
            
	        for t in range(1,T[n]):
	            if TT.eq(S[n,t],S[n,t-1]):
	            	l += 0.0
	            else:
	            	turned_on = (X[:,t-1,n] == 1).nonzero()
	            	stayed_off = (X[:,t-1,n] != 1).nonzero()
	            	l += TT.sum(Binomial.dist(n=1, p=B[turned_on,S[n,t]]).logp(X[turned_on,t,n]))
	            	l += TT.sum(Binomial.dist(n=1, p=B[stayed_off,S[n,t]]).logp(X[stayed_off,t,n]))
	    '''

        
        return l