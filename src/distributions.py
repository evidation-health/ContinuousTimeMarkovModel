from pymc3 import Continuous
from pymc3.distributions.discrete import Binomial
from .transforms import rate_matrix
import numpy as np
import theano.tensor as T
from theano.tensor.nlinalg import eig, matrix_inverse
from theano.compile.sharedvalue import shared

class DiscreteObsMJP_unif_prior(Continuous):

    def __init__(self, M,transform=rate_matrix, *args, **kwargs):
        super(DiscreteObsMJP_unif_prior, self).__init__(transform=transform, *args, **kwargs)
        Q = np.ones((M, M))
        self.mode = Q

    def logp(self, value):
        return T.as_tensor_variable(0)

class DiscreteObsMJP(Continuous):

    def __init__(self, pi, Q, observed_jumps, *args, **kwargs):
        super(DiscreteObsMJP, self).__init__(dtype='int32',*args, **kwargs)
        self.pi = pi
        self.Q = Q
        self.observed_jumps = observed_jumps
        self.step_sizes = np.unique(observed_jumps)
        self.mode = 0

    def computeC(self,S):
    	Q = self.Q
        step_sizes = self.step_sizes
        observed_jumps = self.observed_jumps
        n_step_sizes = len(self.step_sizes)

        nrows_s, ncols_s = Q.shape
        n_step_sizes_s = T.as_tensor_variable(n_step_sizes)
        
        #create C
        C = T.alloc(0, *(nrows_s, ncols_s, n_step_sizes_s))
        C.name = 'C'
        
        #fill C
        for i in range(len(observed_jumps)):
            tau = observed_jumps[i]
            tau_index = 0
            for j in range(len(step_sizes)):
                if tau == step_sizes[j]:
                    tau_index = j
                    break
            #tau_index = np.where(step_sizes==tau)[0].item()
            
            Ci = S[i]
            Cj = S[i+1]
            C = T.set_subtensor(C[Ci, Cj, tau_index], 
                                     C[Ci, Cj, tau_index]+1)
        
        return C
        
    def logp(self, S):
    	l = 0.0

    	#add prior
    	pi = self.pi
    	l += pi[S[0]]

    	#add likelihood
        Q = self.Q
        step_sizes = self.step_sizes

        Q_complex = T.cast(Q, 'complex64')
        C = self.computeC(S)

        for i in range(0, len(step_sizes)):
            #get P(tau)
            lambdas, U = eig(Q_complex)
            
            tau = step_sizes[i]
            exp_tD = T.diag(T.exp(tau*lambdas))

            U_inv = matrix_inverse(U)

            P = U.dot(exp_tD).dot(U_inv)
        
            #compute likelihood in terms of P(tau)
            l += T.sum(C[:,:,i]*T.log(P))
            
        return l

class Comorbidities(Continuous):
    def __init__(self, S, B0, B, shape, *args, **kwargs):
        super(Comorbidities, self).__init__(shape = shape, dtype='int8',*args, **kwargs)
        X = np.ones(shape)
        self.K = shape[0]
        self.Tn = shape[1]
        self.S = S
        self.B0 = B0
        self.B = B
        self.mode = X

    def logp(self, X):
        K = self.K
        Tn = self.Tn
        S = self.S
        B0 = self.B0
        B = self.B
        
        l = 0.0
        for k in range(K):
            l += Binomial.dist(n=1, p=B0[k,S[0]]).logp(X[k,0])
            
        for t in range(1,Tn):
            for k in range(K):
                if T.eq(S[t],S[t-1]) or X[k,t-1] == 1:
                    l += 0.0
                else:
                    l += Binomial.dist(n=1, p=B[k,S[t]]).logp(X[k,t])
        
        return l