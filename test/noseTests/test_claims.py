import unittest
import numpy as np
from theano.tensor import as_tensor_variable
from pymc3 import Model, sample, Metropolis, Dirichlet, Potential, Binomial, Beta, Slice
import theano.tensor as TT
from ContinuousTimeMarkovModel.src.distributions import *
from ContinuousTimeMarkovModel.src.forwardS import *
from ContinuousTimeMarkovModel.src.forwardX import *
from theano import function

class logpTests(unittest.TestCase):
    def setUp(self):
        #test Claims
        N = 5 # Number of patients
        M = 3 # Number of hidden states
        K = 2 # Number of comorbidities
        D = 20 # Number of claims
        Dd = 4 # Maximum number of claims that can occur at once
        min_obs = 2 # Minimum number of observed claims per patient
        max_obs = 4 # Maximum number of observed claims per patient
        #obs_jumps = np.ones((N,max_obs-1))
        obs_jumps = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
        T = np.array([4,2,3,4,2])
        nObs = T.sum()
        obs_jumps = np.hstack([np.zeros((N,1),dtype='int8'),obs_jumps])
        obs_jumps = np.concatenate([obs_jumps[i,0:T[i]] for i in range(N)])
        
        #O(4,4,5)
        #O = np.zeros((nObs,Dd),dtype='int8')
        O = np.zeros((Dd,max_obs,N),dtype='int8')
        #import pdb; pdb.set_trace()
        O[[0,1,3,2,3,3],[0,1,3,2,3,3],[0,1,4,3,3,4]] = 1
        #O[[0,5,11,12],[0,1,2,3]] = 1
        O = np.concatenate([O[:,0:T[i],i].T for i in range(N)])

        with Model() as self.model:
            self.pi = Dirichlet('pi', a = as_tensor_variable([0.5, 0.5, 0.5]), shape=M)
            pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(self.pi) < .1, -np.inf, 0))
            self.Q = DiscreteObsMJP_unif_prior('Q', M=M, lower=0.0, upper=1.0, shape=(M,M))
            self.S = DiscreteObsMJP('S', pi=self.pi, Q=self.Q, M=M, nObs=nObs, observed_jumps=obs_jumps, T=T, shape=(nObs))
            self.B0 = Beta('B0', alpha = 1., beta = 1., shape=(K,M))
            self.B = Beta('B', alpha = 1., beta = 1., shape=(K,M))
            self.X = Comorbidities('X', S=self.S, B0=self.B0,B=self.B, T=T, shape=(nObs,K))
            self.Z = Beta('Z', alpha = 0.1, beta = 1., shape=(K,D))
            self.L = Beta('L', alpha = 1., beta = 1., shape=D)
            #L = Beta('L', alpha = 0.1, beta = 1, shape=D, transform=None)
            #L = Uniform('L', left = 0.0, right = 1.0, shape=D, transform=None)
            #L = Uniform('L', lower = 0.0, upper = 1.0, shape=D)
            self.testClaims = Claims('O_obs', X=self.X, Z=self.Z, L=self.L, T=T, D=D, O_input=O, shape=(nObs,Dd), observed=O)


        self.myTestPoint = {'Z_logodds': np.array([[-2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509],
   [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509]]), 'Q_ratematrixoneway': np.array([ 0.1,  0.1]), 'pi_stickbreaking': np.array([0.2,0.1]), 'S': np.array([[0, 0, 1, 1],
   [1, 1, 1, 1],
   [1, 1, 2, 2],
   [0, 2, 2, 2],                                                                                                                                                                 
   [0, 0, 0, 1]], dtype=np.int32), 'B0_logodds': np.array([[ 0.,  1.,  0.],
   [ 0.,  0.,  1.]]), 'X': np.array([[[0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]],

   [[1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]]], dtype=np.int8), 'L_logodds': np.array([ 0.1,  0.1,  0.1,  0.1,  0.01,  0.01,  0.01,  0.01,  0.0011,  0.0011,  0.0011,  0.0011,  0.0011,
    0.,  0.0101,  0.0101,  0.0101,  0.01,  0.01,  0.01]), 'B_logodds': np.array([[ 1.,  0.,  1.],
   [ 0.,  1.,  0.]])}
        self.myTestPoint['S'] = np.concatenate([self.myTestPoint['S'][i,0:T[i]] for i in range(N)])
        self.myTestPoint['X'] = np.concatenate([self.myTestPoint['X'][:,0:T[i],i].T for i in range(N)])

        #import pdb; pdb.set_trace()

    def test_claims_pi_same_as_old(self):
        pi_LL = self.pi.transformed.logp(self.myTestPoint)
        pi_LL_Correct = -3.493851901732915
        np.testing.assert_almost_equal(pi_LL, pi_LL_Correct, decimal = 6, err_msg="logp of pi is incorrect")

    def test_claims_Q_same_as_old(self):
        with self.model:
            Q_LL = self.Q.transformed.logp(self.myTestPoint)
            Q_LL_Correct = -2.7775866402942837
            np.testing.assert_almost_equal(Q_LL, Q_LL_Correct, decimal = 6, err_msg="logp of Q is incorrect")

    def test_claims_S_same_as_old(self):
        with self.model:
            S_LL = self.S.logp(self.myTestPoint)
            S_LL_Correct = -12.165383116823502
            np.testing.assert_almost_equal(S_LL, S_LL_Correct, decimal = 6, err_msg="logp of S is incorrect")

    def test_claims_B0_same_as_old(self):
        with self.model:
            B0_LL = self.B0.transformed.logp(self.myTestPoint)
            B0_LL_Correct = -8.798224194552454
            np.testing.assert_almost_equal(B0_LL, B0_LL_Correct, decimal = 6, err_msg="logp of B0 is incorrect")

    def test_claims_B_same_as_old(self):
        with self.model:
            B_LL = self.B.transformed.logp(self.myTestPoint)
            B_LL_Correct = -9.038453208469008
            np.testing.assert_almost_equal(B_LL, B_LL_Correct, decimal = 6, err_msg="logp of B is incorrect")

    def test_claims_Z_same_as_old(self):
        with self.model:
            Z_LL = self.Z.transformed.logp(self.myTestPoint)
            Z_LL_Correct = -105.50739200312837
            np.testing.assert_almost_equal(Z_LL, Z_LL_Correct, decimal = 6, err_msg="logp of Z is incorrect")

    def test_claims_X_same_as_old(self):
        with self.model:
            X_LL = self.X.logp(self.myTestPoint)
            X_LL_Correct = -8.2511423611958445
            #X_LL_Correct = -13.10317262511546
            np.testing.assert_almost_equal(X_LL, X_LL_Correct, decimal = 6, err_msg="logp of X is incorrect")

    def test_claims_L_same_as_old(self):
        with self.model:
            L_LL = self.L.transformed.logp(self.myTestPoint)
            L_LL_Correct = -27.736136077452397
            np.testing.assert_almost_equal(L_LL, L_LL_Correct, decimal = 6, err_msg="logp of O is incorrect")

    def test_claims_O_same_as_old(self):
        with self.model:
            defaultVal = self.testClaims.logp(self.model.test_point)
            defaultCorrect = -258.46778148992826
            claims_LL = self.testClaims.logp(self.myTestPoint)
            claims_LL_Correct = -252.1491658970591

            np.testing.assert_almost_equal(defaultVal, defaultCorrect, decimal = 6, err_msg="logp of O is incorrect for default input")
            np.testing.assert_almost_equal(claims_LL, claims_LL_Correct, decimal = 6, err_msg="logp of O is incorrect")

if __name__ == '__main__':
    unittest.main()


#        self.myTestPointNew = {'Z_logodds': np.array([[-2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
#    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
#    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
#    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509],
#   [-2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
#    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
#    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509,
#    -2.30258509, -2.30258509, -2.30258509, -2.30258509, -2.30258509]]), 'Q_ratematrixoneway': np.array([ 0.1,  0.1]), 'pi_stickbreaking': np.array([0.2,0.1]), 'S': np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 0, 0], dtype=np.int32), 'B0_logodds': np.array([[ 0.,  1.,  0.],
#   [ 0.,  0.,  1.]]), 'X': np.ones((nObs,K), dtype=np.int8), 'L_logodds': np.array([ 0.1,  0.1,  0.1,  0.1,  0.01,  0.01,  0.01,  0.01,  0.0011,  0.0011,  0.0011,  0.0011,  0.0011,
#    0.,  0.0101,  0.0101,  0.0101,  0.01,  0.01,  0.01]), 'B_logodds': np.array([[ 1.,  0.,  1.],
#   [ 0.,  1.,  0.]])}
#
