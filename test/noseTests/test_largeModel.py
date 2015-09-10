import unittest
from scipy.stats import logistic
import numpy as np
from theano.tensor import as_tensor_variable
from pymc3 import Model, sample, Metropolis, Dirichlet, Potential, Binomial, Beta, Slice
import theano.tensor as TT
from ContinuousTimeMarkovModel.src.distributions import *
from ContinuousTimeMarkovModel.src.forwardS import *
from ContinuousTimeMarkovModel.src.forwardX import *
from theano import function
from pickle import load

class logpTests(unittest.TestCase):
    def setUp(self):
        #test Claims
        N = 100 # Number of patients
        M = 6 # Number of hidden states
        K = 10 # Number of comorbidities
        D = 721 # Number of claims
        Dd = 80 # Maximum number of claims that can occur at once
        min_obs = 10 # Minimum number of observed claims per patient
        max_obs = 30 # Maximum number of observed claims per patient
        self.M = M
        self.N = N
        self.K = K
        self.max_obs = max_obs
        # Load pre-generated data

        T = load(open('../../data/X_layer_100_patients/T.pkl', 'rb'))
        self.T = T
        obs_jumps = load(open('../../data/X_layer_100_patients/obs_jumps.pkl', 'rb'))
        S_start = load(open('../../data/X_layer_100_patients/S.pkl', 'rb'))
        X_start = load(open('../../data/X_layer_100_patients/X.pkl', 'rb'))
        Z_start = load(open('../../data/X_layer_100_patients/Z.pkl', 'rb'))
        L_start = load(open('../../data/X_layer_100_patients/L.pkl', 'rb'))
        O = load(open('../../data/X_layer_100_patients/O_input.pkl', 'rb'))

        with Model() as self.model:
            self.pi = Dirichlet('pi', a = as_tensor_variable([0.5,0.5,0.5,0.5,0.5,0.5]), shape=M)
            pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(self.pi) < .1, -np.inf, 0))
            self.Q = DiscreteObsMJP_unif_prior('Q', M=M, lower=0.0, upper=1.0, shape=(M,M))
            self.S = DiscreteObsMJP('S', pi=self.pi, Q=self.Q, M=M, N=N, observed_jumps=obs_jumps, T=T, shape=(N,max_obs))
            self.B0 = Beta('B0', alpha = 1., beta = 1., shape=(K,M))
            self.B = Beta('B', alpha = 1., beta = 1., shape=(K,M))
            self.X = Comorbidities('X', S=self.S, B0=self.B0,B=self.B, T=T, shape=(K, max_obs, N))
            self.Z = Beta('Z', alpha = 0.1, beta = 1., shape=(K,D))
            self.L = Beta('L', alpha = 1., beta = 1., shape=D)
            #L = Beta('L', alpha = 0.1, beta = 1, shape=D, transform=None)
            #L = Uniform('L', left = 0.0, right = 1.0, shape=D, transform=None)
            #L = Uniform('L', lower = 0.0, upper = 1.0, shape=D)
            self.testClaims = Claims('O_obs', X=self.X, Z=self.Z, L=self.L, T=T, D=D, max_obs=max_obs, O_input=O, shape=(Dd,max_obs,N), observed=O)

            self.forS = ForwardS(vars=[self.S], N=N, T=T, max_obs=max_obs, observed_jumps=obs_jumps)
            self.forX = ForwardX(vars=[self.X], N=N, T=T, K=K, D=D,Dd=Dd, O=O, max_obs=max_obs)

        from scipy.special import logit

        self.Q_raw_log = logit(np.array([0.631921, 0.229485, 0.450538, 0.206042, 0.609582]))

        B_lo = logit(np.array([
        [0.000001,0.760000,0.720000,0.570000,0.700000,0.610000],
        [0.000001,0.460000,0.390000,0.220000,0.200000,0.140000],
        [0.000001,0.620000,0.620000,0.440000,0.390000,0.240000],
        [0.000001,0.270000,0.210000,0.170000,0.190000,0.070000],
        [0.000001,0.490000,0.340000,0.220000,0.160000,0.090000],
        [0.000001,0.620000,0.340000,0.320000,0.240000,0.120000],
        [0.000001,0.550000,0.390000,0.320000,0.290000,0.150000],
        [0.000001,0.420000,0.240000,0.170000,0.170000,0.110000],
        [0.000001,0.310000,0.300000,0.230000,0.190000,0.110000],
        [0.000001,0.470000,0.340000,0.190000,0.190000,0.110000]]))

        B0_lo = logit(np.array([
        [0.410412,0.410412,0.418293,0.418293,0.429890,0.429890],
        [0.240983,0.240983,0.240983,0.240983,0.240983,0.240983],
        [0.339714,0.339714,0.339714,0.339714,0.339714,0.339714],
        [0.130415,0.130415,0.130415,0.130415,0.130415,0.130415],
        [0.143260,0.143260,0.143260,0.143260,0.143260,0.143260],
        [0.211465,0.211465,0.211465,0.211465,0.211465,0.211465],
        [0.194187,0.194187,0.194187,0.194187,0.194187,0.194187],
        [0.185422,0.185422,0.185422,0.185422,0.185422,0.185422],
        [0.171973,0.171973,0.171973,0.171973,0.171973,0.171973],
        [0.152277,0.152277,0.152277,0.152277,0.152277,0.152277]]))

        Z_lo = logit(Z_start)
        L_lo = logit(L_start)
        #import pdb; pdb.set_trace()
        self.myTestPoint = {'Q_ratematrixoneway': self.Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo, 'S':S_start, 'X':X_start, 'Z_logodds':Z_lo, 'L_logodds':L_lo, 'pi_stickbreaking':np.array([0.5,0.5,0.5,0.5,0.5,0.5])}

#    def test_claims_pi_same_as_old(self):
#        pi_LL = self.pi.transformed.logp(self.myTestPoint)
#        pi_LL_Correct = -3.493851901732915
#        np.testing.assert_almost_equal(pi_LL, pi_LL_Correct, decimal = 6, err_msg="logp of pi is incorrect")

    def test_claims_Q_same_as_old(self):
        with self.model:
            Q_LL = self.Q.transformed.logp(self.myTestPoint)
            Q_LL_Correct = -7.833109951183136
            np.testing.assert_almost_equal(Q_LL, Q_LL_Correct, decimal = 6, err_msg="logp of Q is incorrect")

    def test_claims_S_same_as_old(self):
        with self.model:
            S_LL = self.S.logp(self.myTestPoint)
            S_LL_Correct = -602.4680678270764
            np.testing.assert_almost_equal(S_LL, S_LL_Correct, decimal = 6, err_msg="logp of S is incorrect")

    def test_claims_B0_same_as_old(self):
        with self.model:
            B0_LL = self.B0.transformed.logp(self.myTestPoint)
            B0_LL_Correct = -110.48096505515802
            np.testing.assert_almost_equal(B0_LL, B0_LL_Correct, decimal = 6, err_msg="logp of B0 is incorrect")

    def test_claims_B_same_as_old(self):
        with self.model:
            B_LL = self.B.transformed.logp(self.myTestPoint)
            B_LL_Correct = -224.71212579379755
            np.testing.assert_almost_equal(B_LL, B_LL_Correct, decimal = 6, err_msg="logp of B is incorrect")

    def test_claims_Z_same_as_old(self):
        with self.model:
            Z_LL = self.Z.transformed.logp(self.myTestPoint)
            Z_LL_Correct = -21916.630568050998
            np.testing.assert_almost_equal(Z_LL, Z_LL_Correct, decimal = 6, err_msg="logp of Z is incorrect")

    def test_claims_X_same_as_old(self):
        with self.model:
            X_LL = self.X.logp(self.myTestPoint)
            X_LL_Correct = -1069.7455676992997
            np.testing.assert_almost_equal(X_LL, X_LL_Correct, decimal = 6, err_msg="logp of X is incorrect")

    def test_claims_L_same_as_old(self):
        with self.model:
            L_LL = self.L.transformed.logp(self.myTestPoint)
            L_LL_Correct = -5461.275093993832
            np.testing.assert_almost_equal(L_LL, L_LL_Correct, decimal = 6, err_msg="logp of O is incorrect")

    def test_claims_O_same_as_old(self):
        with self.model:
            defaultVal = self.testClaims.logp(self.model.test_point)
            defaultCorrect = -2280947.613994252
            claims_LL = self.testClaims.logp(self.myTestPoint)
            claims_LL_Correct = -227668.7133875753

            np.testing.assert_almost_equal(defaultVal, defaultCorrect, decimal = 6, err_msg="logp of O is incorrect for default input")
            np.testing.assert_almost_equal(claims_LL, claims_LL_Correct, decimal = 6, err_msg="logp of O is incorrect")

    def test_forwardS_same_as_old(self):
#        Qvals = logistic.cdf(self.Q_raw_log)
#        Qtest = np.zeros((self.M,self.M))
#        for i in range(self.M-1):
#            Qtest[i,i+1] = Qvals[i]
#            Qtest[i,i] = -Qvals[i]
        self.forS.astep(0.)
        pS0_Test = self.forS.compute_S0_GIVEN_X0()
        #pS_Test = self.forS.compute_pS(Qtest,self.M)
        pS0_Correct = load(open('pS0_Test_data.pkl', 'rb'))

        pSnt_Test = np.zeros((self.N,self.max_obs,self.M))
        for n in xrange(self.N):
            for t in xrange(self.T[n]-1):
                pSnt_Test[n,t] = self.forS.compute_pSt_GIVEN_St1(n,t,self.myTestPoint['S'][n,t])
        pSnt_Correct = load(open('pSnt_Test_data.pkl', 'rb'))

        #import pdb; pdb.set_trace()
        np.testing.assert_array_almost_equal(pS0_Test, pS0_Correct, err_msg="forwardS test off",decimal = 6)
        np.testing.assert_array_almost_equal(pSnt_Test, pSnt_Correct, err_msg="forwardS test off",decimal = 6)

    def test_forwardX_same_as_old(self):
        #pX_Test = self.forX.computeLikelihoodOfXk(0,self.myTestPoint['X'],logistic.cdf(self.myTestPoint['Z_logodds']),logistic.cdf(self.myTestPoint['L_logodds']))
        pX_Test = np.zeros((self.N,self.max_obs,self.K,2))
        Psi = self.forX.computePsi(self.myTestPoint['S'],logistic.cdf(self.myTestPoint['B_logodds']))
        for k in range(self.K):
            LikelihoodOfXk = self.forX.computeLikelihoodOfXk(k,self.myTestPoint['X'],logistic.cdf(self.myTestPoint['Z_logodds']),logistic.cdf(self.myTestPoint['L_logodds']))
            beta = self.forX.computeBeta(k,Psi,LikelihoodOfXk)
            pX_Test[:,0,k,:] = self.forX.computePX0(k,beta,logistic.cdf(self.myTestPoint['B0_logodds']),self.myTestPoint['S'],LikelihoodOfXk)
            for t in range(self.max_obs-1):
                pX_Test[:,t+1,k,:] = self.forX.computePXt(k,t,beta,self.myTestPoint['X'],Psi,LikelihoodOfXk)
        #import pdb; pdb.set_trace()
        pX_Correct = load(open('pX_Test_data.pkl','rb'))
        np.testing.assert_array_almost_equal(pX_Test, pX_Correct, err_msg="forwardX likelihood test off",decimal = 6)

if __name__ == '__main__':
    unittest.main()
