import unittest

import numpy as np
from ContinuousTimeMarkovModel.src.distributions import *
from pymc3 import Model, Dirichlet, Beta
from ContinuousTimeMarkovModel.src.forwardX import *
from theano.tensor import as_tensor_variable

N = 100 # Number of patients
M = 6 # Number of hidden states
K = 10 # Number of comorbidities
D = 721 # Number of claims
Dd = 80 # Maximum number of claims that can occur at once
min_obs = 10 # Minimum number of observed claims per patient
max_obs = 30 # Maximum number of observed claims per patient

from pickle import load
T = load(open('../data/X_layer_100_patients/T.pkl', 'rb'))
obs_jumps = load(open('../data/X_layer_100_patients/obs_jumps.pkl', 'rb'))
S_input = load(open('../data/X_layer_100_patients/S.pkl', 'rb'))
X_input = load(open('../data/X_layer_100_patients/X.pkl', 'rb'))
Z_input = load(open('../data/X_layer_100_patients/Z.pkl', 'rb'))
L_input = load(open('../data/X_layer_100_patients/L.pkl', 'rb'))
B_input = np.loadtxt('../data/synthetic/B.txt')
O = load(open('../data/X_layer_100_patients/O_input.pkl', 'rb'))

class ForwardXTestCase(unittest.TestCase):
  def setUp(self):
    self.model = Model()
    with self.model:
      pi = Dirichlet('pi', a = as_tensor_variable([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), shape=M)
      pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(pi) < .1, -np.inf, 0))

      Q = DiscreteObsMJP_unif_prior('Q', M=M, lower=0.0, upper=1.0, shape=(M,M))
      
      S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, N=N, observed_jumps=obs_jumps, T=T, shape=(N,max_obs))

      B0 = Beta('B0', alpha = 1, beta = 1, shape=(K,M))
      B = Beta('B', alpha = 1, beta = 1, shape=(K,M))

      X = Comorbidities('X', S=S, B0=B0,B=B, T=T, shape=(K, max_obs, N))

      Z = Beta('Z', alpha = 0.1, beta = 1, shape=(K,D))
      L = Beta('L', alpha = 1, beta = 1, shape=D)
      O_obs = Claims('O_obs', X=X, Z=Z, L=L, T=T, D=D, max_obs=max_obs, O_input=O, shape=(Dd,max_obs,N), observed=O)

      self.StepX = ForwardX(vars=[X], N=N, T=T, K=K, D=D, Dd=Dd, O=O, max_obs=max_obs)

  def testComputePsi(self):
    with self.model:
      Psi = self.StepX.computePsi(S_input, B_input)
      self.assertTrue(np.allclose(Psi[0,0,1,:,:], np.array([[1.0,0.0],[0.0,1.0]])), '0th user did not change state at time 1')
      self.assertTrue(np.allclose(Psi[0,0,4,:,:], np.array([[1.0,0.0],[0.0,1.0]])), '0th user did not change state at time 4')
      self.assertTrue(np.allclose(Psi[0,0,5,:,:], np.array([[0.3,0.7],[0.0,1.0]])), '0th user changed to state 4 at time 5')

  def testComputeLikelihoodOfXk(self):
    with self.model:
      LikelihoodOfXk = self.StepX.computeLikelihoodOfXk(9,X_input,Z_input,L_input)
      
      l0 = LikelihoodOfXk[0,:,:]
      self.assertTrue(np.allclose(l0[0,1]/l0[0,0],0.17522023559),'0th user at time 0 incorrect')
      #self.assertTrue(np.allclose(l0[22,1]/l0[22,0],10.7035445229),'0th user at time 22 incorrect')

      LikelihoodOfXk = self.StepX.computeLikelihoodOfXk(0,X_input,Z_input,L_input)
      l5 = LikelihoodOfXk[5,:,:]
      self.assertTrue(np.allclose(l5[0,1]/l5[0,0],0.631933932521),'5th user at time 0 incorrect')


suite = unittest.TestLoader().loadTestsFromTestCase(ForwardXTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)