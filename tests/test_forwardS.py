import numpy as np
from ContinuousTimeMarkovModel.distributions import *
from pymc3 import Model, Metropolis, Binomial, Beta
from ContinuousTimeMarkovModel.forwardS import *

M = 6
K = 10
D = 250
Tn = 20

observed_jumps = np.array([2,4,2,2,4,2,2,2,1,2,2,2,4,1,1,1,1,4,2])
S = np.array([0,1,1,2,2,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
X = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)

model = Model()

Q_test = np.array([[-6, 2, 2, 1, 1], 
   [1, -4, 0, 1, 2],
   [1, 0, -4, 2, 1],
   [2, 1, 0, -3, 0],
   [1, 1, 1, 1, -4]])

with model:
    Q = DiscreteObsMJP_unif_prior('Q', M=M, shape=(M,M))
    
    S = DiscreteObsMJP('S', Q=Q, observed_jumps=observed_jumps, shape=(Tn))

    B0 = Beta('B0', alpha = 1, beta = 1, shape=(K,M))
    B = Beta('B', alpha = 1, beta = 1, shape=(K,M))
    Xobs = Comorbidities('Xobs', S=S, B0=B0,B=B, shape=(K, Tn), observed = X)

    step2 = ForwardS(vars=[S], X=X, observed_jumps=observed_jumps)
    step2.step_sizes = np.array([0.1, 1, 100])
    pS = step2.compute_pS(Q_test, 5)
    print pS
