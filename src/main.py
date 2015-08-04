import numpy as np
from theano.tensor import as_tensor_variable
from ContinuousTimeMarkovModel.src.distributions import *
from pymc3 import Model, Metropolis, Dirichlet, Binomial, Beta
from ContinuousTimeMarkovModel.src.forwardS import *

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

with model:
    pi = Dirichlet('pi', a = as_tensor_variable([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), shape=M)
    Q = DiscreteObsMJP_unif_prior('Q', M=M, shape=(M,M))
    
    S = DiscreteObsMJP('S', pi=pi, Q=Q, observed_jumps=observed_jumps, shape=(Tn))

    B0 = Beta('B0', alpha = 1, beta = 1, shape=(K,M))
    B = Beta('B', alpha = 1, beta = 1, shape=(K,M))
    Xobs = Comorbidities('Xobs', S=S, B0=B0,B=B, shape=(K, Tn), observed = X)

    #Z = Beta('Z')
    #L = Beta('L')
    #O = Observations('O', X=X, Z=Z, L=L, shape=(D, T_n))

with model:
    step1 = Metropolis(vars=[pi,Q])
    step2 = ForwardS(vars=[S], X=X, observed_jumps=observed_jumps)
    step3 = Metropolis(vars=[B])