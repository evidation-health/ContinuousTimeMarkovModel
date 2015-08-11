import numpy as np
from theano.tensor import as_tensor_variable
from ContinuousTimeMarkovModel.src.distributions import *
from pymc3 import Model, sample, Metropolis, Dirichlet, Binomial, Beta
from ContinuousTimeMarkovModel.src.forwardS import *

N = 100
M = 6
K = 10
D = 250
min_obs = 10
max_obs = 30

from pickle import load
T = load(open('../data/X_layer_100_patients/T.pkl', 'rb'))
obs_jumps = load(open('../data/X_layer_100_patients/obs_jumps.pkl', 'rb'))
X = load(open('../data/X_layer_100_patients/X_input.pkl', 'rb'))

model = Model()

with model:
    pi = Dirichlet('pi', a = as_tensor_variable([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), shape=M)
    Q = DiscreteObsMJP_unif_prior('Q', M=M, shape=(M,M))
    
    S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, N=N, observed_jumps=obs_jumps, T=T, shape=(N,max_obs))

    B0 = Beta('B0', alpha = 1, beta = 1, shape=(K,M))
    B = Beta('B', alpha = 1, beta = 1, shape=(K,M))

    Xobs = Comorbidities('Xobs', S=S, B0=B0,B=B, T=T, shape=(K, max_obs, N), observed = X)

    import pdb; pdb.set_trace()

    #Z = Beta('Z')
    #L = Beta('L')
    #O = Observations('O', X=X, Z=Z, L=L, shape=(D, T_n))

Q_raw_log = np.log(np.array([[0.631921, 0.0000001, 0.0000001, 0.0000001, 0.0000001], 
                             [0.0000001, 0.229485, 0.0000001, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.450538, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.206042, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.609582],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]]))
'''
Q_raw_log = np.log(np.array([[1, 0.0000001, 0.0000001, 0.0000001, 0.0000001], 
                             [0.0000001, 1, 0.0000001, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 1, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 1, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 1],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]]))
'''

start = {'Q_ratematrix': Q_raw_log}

with model:
    step1 = Metropolis(vars=[pi,Q])
    step2 = ForwardS(vars=[S], X=X, observed_jumps=obs_jumps)
    step3 = Metropolis(vars=[B0,B])
    trace = sample(300, [step1, step2, step3], start=start)

pi = trace[pi]
Q = trace[Q]
S = trace[S]
np.set_printoptions(2);np.set_printoptions(linewidth=160)
for i in range(300):
    print pi[i,:]
    print Q[i,:,:]
    print S[i,:]
    print "\n"