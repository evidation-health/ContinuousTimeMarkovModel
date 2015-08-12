import numpy as np
from theano.tensor import as_tensor_variable
from ContinuousTimeMarkovModel.src.distributions import *
from pymc3 import Model, sample, Metropolis, Dirichlet, Potential, Binomial, Beta
import theano.tensor as TT
from ContinuousTimeMarkovModel.src.forwardS import *

N = 100 # Number of patients
M = 6 # Number of hidden states
K = 10 # Number of comorbidities
D = 250 # Number of claims
min_obs = 10 # Minimum number of observed claims per patient
max_obs = 30 # Maximum number of observed claims per patient

# Load pre-generated data
from pickle import load
T = load(open('../data/X_layer_100_patients/T.pkl', 'rb'))
obs_jumps = load(open('../data/X_layer_100_patients/obs_jumps.pkl', 'rb'))
X = load(open('../data/X_layer_100_patients/X_input.pkl', 'rb'))

model = Model()
with model:
    pi = Dirichlet('pi', a = as_tensor_variable([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), shape=M)
    pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(pi) < .1, -np.inf, 0))

    Q = DiscreteObsMJP_unif_prior('Q', M=M, shape=(M,M))
    
    S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, N=N, observed_jumps=obs_jumps, T=T, shape=(N,max_obs))

    B0 = Beta('B0', alpha = 1, beta = 1, shape=(K,M))
    B = Beta('B', alpha = 1, beta = 1, shape=(K,M))

    Xobs = Comorbidities('Xobs', S=S, B0=B0,B=B, T=T, shape=(K, max_obs, N), observed = X)

    #Z = Beta('Z')
    #L = Beta('L')
    #O = Observations('O', X=X, Z=Z, L=L, shape=(D, T_n))

Q_raw_log = np.log(np.array([[0.631921, 0.0000001, 0.0000001, 0.0000001, 0.0000001], 
                             [0.0000001, 0.229485, 0.0000001, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.450538, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.206042, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.609582],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]]))

from scipy.special import logit
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
'''
Q_raw_log = np.log(np.array([[1, 0.0000001, 0.0000001, 0.0000001, 0.0000001], 
                             [0.0000001, 1, 0.0000001, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 1, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 1, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 1],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]]))
'''

start = {'Q_ratematrix': Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo}

with model:
    #import pdb; pdb.set_trace()
    step1 = Metropolis(vars=[pi,Q], scaling=0.1)
    step2 = ForwardS(vars=[S], N=N, T=T, max_obs=max_obs, X=X, observed_jumps=obs_jumps)
    step3 = Metropolis(vars=[B0])
    step4 = Metropolis(vars=[B])
    trace = sample(1001, [step1, step2, step3, step4], start=start, random_seed=1991)

pi = trace[pi]
Q = trace[Q]
S = trace[S]
S0 = S[:,:,0]
B0 = trace[B0]
np.set_printoptions(2);np.set_printoptions(linewidth=160)
'''
for i in range(1001):
    print "~~~",i ,"~~~"
    print pi[i,:]
    print "Bincount S0:", np.bincount(S0[i,:],minlength=6)
    print "\n"
'''
