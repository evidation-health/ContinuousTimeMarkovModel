import numpy as np
from theano.tensor import as_tensor_variable
from ContinuousTimeMarkovModel.distributions import *
from pymc3 import Model, sample, Metropolis, Dirichlet, Potential, Binomial, Beta, Slice, NUTS
import theano.tensor as TT
from ContinuousTimeMarkovModel.samplers.forwardS import *
from ContinuousTimeMarkovModel.samplers.forwardX import *

#import sys; sys.setrecursionlimit(50000)
#theano.config.compute_test_value = 'off'

N = 100 # Number of patients
M = 6 # Number of hidden states
K = 10 # Number of comorbidities
D = 721 # Number of claims
Dd = 80 # Maximum number of claims that can occur at once
min_obs = 10 # Minimum number of observed claims per patient
max_obs = 30 # Maximum number of observed claims per patient

# Load pre-generated data
from pickle import load


S_start = load(open('../data/X_layer_100_patients/S.pkl', 'rb'))
''' S_start[zeroIndices]
[3, 0, 0, 4, 1, 0, 3, 4, 4, 2, 2, 4, 5, 2, 2, 2, 2, 0, 2, 1, 1, 0, 1, 0, 3, 4, 0, 0, 3, 4, 1, 5, 0, 5, 3, 0, 3, 2, 4, 1, 4, 5, 4, 0, 1, 1, 1, 2, 3, 0, 1, 3, 0, 2, 4, 2, 4, 3, 5, 0, 4, 0, 1, 4, 4, 0, 4, 1, 3, 2, 2, 0, 0, 2, 4, 4, 4, 5, 0, 2, 2, 0, 1, 2, 2, 3, 5, 3, 3, 4, 2, 2, 4, 3, 5, 5, 3, 2, 0, 3]
'''
X_start = load(open('../data/X_layer_100_patients/X.pkl', 'rb'))
Z_start = load(open('../data/X_layer_100_patients/Z.pkl', 'rb'))
L_start = load(open('../data/X_layer_100_patients/L.pkl', 'rb'))
obs_jumps = load(open('../data/X_layer_100_patients/obs_jumps.pkl', 'rb'))
T = load(open('../data/X_layer_100_patients/T.pkl', 'rb'))
O = load(open('../data/X_layer_100_patients/O_input.pkl', 'rb'))

'''
T = load(open('../data/synthetic2000/T.pkl', 'rb'))
obs_jumps = load(open('../data/synthetic2000/obs_jumps.pkl', 'rb'))
S_start = load(open('../data/synthetic2000/S.pkl', 'rb'))
X_start = load(open('../data/synthetic2000/X.pkl', 'rb'))
Z_start = load(open('../data/synthetic2000/Z.pkl', 'rb'))
L_start = load(open('../data/synthetic2000/L.pkl', 'rb'))
O = load(open('../data/synthetic2000/O_input.pkl', 'rb'))


T = load(open('../data/small_model/data/T.pkl', 'rb'))
obs_jumps = load(open('../data/small_model/data/obs_jumps.pkl', 'rb'))
S_start = load(open('../data/small_model/data/S.pkl', 'rb'))
X_start = load(open('../data/small_model/data/X.pkl', 'rb'))
Z_start = load(open('../data/small_model/data/Z.pkl', 'rb'))
L_start = load(open('../data/small_model/data/L.pkl', 'rb'))
O = load(open('../data/small_model/data/O_input.pkl', 'rb'))
'''

#DES: nObs is total number of observations
nObs = T.sum()
#compress n and t indices
# S is (nObs) vector
S_start = np.concatenate([S_start[i,0:T[i]] for i in range(N)])
# add 0 to start for intial steps
obs_jumps = np.hstack([np.zeros((N,1),dtype='int8'),obs_jumps])
obs_jumps = np.concatenate([obs_jumps[i,0:T[i]] for i in range(N)])
# X is now (nObs,K)
X_start = np.concatenate([X_start[:,0:T[i],i].T for i in range(N)])
# O is now (nObs, Dd)
# TODO: implement this with sparse matrices
O = np.concatenate([O[:,0:T[i],i].T for i in range(N)])

#import pdb; pdb.set_trace()


model = Model()
with model:
   #Fails: #pi = Dirichlet('pi', a = as_tensor_variable([0.147026,0.102571,0.239819,0.188710,0.267137,0.054738]), shape=M, testval = np.ones(M)/float(M))
    pi = Dirichlet('pi', a = as_tensor_variable([0.147026,0.102571,0.239819,0.188710,0.267137,0.054738]), shape=M)
    pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(pi) < .001, -np.inf, 0))

    Q = DiscreteObsMJP_unif_prior('Q', M=M, lower=0.0, upper=1.0, shape=(M,M))
    
    #S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, nObs=nObs, observed_jumps=obs_jumps, T=T, shape=(nObs), testval=np.ones(nObs,dtype='int32'))
    S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, nObs=nObs, observed_jumps=obs_jumps, T=T, shape=(nObs))

    #B0 = Beta('B0', alpha = 1., beta = 1., shape=(K,M), testval=0.2*np.ones((K,M)))
    #B = Beta('B', alpha = 1., beta = 1., shape=(K,M), testval=0.2*np.ones((K,M)))
    B0 = Beta('B0', alpha = 1., beta = 1., shape=(K,M))
    B = Beta('B', alpha = 1., beta = 1., shape=(K,M))

    #X = Comorbidities('X', S=S, B0=B0,B=B, T=T, shape=(nObs, K), testval=np.ones((nObs,K),dtype='int8'))
    X = Comorbidities('X', S=S, B0=B0,B=B, T=T, shape=(nObs, K))

    #Z = Beta('Z', alpha = 0.1, beta = 1., shape=(K,D), testval=0.5*np.ones((K,D)))
    #L = Beta('L', alpha = 1., beta = 1., shape=D, testval=0.5*np.ones(D))
    Z = Beta('Z', alpha = 0.1, beta = 1., shape=(K,D))
    L = Beta('L', alpha = 1., beta = 1., shape=D)
    O_obs = Claims('O_obs', X=X, Z=Z, L=L, T=T, D=D, O_input=O, shape=(nObs,Dd), observed=O)
    #O_obs = Claims('O_obs', X=X, Z=Z, L=L, T=T, D=D, max_obs=max_obs, O_input=O, shape=(Dd,max_obs,N), observed=O)
#import pdb; pdb.set_trace()

import scipy.special
Q_raw_log = scipy.special.logit(np.array([0.631921, 0.229485, 0.450538, 0.206042, 0.609582]))

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

Z_lo = logit(Z_start)
L_lo = logit(L_start)
#L_lo = np.ones_like(L_start)*-4.0
'''
Q_raw_log = np.log(np.array([[1, 0.0000001, 0.0000001, 0.0000001, 0.0000001], 
                             [0.0000001, 1, 0.0000001, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 1, 0.0000001, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 1, 0.0000001],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 1],
                             [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]]))
'''

start = {'Q_ratematrixoneway': Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo, 'S':S_start, 'X':X_start, 'Z_logodds':Z_lo, 'L_logodds':L_lo}
#teststart = {'Q_ratematrixoneway': Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo, 'S':S_start, 'X':X_start, 'Z_logodds':Z_lo, 'L_logodds':L_lo, 'pi_stickbreaking':np.ones(M)/float(M)}
#start = {'Q_ratematrixoneway': Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo, 'S':S_start, 'X':X_start, 'Z_logodds':Z_lo, 'L_logodds':L_start}

with model:
    #import pdb; pdb.set_trace()

    steps = []
    steps.append(NUTS(vars=[pi]))
    #steps.append(NUTS(vars=[pi], scaling=np.ones(M-1)*0.058))
    #steps.append(Metropolis(vars=[pi], scaling=0.058, tune=False))
    steps.append(NUTS(vars=[Q],scaling=np.ones(M-1,dtype=float)*10.))
    #steps.append(Metropolis(vars=[Q], scaling=0.2, tune=False))
    steps.append(ForwardS(vars=[S], nObs=nObs, T=T, N=N, observed_jumps=obs_jumps))
    steps.append(NUTS(vars=[B0,B]))
    #steps.append(Metropolis(vars=[B0], scaling=0.2, tune=False))
    #steps.append(NUTS(vars=[B]))
    #steps.append(Metropolis(vars=[B], scaling=0.198, tune=False))
    steps.append(ForwardX(vars=[X], N=N, T=T, K=K, D=D,Dd=Dd, O=O, nObs=nObs))
    #steps.append(NUTS(vars=[Z], scaling=np.ones(K*D)))
    steps.append(Metropolis(vars=[Z], scaling=0.0132, tune=False))
    #steps.append(NUTS(vars=[L],scaling=np.ones(D)))
    steps.append(Metropolis(vars=[L],scaling=0.02, tune=False, ))

## 22 minutes per step with all NUTS set

    #import pdb; pdb.set_trace()
    #model.dlogp()
    trace = sample(1001, steps, start=start, random_seed=111,progressbar=True)
    #trace = sample(11, steps, start=start, random_seed=111,progressbar=True)
    #trace = sample(11, steps, start=start, random_seed=[111,112,113],progressbar=False,njobs=3)

pi = trace[pi]
Q = trace[Q]
S = trace[S]
#S0 = S[:,0]    #now pibar
B0 = trace[B0]
B = trace[B]
X = trace[X]
Z = trace[Z]
L = trace[L]
Sbin = np.vstack([np.bincount(S[i],minlength=6)/float(len(S[i])) for i in range(len(S))])
zeroIndices = np.roll(T.cumsum(),1)
zeroIndices[0] = 0
pibar = np.vstack([np.bincount(S[i][zeroIndices],minlength=M)/float(zeroIndices.shape[0]) for i in range(len(S))])
pibar = np.vstack([np.bincount(S_start[zeroIndices],minlength=M)/float(zeroIndices.shape[0]),pibar])
SEnd = np.vstack([np.bincount(S[i][zeroIndices-1],minlength=M)/float(zeroIndices.shape[0]) for i in range(len(S))])
SEnd = np.vstack([np.bincount(S_start[zeroIndices-1],minlength=M)/float(zeroIndices.shape[0]),SEnd])
#logp = steps[2].logp
#Xlogp = steps[4].logp
#XChanges = np.insert(1-(1-(X[:,1:]-X[:,:-1])).prod(axis=2),0,0,axis=1)
#XChanges.T[zeroIndices] = 0
#XChanges[XChanges.nonzero()] = XChanges[XChanges.nonzero()]/XChanges[XChanges.nonzero()]
#XChanges = XChanges.sum(axis=1)/float(N)
#logpTotal = [model.logp(trace[i]) for i in range(len(trace))]

#np.set_printoptions(2);np.set_printoptions(linewidth=160)
'''
for i in range(1001):
    print "~~~",i ,"~~~"
    print pi[i,:]
    print "Bincount S0:", np.bincount(S0[i,:],minlength=6)
    print "\n"
'''

#from pickle import dump
#with open('file.pkl','wb') as file:
#   dump(trace,file)
