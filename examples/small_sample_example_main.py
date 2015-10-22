import numpy as np
from theano.tensor import as_tensor_variable
from ContinuousTimeMarkovModel.distributions import *
from pymc3 import Model, sample, Metropolis, Dirichlet, Potential, Binomial, Beta, Slice, NUTS
import theano.tensor as TT
from ContinuousTimeMarkovModel.samplers.forwardS import *
from ContinuousTimeMarkovModel.samplers.forwardX import *

#import sys; sys.setrecursionlimit(50000)
#theano.config.compute_test_value = 'off'

# Load pre-generated data
from pickle import load

datadir = '../data/small_sample/'

infile = open(datadir+'pi.pkl','rb')
pi_start = load(infile)
infile.close()
infile = open(datadir+'Q.pkl','rb')
Q_start = load(infile)
infile.close()
infile = open(datadir+'S.pkl','rb')
S_start = load(infile)
infile.close()
infile = open(datadir+'B.pkl','rb')
B_start = load(infile)
infile.close()
infile = open(datadir+'B0.pkl','rb')
B0_start = load(infile)
infile.close()
infile = open(datadir+'X.pkl','rb')
X_start = load(infile)
infile.close()
infile = open(datadir+'Z.pkl','rb')
Z_start = load(infile)
infile.close()
infile = open(datadir+'L.pkl','rb')
L_start = load(infile)
infile.close()
infile = open(datadir+'obs_jumps.pkl','rb')
obs_jumps = load(infile)
infile.close()
infile = open(datadir+'T.pkl','rb')
T = load(infile)
infile.close()
infile = open(datadir+'O.pkl','rb')
O = load(infile)
infile.close()

#Cut down to 100 people
newN = 100
T = T[:newN]
nObs = T.sum()
S_start = S_start[0:nObs]
obs_jumps = obs_jumps[0:nObs]
X_start = X_start[0:nObs]
O = O[0:nObs]

nObs = S_start.shape[0]
N = T.shape[0] # Number of patients
M = pi_start.shape[0] # Number of hidden states
K = Z_start.shape[0] # Number of comorbidities
D = Z_start.shape[1] # Number of claims
Dd = 16 # Maximum number of claims that can occur at once

#import pdb; pdb.set_trace()

model = Model()
with model:
   #Fails: #pi = Dirichlet('pi', a = as_tensor_variable([0.147026,0.102571,0.239819,0.188710,0.267137,0.054738]), shape=M, testval = np.ones(M)/float(M))
    pi = Dirichlet('pi', a = as_tensor_variable(pi_start.copy()), shape=M)
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

from scipy.special import logit

Q_raw = []
for i in range(Q_start.shape[0]-1):
    Q_raw.append(Q_start[i,i+1])
Q_raw_log = logit(np.asarray(Q_raw))
B_lo = logit(B_start)
B0_lo = logit(B0_start)
Z_lo = logit(Z_start)
L_lo = logit(L_start)

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
    steps.append(NUTS(vars=[Z], scaling=np.ones(K*D)))
    #steps.append(Metropolis(vars=[Z], scaling=0.0132, tune=False))
    steps.append(NUTS(vars=[L],scaling=np.ones(D)))
    #steps.append(Metropolis(vars=[L],scaling=0.02, tune=False, ))

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
logp = steps[2].logp
Xlogp = steps[4].logp
XChanges = np.insert(1-(1-(X[:,1:]-X[:,:-1])).prod(axis=2),0,0,axis=1)
XChanges.T[zeroIndices] = 0
XChanges[XChanges.nonzero()] = XChanges[XChanges.nonzero()]/XChanges[XChanges.nonzero()]
XChanges = XChanges.sum(axis=1)/float(N)
logpTotal = [model.logp(trace[i]) for i in range(len(trace))]

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
