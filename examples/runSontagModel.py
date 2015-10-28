import numpy as np
from theano.tensor import as_tensor_variable
from ContinuousTimeMarkovModel.distributions import *
from pymc3 import Model, sample, Metropolis, Dirichlet, Potential, Binomial, Beta, Slice, NUTS, Constant
import theano.tensor as TT
from ContinuousTimeMarkovModel.samplers.forwardS import *
from ContinuousTimeMarkovModel.samplers.forwardX import *
from theanoMod import *

import argparse

sampleVars = ['Q','pi','S','B0','B','X','Z','L']

#Set up inputs
parser = argparse.ArgumentParser(description='Run Sontag disease progression model.')
parser.add_argument('-d','--dir', action='store', default='../data/small_sample/', type=str, dest = 'datadir',
                        help='directory with pickled initial model parameters and observations')
parser.add_argument('-n','--sampleNum', action='store', default=1001, type=int, dest = 'sampleNum',
                        help='number of samples to run')
parser.add_argument('-t','--truncN', action='store', default=None, type=int, dest = 'newN',
                        help='number of people to truncate sample to')
parser.add_argument('-c','--const', action='store', default=[], nargs='+', type=str, choices=sampleVars, dest = 'constantVars',
                        help='list of variables to hold constant during sampling')
parser.add_argument('--seed', action='store', default=111, type=int, dest = 'random_seed',
                        help='random seed for sampling')
parser.add_argument('-p','--hide-progressbar', action='store_false', dest = 'progressbar',
                        help='hides progress bar in sample')
#parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),default=sys.stdout)
args = parser.parse_args()

#import sys; sys.setrecursionlimit(50000)
#theano.config.compute_test_value = 'off'

# Load pre-generated data
from pickle import load

#datadir = '../data/small_sample/'
datadir = args.datadir

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
# obs_jumps[o]: number of time periods that have passed between observation o and o-1
infile = open(datadir+'obs_jumps.pkl','rb')
obs_jumps = load(infile)
infile.close()
# T[n]: total number of observations of patient n
infile = open(datadir+'T.pkl','rb')
T = load(infile)
infile.close()
# O[o,:]: claims numbers present at observation o padded by -1's
infile = open(datadir+'O.pkl','rb')
O = load(infile)
infile.close()

infile = open(datadir+'anchors.pkl','rb')
anchors = load(infile)
infile.close()
#anchors = []

#import pdb; pdb.set_trace()

#Truncate to newN people
if args.newN is not None:
    T = T[:args.newN]
    nObs = T.sum()
    S_start = S_start[0:nObs]
    obs_jumps = obs_jumps[0:nObs]
    X_start = X_start[0:nObs]
    O = O[0:nObs]

nObs = S_start.shape[0] # Number of observations
N = T.shape[0] # Number of patients
M = pi_start.shape[0] # Number of hidden states
K = Z_start.shape[0] # Number of comorbidities
D = Z_start.shape[1] # Number of claims
Dmax = O.shape[1] # Maximum number of claims that can occur at once

mask = np.ones((K,D))
for anchor in anchors:
    for hold in anchor[1]:
        mask[:,hold] = 0
        mask[anchor[0],hold] = 1
Z_start = Z_start[mask.nonzero()]

model = Model()
with model:
# pi[m]: probability of starting in disease state m
    pi = Dirichlet('pi', a = as_tensor_variable(pi_start.copy()), shape=M)
    pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(pi) < .001, -np.inf, 0))

# exp(t*Q)[m,m']: probability of transitioning from disease state m to m' after a period of time t
    Q = DiscreteObsMJP_unif_prior('Q', M=M, lower=0.0, upper=1.0, shape=(M,M))
    
# S[o]: disease state (between 0 and M-1) at obeservation o
#       
    S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, nObs=nObs, observed_jumps=obs_jumps, T=T, shape=(nObs))

    B0 = Beta('B0', alpha = 1., beta = 1., shape=(K,M))
    B0_monotonicity_constraint = Potential('B0_monotonicity_constraint',TT.switch(TT.min(DES_diff(B0)) < 0., 100.0*TT.min(DES_diff(B0)), 0))

    B = Beta('B', alpha = 1., beta = 1., shape=(K,M))

    X = Comorbidities('X', S=S, B0=B0,B=B, T=T, shape=(nObs, K))

    #Z = Beta('Z', alpha = 0.1, beta = 1., shape=(K,D))
    Z = Beta_with_anchors('Z', anchors=anchors, K=K, D=D, alpha = 0.1, beta = 1., shape=(K,D))
    L = Beta('L', alpha = 1., beta = 1., shape=D)
    O_obs = Claims('O_obs', X=X, Z=Z, L=L, T=T, D=D, O_input=O, shape=(nObs,Dmax), observed=O)

from scipy.special import logit

#Transform the initial parameters
Q_raw = []
for i in range(Q_start.shape[0]-1):
    Q_raw.append(Q_start[i,i+1])
Q_raw_log = logit(np.asarray(Q_raw))
B_lo = logit(B_start)
B0_lo = logit(B0_start)
Z_lo = logit(Z_start)
L_lo = logit(L_start)

start = {'Q_ratematrixoneway': Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo, 'S':S_start, 'X':X_start, 'Z_anchoredbeta':Z_lo, 'L_logodds':L_lo}
#start = {'Q_ratematrixoneway': Q_raw_log, 'B_logodds':B_lo, 'B0_logodds':B0_lo, 'S':S_start, 'X':X_start, 'Z_logodds':Z_lo, 'L_logodds':L_lo}

with model:

    steps = []
    if 'pi' in args.constantVars:
        steps.append(Constant(vars=[pi]))
    else:
        steps.append(NUTS(vars=[pi]))
        #steps.append(Metropolis(vars=[pi], scaling=0.058, tune=False))
    if 'Q' in args.constantVars:
        steps.append(Constant(vars=[Q]))
    else:
        steps.append(NUTS(vars=[Q],scaling=np.ones(M-1,dtype=float)*10.)) #steps.append(Metropolis(vars=[Q], scaling=0.2, tune=False))
    if 'S' in args.constantVars:
        steps.append(Constant(vars=[S]))
    else:
        steps.append(ForwardS(vars=[S], nObs=nObs, T=T, N=N, observed_jumps=obs_jumps))
    if 'B0' in args.constantVars:
        steps.append(Constant(vars=[B0]))
        if 'B' in args.constantVars:
            steps.append(Constant(vars=[B]))
        else:
            steps.append(NUTS(vars=[B]))
    elif 'B' in args.constantVars:
        steps.append(NUTS(vars=[B0]))
        steps.append(Constant(vars=[B]))
    else:
        steps.append(NUTS(vars=[B0,B]))
        #steps.append(Metropolis(vars=[B0], scaling=0.2, tune=False))
        #steps.append(Metropolis(vars=[B], scaling=0.198, tune=False))
    if 'X' in args.constantVars:
        steps.append(Constant(vars=[X]))
    else:
        steps.append(ForwardX(vars=[X], N=N, T=T, K=K, D=D,Dd=Dmax, O=O, nObs=nObs))
    if 'Z' in args.constantVars:
        steps.append(Constant(vars=[Z]))
    else:
        #import pdb; pdb.set_trace()
        steps.append(NUTS(vars=[Z]))
        #steps.append(NUTS(vars=[Z], scaling=np.ones_like(Z_lo)))
        #steps.append(Metropolis(vars=[Z], scaling=0.0132, tune=False))
    if 'L' in args.constantVars:
        steps.append(Constant(vars=[L]))
    else:
        steps.append(NUTS(vars=[L],scaling=np.ones(D)))
        #steps.append(Metropolis(vars=[L],scaling=0.02, tune=False, ))

    trace = sample(args.sampleNum, steps, start=start, random_seed=args.random_seed ,progressbar=args.progressbar)
    #trace = sample(11, steps, start=start, random_seed=[111,112,113],progressbar=False,njobs=3)

pi = trace[pi]
Q = trace[Q]
S = trace[S]
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
XChanges = np.insert(1-(1-(X[:,1:]-X[:,:-1])).prod(axis=2),0,0,axis=1)
XChanges.T[zeroIndices] = 0
XChanges[XChanges.nonzero()] = XChanges[XChanges.nonzero()]/XChanges[XChanges.nonzero()]
XChanges = XChanges.sum(axis=1)/float(N)
logpTotal = [model.logp(trace[i]) for i in range(len(trace))]
#B0diff = np.array([np.diff(b) for b in B0])

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
