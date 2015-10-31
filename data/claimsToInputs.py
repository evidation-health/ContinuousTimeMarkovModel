import pandas as pd
import datetime
import numpy as np
from scipy.linalg import expm
import os
from pickle import dump
import re

import argparse
pd.options.mode.chained_assignment = None

#Set up inputs
parser = argparse.ArgumentParser(description='Convert claims into inputs for the Sontag disease progression model.')
parser.add_argument(action='store', default='new_sample/', type=str, dest = 'claimsfile',
                        help='claims csv file to read in')
parser.add_argument('-o','--outdir', action='store', default='new_sample/', type=str, dest = 'outdir',
                        help='directory to output data')
parser.add_argument('-p','--paramdir', action='store', default=None, type=str, dest = 'paramdir',
                        help='directory to grab parameter initializations from')
parser.add_argument('-a','--anchorsdir', action='store', default=None, type=str, dest = 'anchorsdir',
                        help='directory to grab anchors from if not specifying paramdir')
parser.add_argument('-t','--timeperiod', action='store', default=90, type=int, dest = 'timeperiod',
                        help='number of days per time period')
parser.add_argument('-c','--maxclaims', action='store', default=None, type=int, dest = 'maxclaims',
                        help='number of days per time period')
parser.add_argument('-s','--minsteps', action='store', default=3, type=int, dest = 'minsteps',
                        help='minimum number of active time periods')
parser.add_argument('--seed', action='store', default=111, type=int, dest = 'randomseed',
                        help='random seed for sampling')
parser.add_argument('--kcomorbid', action='store', default=4, type=int, dest = 'K',
                        help='specify K if not specifying paramdir')
parser.add_argument('--mstates', action='store', default=4, type=int, dest = 'M',
                        help='specify M if not specifying paramdir')
#parser.add_argument('-p','--profile', action='store_true', dest = 'profile',
#                        help='turns on theano profiler')

args = parser.parse_args()

def loadAnchors(dataDirectory):
    icd9Map = {}
    with open(dataDirectory+'/fid.txt') as mapFile:
        for i,icd9 in enumerate(mapFile):
            icd9Map[icd9.strip()] = i
    mapFile.close()
    #print icd9Map
    comorbidityNames = []
    anchors = []
    with open(dataDirectory+'/anchor_icd9.csv') as anchorFile:
        for i,line in enumerate(anchorFile):
            text = line.strip().split(',')
            comorbidityNames.append(text[0])
            comorbAnchors = []
            for codeStr in text[1:]:
                for key in icd9Map.keys():
                    l = re.search(codeStr,key)
                    if l is not None:
                        comorbAnchors.append(icd9Map[l.group(0)])
            anchors.append((i,comorbAnchors))
    anchorFile.close()
    return anchors,comorbidityNames

claimsDF = pd.read_csv(args.claimsfile,index_col=0,parse_dates='date_of_service')
claimsDF.date_of_service = claimsDF.date_of_service.astype(np.datetime64)

if args.maxclaims is not None:
    fid = claimsDF.primary_diag_cd.value_counts()[0:args.maxclaims].index.values
    claimsDF.primary_diag_cd = claimsDF.primary_diag_cd.apply(lambda x: x if x in fid else np.nan)
    claimsDF.dropna(inplace=True)
else:
    fid = claimsDF.primary_diag_cd.unique()
D = len(fid)

tstepClaims = []
for user in claimsDF.groupby('pers_uniq_id'):
    user[1].date_of_service = (user[1].date_of_service-user[1].date_of_service.min())/pd.Timedelta('1 days')
#    user[1].date_of_service.max()/
    nbins = np.ceil(user[1].date_of_service.max()/args.timeperiod)
    bins = np.arange(0,(nbins+1)*args.timeperiod,args.timeperiod)
    user[1].loc[:,'timeperiod'] = pd.cut(user[1].loc[:,'date_of_service'], bins, include_lowest=True,labels = range(int(nbins)))
    user[1].loc[:,'timeperiod'] = user[1].loc[:,'timeperiod'].dropna().astype(int)
    tstepClaims.append(user[1][['pers_uniq_id','timeperiod','primary_diag_cd']].drop_duplicates())

finalClaims = pd.concat(tstepClaims)
finalClaims = finalClaims.dropna()

fidDict = {}
for i,icd9 in enumerate(fid):
    fidDict[icd9] = i
finalClaims.loc[:,'primary_diag_cd'] = finalClaims.primary_diag_cd.apply(lambda x: fidDict[x])

finalClaims = finalClaims.groupby(['pers_uniq_id'],as_index=False).apply(lambda x: x if x.timeperiod.nunique()>=args.minsteps else None).reset_index(drop=True)

Dmax = finalClaims.groupby(['pers_uniq_id','timeperiod']).count().max()[0]

T = finalClaims.groupby(['pers_uniq_id']).timeperiod.nunique().values
nObs = T.sum()
N = len(T)
zeroIndices = np.roll(T.cumsum(),1)
zeroIndices[0] = 0

O = np.ones((nObs,Dmax),dtype=int)*-1
obs_jumps = np.zeros((nObs),dtype=int)

counter = 0
prevTime = 0
for group in finalClaims.groupby(['pers_uniq_id','timeperiod']):
    for i,val in enumerate(group[1].primary_diag_cd):
        O[counter,i]=val
    curTime = group[1].timeperiod.values[0]
    obs_jumps[counter] = curTime-prevTime
    prevTime = curTime
    counter += 1
obs_jumps[zeroIndices] = 0

if args.paramdir is not None:
    dataDirectory = args.paramdir
    Q = np.loadtxt(dataDirectory+'/Q.txt')
    pi = np.loadtxt(dataDirectory+'/pi.txt')
    B0 = np.loadtxt(dataDirectory+'/piB.txt')
    B = np.loadtxt(dataDirectory+'/B.txt')
    Z = np.loadtxt(dataDirectory+'/Z.txt')
    L = np.loadtxt(dataDirectory+'/L.txt')
    anchors,comorbidityNames = loadAnchors(dataDirectory)
    M = pi.shape[0]
    K = Z.shape[0]
else:
    #DES Random inputs
    K = args.K
    M = args.M
    ranSeed = args.randomseed
    np.random.seed(ranSeed)
    L = np.random.rand(D)*0.3
    np.random.seed(ranSeed+1)
    Z = np.random.rand(K,D)
    np.random.seed(ranSeed+2)
    B = np.random.rand(K,M)
    np.random.seed(ranSeed+3)
    B0 = np.random.rand(K,M)
    B0.sort(axis=1)
    np.random.seed(ranSeed+4)
    pi = np.random.rand(M)*(1-M*0.001)+0.001*M
    pi = pi/pi.sum()
    pi[::-1].sort()
    np.random.seed(ranSeed+5)
    Qvals = np.random.rand(M-1)
    Q = np.zeros((M,M))
    for i,val in enumerate(Qvals):
        Q[i,i+1] = val
        Q[i,i] = -val
    if args.anchorsdir is not None:
        anchors,comorbidityNames = loadAnchors(args.anchorsdir)
    else:
        anchors = []
        comorbidityNames = []

jumpInd = {}
transMat = []
for i,jump in enumerate(np.unique(obs_jumps)[1:]):
    jumpInd[jump] = i
    transMat.append(expm(jump*Q))

#Generate S from parameters
S = np.zeros(nObs,dtype=np.int32)
S[zeroIndices] = np.random.choice(np.arange(M),size=(N),p=pi)
for n in range(N):
    n0 = zeroIndices[n]
    for t in range(1,T[n]):
        S[n0+t] = np.random.choice(np.arange(M),p=transMat[jumpInd[obs_jumps[n0+t]]][S[n0+t-1]])

#Generate X from parameters
X = np.zeros((nObs,K))
X[zeroIndices] = np.random.binomial(n=1,p=B0[:,S[zeroIndices]].T)
for k in range(K):
    for n in range(N):
        n0 = zeroIndices[n]
        if X[n0,k] == 1:
            X[zeroIndices[n]:(zeroIndices[n]+T[n]),k] = 1
        else:
            changed = np.diff(S[zeroIndices[n]:(zeroIndices[n]+T[n])])
            for t in range(1,T[n]):
                if changed[t-1]==1 and np.random.rand()<B[k,S[n0+t]]:
                        X[(n0+t):(zeroIndices[n]+T[n]),k] = 1
                        break
X = X.astype(np.int8)

#Write pickled files
variables = [Q,pi,S,T,obs_jumps,B0,B,X,Z,L,O,anchors,comorbidityNames]
names = ['Q','pi','S','T','obs_jumps','B0','B','X','Z','L','O','anchors','comorbidityNames']
if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)
for var,name in zip(variables,names):
    outfile = open(args.outdir+'/'+name+'.pkl','wb')
    dump(var,outfile)
    outfile.close()
