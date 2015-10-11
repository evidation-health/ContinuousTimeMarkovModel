from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import stick_breaking, logodds
from ContinuousTimeMarkovModel.transforms import rate_matrix, rate_matrix_one_way
from scipy import linalg

import theano

import ContinuousTimeMarkovModel.profilingUtil

class ForwardS(ArrayStepShared):
    """
    Use forward sampling (equation 10) to sample a realization of S_t, t=1,...,T_n
    given Q, B, and X constant.
    """
    def __init__(self, vars, nObs, T, N, observed_jumps, model=None):
    #DES Temp:
        self.logp = []
        self.nObs = nObs
        self.T = T
        self.N = N
        self.zeroIndices = np.roll(self.T.cumsum(),1)
        self.zeroIndices[0] = 0
        #self.max_obs = max_obs

        model = modelcontext(model)
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)

        super(ForwardS, self).__init__(vars, shared)
        
        self.observed_jumps = observed_jumps
        step_sizes = np.sort(np.unique(observed_jumps))
        self.step_sizes = step_sizes = step_sizes[step_sizes > 0]

        pi = stick_breaking.backward(self.shared['pi_stickbreaking'])
        lower = model.free_RVs[1].distribution.dist.lower
        upper = model.free_RVs[1].distribution.dist.upper
        Q = rate_matrix_one_way(lower, upper).backward(self.shared['Q_ratematrixoneway'])
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        X = self.shared['X']

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate them
        self.get_params = evaluate_symbolic_shared(pi, Q, B0, B, X)

    def compute_pSt_GIVEN_Stm1(self,Q,M):
# pS[tau,i,j] is probability of transitioning from S_i to S_j given time step of tau
        pS = np.zeros((len(self.step_sizes), M, M))

        for tau in self.step_sizes:
            pS_tau = linalg.expm(tau*Q)
            tau_ind = np.where(self.step_sizes == tau)[0][0]
            pS[tau_ind,:,:] = pS_tau

        return pS

    def computeBeta(self, Q, B, likelihoodOfS):

## beta_i(t) = p(X_{k,nt},X_{k,nt+1},...X_{k,nT} | S_nt, Theta)
## beta_i(t-1) = sum_{j=0,M} beta_j(t) * L(S_nt=j|pi,S_{nt-1}=i) * prod_{k=0,K} L(X_{nt,k}|S_nt=j,S_{nt-1}=i)
        M = self.M
        X = self.X
        T = self.T
        pS = self.pS = self.compute_pSt_GIVEN_Stm1(Q,M)
        observed_jumps = self.observed_jumps
        
        #beta = np.ones((M,#self.max_obs,#self.N))
        #beta = np.ones((M,self.nObs))
        beta = np.ones((self.nObs,M))

        #was_changed = X[1:,:] != X[:-1,:]
        
        for n in xrange(self.N):
            n0 = self.zeroIndices[n]
            for t in np.arange(T[n]-1, 0, -1):
                #import pdb; pdb.set_trace()
                tau_ind = np.where(self.step_sizes==observed_jumps[n0+t])[0][0]
##                #tau_ind = np.where(self.step_sizes==observed_jumps[n,t-1])[0][0]
##                was_changed = X[n0+t,:] != X[n0+t-1,:]
##                #was_changed = X[:,t,n] != X[:,t-1,n]
##                
##                #include the B prob. in the calculation only if the comorbidity
##                #was not on the previous step
##                not_on_yet = np.logical_not(X[n0+t-1,:].astype(np.bool))
##                #not_on_yet = np.logical_not(X[:,t-1,n].astype(np.bool))
##                pXt_GIVEN_St_St1 = np.prod(B[was_changed & not_on_yet,:], axis=0) * np.prod(1-B[(~was_changed)&not_on_yet,:], axis=0)
##                pXt_GIVEN_St_St1 = np.tile([pXt_GIVEN_St_St1], (M,1))
##                np.fill_diagonal(pXt_GIVEN_St_St1,np.float(not np.any(was_changed)))
##            #pXt_GIVEN_St_St1[i,j] is likelihood of X given S_t = i, S_t-1 = j
##                beta[:,n0+t-1] = np.sum(beta[:,n0+t]*(pS[tau_ind,:,:]*pXt_GIVEN_St_St1), axis=1)
# DES: With likelihood
                beta[n0+t-1,:] = np.sum(beta[n0+t,np.newaxis,:]*pS[tau_ind,:,:]*likelihoodOfS[n0+t,:,:], axis=1)
                #beta[:,t-1,n] = np.sum(beta[:,t,n]*(pS[tau_ind,:,:]*pXt_GIVEN_St_St1), axis=1)

        return beta
    
    def drawState(self, pS):
        cdf = np.cumsum(pS, axis=1)
        r = np.random.uniform(size=self.N) * cdf[:,-1]
        drawn_state = np.zeros(self.N)
        for n in range(self.N):
            drawn_state[n] = np.searchsorted(cdf[n,:], r[n])
        return drawn_state

    def drawStateSingle(self, pS):
        cdf = np.cumsum(pS)
        r = np.random.uniform() * cdf[-1]
        drawn_state = np.searchsorted(cdf, r)
#    #DES Super Temp:
        #drawn_state = np.argmax(pS)
        return drawn_state

    def compute_S0_GIVEN_X0(self,likelihoodOfS):
##        N = self.N
##        M = self.M
##        K = self.K
        pi = self.pi
##        B0 = self.B0
##        X = self.X

        #pS0 = np.zeros((N,M))
        pS0 = self.beta[self.zeroIndices,:]*pi[np.newaxis,:]*likelihoodOfS[self.zeroIndices,0,:]
##        for n in xrange(N):
##            on = X[self.zeroIndices[n],:] == 1
##            #on = X[:,0,n] == 1
##            off = np.invert(on)
##            likelihoodOfS0 = np.prod(1-B0[off,:],axis=0) * np.prod(B0[on,:],axis=0)
##            #pX_GIVEN_S0 = self.beta[:,0,n]
##            pS0[n,:] = self.beta[:,self.zeroIndices[n]] * pi * likelihoodOfS0
        return pS0

    def computeLikelihoodOfS(self,X,B,B0):
#likelihoodOfS[nt,j,i] is likelihood that S_nt=i given S_{nt-1}=j
#    prod_k p(X_{nt,k} | S_nt)

        likelihoodOfS = np.zeros((self.nObs,self.M,self.M),dtype=float)

        commorb_changed = np.insert(X[1:]-X[:-1],0,0,axis=0)
        commorb_changed[self.zeroIndices] = 0
        commorb_was_off = 1-np.insert(X[:-1],0,0,axis=0)
        commorb_was_off[self.zeroIndices] = 1

        #Upper triangular part (S increases)
        #upper[nt,S_t-1,S_t] = (1-commorb_was_off)*(1-commorb_changed) + commorb_was_off*(B[commorb_changed,:] + (1-commorb_was_off)(1-B[commorb_changed,:]
        #likelihoodOfS[:,:,:] = np.triu(np.tile((((1-commorb_was_off)*(1-commorb_changed))[:,:,np.newaxis] + commorb_was_off[:,:,np.newaxis]*(commorb_changed[:,:,np.newaxis]*B[np.newaxis,:,:]+(1-commorb_changed[:,:,np.newaxis])*(1-B[np.newaxis,:,:]))).prod(axis=1)[:,:,np.newaxis],(1,1,self.M)),k=1)
        probTemp = ((1-commorb_was_off)*(1-commorb_changed))[:,:,np.newaxis] + commorb_was_off[:,:,np.newaxis]*(commorb_changed[:,:,np.newaxis]*B[np.newaxis,:,:]+(1-commorb_changed[:,:,np.newaxis])*(1-B[np.newaxis,:,:]))
        #likelihoodOfS = np.tile(probTemp.prod(axis=1)[:,:,np.newaxis],(1,1,self.M))
        likelihoodOfS = np.tile(probTemp.prod(axis=1)[:,np.newaxis,:],(1,self.M,1))
        likelihoodOfS = np.triu(likelihoodOfS)
        #likelihoodOfS[:,:,:] = np.triu(np.tile((((1-commorb_was_off)*(1-commorb_changed))[:,:,np.newaxis] + commorb_was_off[:,:,np.newaxis]*(commorb_changed[:,:,np.newaxis]*np.roll(B,-1,axis=1)[np.newaxis,:,:]+(1-commorb_changed[:,:,np.newaxis])*(1-np.roll(B,-1,axis=1)[np.newaxis,:,:]))).prod(axis=1)[:,:,np.newaxis],(1,1,self.M)),k=1)
        #likelihoodOfS[:,:,:] = np.triu(np.tile((((1-commorb_was_off)*(1-commorb_changed))[:,:,np.newaxis] + commorb_was_off[:,:,np.newaxis]*(commorb_changed[:,:,np.newaxis]*np.roll(B,-1,axis=1)[np.newaxis,:,:]+(1-commorb_changed[:,:,np.newaxis])*(1-np.roll(B,-1,axis=1)[np.newaxis,:,:]))).prod(axis=1)[:,np.newaxis,:],(1,self.M,1)),k=1)
        #TODO: I do this a stupid lazy way using weird behaviour of tril on 3d arrays
        #Diagonal (S doesn't change)
        diagTemp = np.tile((1-commorb_changed).prod(axis=1)[:,np.newaxis,np.newaxis],(1,self.M,self.M))
        diagOnes = np.tile(np.identity(self.M)[np.newaxis,:,:],(self.nObs,1,1))
        #diagTemp = np.triu(np.tril(np.tile((1-commorb_changed).prod(axis=1)[:,np.newaxis,np.newaxis],(1,self.M,self.M))))
        likelihoodOfS[diagOnes.nonzero()] = diagTemp[diagOnes.nonzero()]
        #Set t=0
        likelihoodOfS[self.zeroIndices,:,:] = (X[self.zeroIndices][:,:,np.newaxis]*B0[np.newaxis,:,:] + (1-X[self.zeroIndices])[:,:,np.newaxis]*(1-B0)[np.newaxis,:,:]).prod(axis=1)[:,np.newaxis,:]
        
##        not_on_yet = np.logical_not(X[n0+t-1,:].astype(np.bool))
##        likelihoodOfS = np.prod(B[was_changed & not_on_yet,:], axis=0) * np.prod(1-B[(~was_changed)&not_on_yet,:], axis=0)
##        likelihoodOfS = np.tile([likelihoodOfS], (M,1))
        return likelihoodOfS
        

    def compute_pSt_GIVEN_St1(self,n0,t,Sprev,likelihoodOfS):
        i = Sprev.astype(np.int)

##        was_changed = self.X[n0+t+1,:] != self.X[n0+t,:]
##        not_on_yet = np.logical_not(self.X[n0+t].astype(np.bool))
##
##        pXt_GIVEN_St_St1 = np.prod(self.B[was_changed & not_on_yet,:], axis=0) * np.prod(1-self.B[(~was_changed) & not_on_yet,:], axis=0)
##        if np.any(was_changed):
##            pXt_GIVEN_St_St1[i] = 0.0
##        else:
##            pXt_GIVEN_St_St1[i] = 1.0

        #import pdb; pdb.set_trace()
        tau_ind = np.where(self.step_sizes==self.observed_jumps[n0+t+1])[0][0]
        
        #don't divide by beta_t it's just a constant anyway
        return self.beta[t+1+n0,:] * self.pS[tau_ind,i,:] * likelihoodOfS[t+1+n0,i,:]
# beta_{t+1,j} * pS_i->j * like_{t+1,i->j}
        #return self.beta[t+1+n0,:] * self.pS[tau_ind,i,:] * pXt_GIVEN_St_St1

    #@profilingUtil.timefunc
    def astep(self, q0):
        #X change points are the points in time where at least 
        #one comorbidity gets turned on. it's important to track
        #these because we have to make sure constrains on sampling
        #S are upheld. Namely S only goes up in time, and S changes
        #whenever there is an X change point. If we don't keep
        #track of how many change points are left we can't enforce
        #both of these constraints.
        self.pi, self.Q, self.B0, self.B, self.X=self.get_params()

        K = self.K = self.X.shape[1]
        M = self.M = self.Q.shape[0]
        T = self.T
        X = self.X
        #S = np.zeros((#self.N,#self.max_obs), dtype=np.int8) - 1
        S = np.zeros((self.nObs), dtype=np.int8) - 1

        #import pdb; pdb.set_trace()
        likelihoodOfS = self.computeLikelihoodOfS(X,self.B,self.B0)

        #calculate pS0(i) | X, pi, B0
        beta = self.beta = self.computeBeta(self.Q, self.B, likelihoodOfS)
        pS0_GIVEN_X0 = self.compute_S0_GIVEN_X0(likelihoodOfS)
        S[self.zeroIndices] = self.drawState(pS0_GIVEN_X0)
        #S[:,0] = self.drawState(pS0_GIVEN_X0)

        #calculate p(S_t=i | S_{t=1}=j, X, Q, B)
        #note: pS is probability of jump conditional on Q
        #whereas pS_ij is also conditional on everything else in the model
        #and is what we're looking for
        #B = self.B
        #observed_jumps = self.observed_jumps
        #pS = self.pS
        
    #DES Super Temp:
        #from pickle import load
        #S_start = load(open('../data/X_layer_100_patients/S.pkl', 'rb')) 
    #DES Temp:
        logp = 0.0
        for n in xrange(self.N):
            n0 = self.zeroIndices[n]
        #DES Super Temp:
            #S[n0] = S_start[n,0]
            logp += np.log(pS0_GIVEN_X0[n,S[n0]])
            #import pdb; pdb.set_trace()
            for t in xrange(0,T[n]-1):
                pSt_GIVEN_St1 = self.compute_pSt_GIVEN_St1(n0,t,S[n0+t],likelihoodOfS)
                #pSt_GIVEN_St1 = beta[:,t+1,n] * pS[tau_ind,i,:] * pXt_GIVEN_St_St1

                #make sure not to go backward or forward too far
                #pSt_GIVEN_St1[0:i] = 0.0
                
                S[n0+t+1] = self.drawStateSingle(pSt_GIVEN_St1)
                #import pdb; pdb.set_trace()
            #DES Super Temp:
                #S[n0+t+1] = S_start[n,t+1]
                logp += np.log(pSt_GIVEN_St1[S[n0+t+1]])
                #S[n,t+1] = self.drawStateSingle(pSt_GIVEN_St1)
            #import pdb; pdb.set_trace()
    #DES Temp:
        self.logp.append(logp)
        return S
        #return q0

def evaluate_symbolic_shared(pi, Q, B0, B, X):
    f = theano.function([], [pi, Q, B0, B, X])
    return f
