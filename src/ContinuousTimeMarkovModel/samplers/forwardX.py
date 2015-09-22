from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import logodds

from theano import function

#import ContinuousTimeMarkovModel.cython.forwardX_cython as cy
import ContinuousTimeMarkovModel.profilingUtil

class ForwardX(ArrayStepShared):
    """
    Use forward sampling (equation 10) to sample a realization of S_t, t=1,...,T_n
    given Q, B, and X constant.
    """
    def __init__(self, vars, N, T, K, D, Dd, O, nObs, model=None):
        self.N = N
        self.T = T
        self.K = K
        self.D = D
        self.Dd = Dd
        self.O = O
        self.nObs = nObs
        #self.max_obs = max_obs
        self.zeroIndices = np.roll(self.T.cumsum(),1)
        self.zeroIndices[0] = 0

        #self.pos_O_idx = np.zeros((D,max_obs,N), dtype=np.bool_)
        #for n in xrange(N):
        #    for t in xrange(self.T[n]):
        #        self.pos_O_idx[:,t,n] = np.in1d(np.arange(self.D), self.O[:,t,n])

        #self.OO = np.zeros((self.nObs,self.Dd),dtype=np.int)
        #self.OO = np.zeros((self.Dd,self.N,self.max_obs),dtype=np.int)
        self.negMask = np.zeros((self.nObs,D),dtype=np.int)
        #self.negMask = np.zeros((self.N,self.max_obs,D),dtype=np.int)
        for n in range(self.N):
            n0 = self.zeroIndices[n]
            for t in range(self.T[n]):
            #for t in range(self.max_obs):
                #self.OO[n0+t,:] = self.O[n0+t,:]
                self.negMask[n0+t,:] = 1-np.in1d(np.arange(self.D), self.O[n0+t,:]).astype(np.int)
        self.posMask = (self.O != -1).astype(np.int)

        #self.betaMask = np.zeros((max_obs,N,2))
        #for n in range(self.N):
        #    self.betaMask[:(T[n]-1),n,:] = 1

        model = modelcontext(model)
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)

        super(ForwardX, self).__init__(vars, shared)

        S = self.shared['S']
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        
        Z = logodds.backward(self.shared['Z_logodds'])
        L = logodds.backward(self.shared['L_logodds'])

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate them
        self.get_params = evaluate_symbolic_shared(S, B0, B, Z, L)

    def sampleState(self, pX):
        #pX_norm = pX/np.sum(pX, axis=0)
        #r = np.random.uniform(size=self.K)
        #drawn_state = np.greater_equal(r, pX_norm[0,:])
        
        pX_norm = (pX.T/np.sum(pX.T,axis=0)).T
        #pX_norm = (pX.T/np.sum(pX,axis=1))
        r = np.random.uniform(size=(self.nObs,self.K))
        #r = np.random.uniform(size=self.N)
        drawn_state = np.greater_equal(r, pX_norm[:,:,0])
        #drawn_state = np.greater_equal(r, pX_norm[0,:])
        return drawn_state.astype(np.int8)

    def computePsi(self, S, B):
        #Psi[nt,k,j,i] is the likelihood of x=i at time t given x=j at time t-1

        #prob. of getting the comorbidity once you already have it is one.
        #prob of not having the comorbidity once you've already had it is zero
        #Since state stays the same more often that it changes, by default we
        #set the prob. of staying in 0 given you're in 0 to 1.0. We then
        #change this to the appropriate B prob. for all instances where there
        #was a state change
        Psi = np.zeros((self.nObs,self.K,2,2),dtype=float)
        #Psi = np.zeros((self.K,self.N,self.max_obs,2,2))
        Psi[:,:,0,0] = 1.0
        Psi[:,:,1,1] = 1.0

        #use diff to see if state increased, if so prob. are based on B. Note
        #we have to insert at the beginning of S to get a diff that is the same
        #size as Psi in the time dimension
        state_change_idx = np.insert(S[1:]-S[:-1],0,0)
        state_change_idx[self.zeroIndices] = 0
        #import pdb; pdb.set_trace()
        Psi[state_change_idx.nonzero(),:,0,0] = (1-B[:,S[state_change_idx.nonzero()]]).T
        Psi[state_change_idx.nonzero(),:,0,1] = B[:,S[state_change_idx.nonzero()]].T
        #state_change_idx = np.diff(np.insert(S[:,:],0,1000,axis=1),axis=1) > 0
        #Psi[:,state_change_idx,0,0] = 1-B[:,S[state_change_idx]]
        #Psi[:,state_change_idx,0,1] = B[:,S[state_change_idx]]

        return Psi

    #@do_profile()
    def computeLikelihoodOfX(self,X,Z,L):
        #import pdb; pdb.set_trace()
        LikelihoodOfX = np.zeros((self.nObs,self.K,2))
        #Add extra column to get trashed by all the -1's, removed in following line
        O_on = np.zeros((self.nObs,self.D+1), dtype='int8')
        O_on[np.arange(self.nObs),self.O.T] = 1
        O_on = O_on[:,:-1]
        Z_on = Z.T[self.O.T]
        XZprod_on =  (1. - X.reshape(1,self.nObs,self.K)*(Z_on))
        otherKProduct_on = np.zeros((self.Dd,self.nObs,self.K))
        for k in xrange(self.K):
            otherKProduct_on[:,:,k] = XZprod_on[:,:,:k].prod(axis=2)*XZprod_on[:,:,k+1:].prod(axis=2)
        probGivenOnX0 = (1-L[self.O.T])[:,:,np.newaxis]*otherKProduct_on
        probGivenOnX1 = probGivenOnX0.copy()
        probGivenOnX0 = (1.-probGivenOnX0).T*self.posMask + (1-self.posMask)
        LikelihoodOfX[:,:,0] = (probGivenOnX0).prod(axis=2).T
        probGivenOnX1 = probGivenOnX1*(1.-Z_on)
        probGivenOnX1 = (1.-probGivenOnX1).T*self.posMask + (1-self.posMask)
        probGivenOffX1 = np.tile((1.-Z.T).reshape(self.D,1,self.K),(1,self.nObs,1))
        # Divide by Z_on values so assumes none are 0 at the moment!
        #TODO: Get this working for Z having 0 values
        totalZ = (1.-Z).prod(axis=1)
        Z_on_mask = (1.-Z_on).T*self.posMask + (1-self.posMask)
        probGivenOffX1 = totalZ/(Z_on_mask).prod(axis=2).T
        
        #probGivenOffX1 = probGivenOffX1.T*self.negMask + (1-self.negMask)
        LikelihoodOfX[:,:,1] = (probGivenOnX1).prod(axis=2).T*probGivenOffX1
#        XZprod =  (1. - X.reshape(self.nObs,1,self.K)*(Z.T).reshape(1,self.D,self.K))
#        otherKProduct = np.zeros((self.nObs,self.D,self.K))
#        for k in xrange(self.K):
#            otherKProduct[:,:,k] = XZprod[:,:,:k].prod(axis=2)*XZprod[:,:,k+1:].prod(axis=2)
#        O_on = O_on.astype(np.bool)
#        probGivenOnX0 = (1-L.reshape(1,self.D,1))*otherKProduct
# Can drop probGivenOffX0 since it can be divided out of probGivenOffX1
#        probGivenOnX1 = probGivenOnX0.copy()
#        probGivenOnX0[~O_on] = 0.
#        LikelihoodOfX[:,:,0] = (1.-probGivenOnX0).prod(axis=1)
#        probGivenOnX1 = probGivenOnX1*(1.-Z.T).reshape(1,self.D,self.K)
#        probGivenOffX1 = np.tile((1.-Z.T).reshape(1,self.D,self.K),(self.nObs,1,1))
#        probGivenOnX1[~O_on] = 0.
#        probGivenOffX1[O_on] = 1.
#        LikelihoodOfX[:,:,1] = (1.-probGivenOnX1).prod(axis=1)*probGivenOffX1.prod(axis=1)
#        #LikelihoodOfX[:,:,1] = (1.-probGivenOnX1).prod(axis=1)*probGivenOffX1.prod(axis=1)
        return LikelihoodOfX

    def computeLikelihoodOfXk(self, k, X, Z, L):
        LikelihoodOfXk = np.zeros((self.nObs,2))
        
        import pdb; pdb.set_trace()
        Z_pos = Z.T[self.OO.T]
        Z_neg = np.tile(Z[k,:],(self.nObs,1))
        XZ = X*Z_pos
        prod_other_k = np.prod((1-XZ[:,:,np.arange(self.K) != k]),axis=2)

        posTerms = (1-(1-L[self.OO.T])*prod_other_k)
        posTermsMasked = posTerms*self.posMask.T + (1-self.posMask.T)
        LikelihoodOfXk[:,0] = np.prod(posTermsMasked,axis=0)

        posTerms = 1-(1-L[self.OO.T])*(1-Z_pos[:,:,k])*prod_other_k
        posTermsMasked = posTerms*self.posMask.T + (1-self.posMask.T)
        negTerms = 1-Z_neg
        negTermsMasked = negTerms*self.negMask + (1-self.negMask)
        LikelihoodOfXk[:,1] = np.prod(negTermsMasked,axis=1)*np.prod(posTermsMasked,axis=0)
        
        return LikelihoodOfXk

    def computeBeta(self,Psi,LikelihoodOfX):
        beta = np.ones((self.nObs,self.K,2))
        #import pdb; pdb.set_trace()
        for n in range(self.N):
            n0 = self.zeroIndices[n]
            for t in range(self.T[n]-1,0,-1):
                beta[n0+t-1,:,:] = np.sum(beta[n0+t,:,np.newaxis,:]*Psi[n0+t,:,:,:]*LikelihoodOfX[n0+t,:,np.newaxis,:],axis=2)
                beta[n0+t-1,:,:] = (beta[n0+t-1,:,:].T/np.sum(beta[n0+t-1,:,:],axis=1)).T
        return beta

    def computePX(self,beta,B0,S,X,LikelihoodOfX,Psi):
        pX0 = beta[self.zeroIndices]*np.array([1-B0[:,S[self.zeroIndices]],B0[:,S[self.zeroIndices]]]).T*LikelihoodOfX[self.zeroIndices,:,:]
        pXt = np.insert(beta[1:,:,:]*Psi[np.tile(np.arange(1,self.nObs)[:,np.newaxis],(1,self.K)),np.tile(np.arange(self.K),(self.nObs-1,1)),X[:-1,:],:]*LikelihoodOfX[1:,:,:],0,-1,axis=0)
        #pXt = beta[:,:,:]*Psi[np.tile(np.arange(self.nObs)[:,np.newaxis],(1,self.K)),np.tile(np.arange(self.K),(self.nObs,1)),X[:,:],:]*LikelihoodOfX[:,:,:]
        #pXt = beta[:,:,:]*Psi[np.arange(self.nObs),:,X[:,:],:]*LikelihoodOfX[:,:,:]
        pXt[self.zeroIndices] = pX0
        return pXt

    #@profilingUtil.timefunc
    def astep(self,X):
        #timer = profilingUtil.timewith('forwardX step')
        S, B0, B, Z, L = self.get_params()
        X = np.reshape(X, (self.nObs,self.K)).astype(np.int8)
        #X = np.reshape(X, (self.K,self.max_obs,self.N)).astype(np.int8)

        Psi = self.computePsi(S,B)
        #timer.checkpoint('Computed Psi')
#        beta = np.ones((self.nObs,self.K,2))
        LikelihoodOfX = self.computeLikelihoodOfX(X,Z,L)
        #import pdb; pdb.set_trace()
        #timer.checkpoint('Computed LikelihoodX')
        beta = self.computeBeta(Psi,LikelihoodOfX)
#        for n in range(self.N):
#            n0 = self.zeroIndices[n]
#            for t in range(self.T[n]-1,0,-1):
#                beta[n0+t-1,:,:] = np.sum(beta[n0+t,:,:,np.newaxis]*Psi[n0+t,:,:,:]*LikelihoodOfX[n0+t,:,:,np.newaxis],axis=1)
#                beta[n0+t-1,:,:] = (beta[n0+t-1,:,:].T/np.sum(beta[n0+t-1,:,:],axis=1)).T
#        import pdb; pdb.set_trace()

#        pX0 = beta[self.zeroIndices]*np.array([1-B0[:,S[self.zeroIndices]],B0[:,S[self.zeroIndices]]]).T*LikelihoodOfX[self.zeroIndices+1,:,:]
#        pXt = beta[:,:,:]*Psi[np.tile(np.arange(self.nObs)[:,np.newaxis],(1,self.K)),np.tile(np.arange(self.K),(self.nObs,1)),X[:,:],:]*LikelihoodOfX[:,:,:]
#        #pXt = beta[:,:,:]*Psi[np.arange(self.nObs),:,X[:,:],:]*LikelihoodOfX[:,:,:]
#        pXt[self.zeroIndices] = pX0
        pXt = self.computePX(beta,B0,S,X,LikelihoodOfX,Psi)
        X[:,:] = self.sampleState(pXt)
#        for k in range(self.K):
#            LikelihoodOfXk = self.computeLikelihoodOfXk(k,X,Z,L)
#            timer.checkpoint('after computeLikelihoodOfXk')
#            
#            for n in range(self.N):
#                n0 = self.zeroIndices[n]
#                for t in range(self.T[n]-1,0,-1):
#                    beta[n0+t-1,:] = np.sum(beta[n0+t,np.newaxis,:]*Psi[n0+t,k,:,:]*LikelihoodOfXk[n0+t,np.newaxis,:],axis=1)
#                    beta[n0+t-1,:] = (beta[n0+t-1,:].T/np.sum(beta[n0+t-1,:])).T
##            for t in range(self.max_obs-1,0,-1):                    
##                beta[t-1,:,:] = np.sum(beta[t,:,np.newaxis,:]*Psi[k,:,t,:,:]*LikelihoodOfXk[:,t,np.newaxis,:],axis=2)
##                beta[t-1,:,:] = (beta[t-1,:,:].T/np.sum(beta[t-1,:,:], axis=1)).T
##                beta[t-1,:,:] = beta[t-1,:,:]*self.betaMask[t-1,:,:]+(1-self.betaMask[t-1,:,:])
#            timer.checkpoint('after nt loops')
#
#            #TODO: double check this zeroIndices+1 here
#            pX0 = beta[self.zeroIndices]*np.array([1-B0[k,S[self.zeroIndices]],B0[k,S[self.zeroIndices]]]).T*LikelihoodOfXk[self.zeroIndices+1,:]
#            #DES: What is the t variable doing here??
#            #pX0 = beta[0,:,:]*np.array([1-B0[k,S[:,0]],B0[k,S[:,0]]]).T*LikelihoodOfXk[:,t,:]
#            pXt = beta[:,:]*Psi[np.arange(self.nObs),k,X[:,k],:]*LikelihoodOfXk[:,:]
#            pXt[self.zeroIndices] = pX0
#            X[:,k] = self.sampleState(pXt)
#            #X[self.zeroIndices,k] = self.sampleState(pX0)
##            for t in range(self.max_obs-1):
##                Xtk = X[k,t,:]
##                pXt = beta[t+1,:,:]*Psi[k,np.arange(self.N),t+1,Xtk,:]*LikelihoodOfXk[:,t+1,:]
##                X[k,t+1,:] = self.sampleState(pXt)

        return X

    def astep_inplace(self,X):
        self.S, self.B0, self.B, self.Z, self.L = self.get_params()
        import pdb; pdb.set_trace()
        self.X = np.reshape(X, (self.K,self.max_obs,self.N)).astype(np.int8)

        #X_new = np.zeros((self.K,self.max_obs,self.N), dtype=np.int8) - 1

        for k in range(self.K):
            #note we keep Psi and pOt_GIVEN_Xt because they are used
            #in the computation of beta ADN then again in the sampling forward of X
            beta = np.ones((self.max_obs,self.N,2))
            Psi = np.zeros((self.max_obs,self.N,2,2))
            pOt_GIVEN_Xt = np.zeros((self.max_obs,self.N,2))
            for n in range(self.N):
                for t in np.arange(self.T[n]-1, -1, -1):
                    Xn = self.X[:,t,n]

                    #(A) Compute Psi which is the probability of jumping to state X_{t+1}=j
                    #given you're in state X_{t}=i and S_{t}=m. 
                    Psi[t,n,1,1] = 1.0

                    #if you did NOT change state the probability of getting a new
                    #comorbidity is zero. If you did change state the new state
                    #has comordbity onsets associated with it i.e. B
                    #if t == 0:
                    #    import pdb; pdb.set_trace()
                    if self.S[n,t] == self.S[n,t-1]:
                        Psi[t,n,0,0] = 1.0
                        Psi[t,n,0,1] = 0.0
                    else:
                        Psi[t,n,0,0] = 1-self.B[k,self.S[n,t]]
                        Psi[t,n,0,1] = self.B[k,self.S[n,t]]

                    #(B) Compute pOt_GIVEN_Xt i.e. the likelihood of X_t,k given Ot and all
                    #other X_t,l where l =/= k
                    pos_O_idx_n_t = self.pos_O_idx[:,t,n]
                    Z_pos = self.Z[:,pos_O_idx_n_t]
                    
                    #compute prod_other_k which is product term in eq. 13 over k' \neq k
                    #we compute it for all k, i.e. the kth row is the product of all k's
                    #except that k. we use Cython here
                    XZ_t = (Xn*Z_pos.T).T
                    prod_other_k = np.prod(1-XZ_t[np.arange(self.K) != k,:], axis=0)

                    pOt_GIVEN_Xt[t,:,0] = np.prod(1-(1-self.L[pos_O_idx_n_t])* \
                                 prod_other_k)
                    pOt_GIVEN_Xt[t,:,1] = np.prod(1-self.Z[k,np.logical_not(pos_O_idx_n_t)]) * \
                            np.prod(1 - (1-self.L[pos_O_idx_n_t])* \
                                (1-Z_pos[k,:])*prod_other_k)

                    #(C) Now actually set the beta (finally)
                    #we want this loop to go down to zero so we compute pOt_GIVEN_Xt[0]
                    #which we need in section (2) to sample the initial X, but obviously
                    #we won't want to go down to beta[-1] so we skip this part. Just
                    # a little trick to not have to repeat that code to get pOt_GIVEN_Xt[0]
                    if t < 1:
                        break
                    #beta[:,:,t] = beta[:,:,t] / np.sum(beta[:,:,t],axis=0)
                    #pOt_GIVEN_Xt[:,:,t] = pOt_GIVEN_Xt[:,:,t] / np.sum(pOt_GIVEN_Xt[:,:,t], axis=0)
                    beta[t-1,:,:] = np.sum(beta[t,n,:] * Psi[t,n,:,:] * \
                        pOt_GIVEN_Xt[t,n,:], axis=1)
                    beta[t-1,:,:] = beta[t-1,n,:] / np.sum(beta[t-1,n,:],axis=0)

                #(2)sample X_new
                #(A) Sample starting comorbidities
                pX0_GIVEN_O0 = beta[0,n,:] * \
                    np.array([1-self.B0[k,self.S[n,0]],self.B0[k,self.S[n,0]]]) * \
                    pOt_GIVEN_Xt[0,n,:]
                self.X[k,0,n] = self.sampleState(pX0_GIVEN_O0)


                #import pdb; pdb.set_trace()
                #(B) Sample rest of X's through time
                for t in xrange(0,self.T[n]-1):
                    X_i = self.X[k,t,n]
                    pXt_next = beta[t+1,n,:] * Psi[t+1,n,X_i,:] * pOt_GIVEN_Xt[t,n,:]
                    self.X[k,t+1,n] = self.sampleState(pXt_next)

        return self.X


    def astep_old(self, X):
        self.S, self.B0, self.B, self.Z, self.L = self.get_params()
        self.K = self.B.shape[0]
        self.X = np.reshape(X, (self.K,self.max_obs,self.N)).astype(np.int8)

        X_new = np.zeros((self.K,self.max_obs,self.N), dtype=np.int8) - 1

        #note we keep Psi and pOt_GIVEN_Xt because they are used
        #in the computation of beta ADN then again in the sampling forward of X
        beta = np.ones((2,self.K,self.max_obs))
        Psi = np.zeros((2,2,self.K,self.max_obs))
        pOt_GIVEN_Xt = np.zeros((2,self.K,self.max_obs))
        for n in xrange(self.N):
            Xn = self.X[:,:,n]
            pos_O_idx_n = self.pos_O_idx[:,:,n]
            
            #(1)compute beta a.k.a. the backwards variables a.k.a. 
            #likelihood of X given the entire time series of observations
            for t in np.arange(1):
            #for t in np.arange(self.T[n]-1, -1, -1):
                #(A) Compute Psi which is the probability of jumping to state X_{t+1}=j
                #given you're in state X_{t}=i and S_{t}=m. Note the probability of
                #getting the comorbidity once you already have it is one. The prob.
                #of not having the comorbidity once you've already had it is zero
                Psi[1,1,:,t] = 1.0

                #if you did NOT change state the probability of getting a new
                #comorbidity is zero. If you did change state the new state
                #has comordbity onsets associated with it i.e. B
                if self.S[n,t] == self.S[n,t-1]:
                    Psi[0,0,:,t] = 1.0
                    Psi[0,1,:,t] = 0.0
                else:
                    Psi[0,0,:,t] = 1-self.B[:,self.S[n,t]]
                    Psi[0,1,:,t] = self.B[:,self.S[n,t]]

                #(B) Compute pOt_GIVEN_Xt i.e. the likelihood of X_t,k given Ot and all
                #other X_t,l where l =/= k
                pos_O_idx_n_t = pos_O_idx_n[:,t]
                Z_pos = self.Z[:,pos_O_idx_n_t]
                
                #compute prod_other_k which is product term in eq. 13 over k' \neq k
                #we compute it for all k, i.e. the kth row is the product of all k's
                #except that k. we use Cython here
                XZ_t = (Xn[:,t]*Z_pos.T).T
                n_pos_O = np.sum(pos_O_idx_n_t)
                prod_other_k = np.zeros((self.K, n_pos_O))

                prod_other_k = compute_prod_other_k.compute(XZ_t, n_pos_O, self.K)

                ####
                self.L[pos_O_idx_n_t] = 0.01
                ####

                pOt_GIVEN_Xt[0,:,t] = np.prod(1-(1-self.L[pos_O_idx_n_t])* \
                             prod_other_k, axis=1)
                pOt_GIVEN_Xt[1,:,t] = np.prod(1-self.Z[:,np.logical_not(pos_O_idx_n_t)], axis=1) * \
                        np.prod(1 - (1-self.L[pos_O_idx_n_t])* \
                            (1-Z_pos)*prod_other_k, axis=1)
                '''
                prob = pOt_GIVEN_Xt[:,:,t] / np.sum(pOt_GIVEN_Xt[:,:,t])
                r = np.random.uniform(size=self.K)
                drawn_state = np.greater_equal(r, prob[0,:])
                Xn[:,t] = drawn_state
                '''
                #if n==5 and t==0:
                #    print Xn[:,t], '\n'
                '''
                for k in range(self.K):
                    X_on = np.copy(Xn[:,t])
                    X_on[k] = 1
                    XZ_on = (X_on*self.Z.T).T
                    
                    X_off = np.copy(Xn[:,t])
                    X_off[k] = 0
                    XZ_off = (X_off*self.Z.T).T

                    pOt_GIVEN_Xt[0,k,t] = np.prod(1-(1-self.L[pos_O_idx_n_t])*np.prod(1-XZ_off[:,pos_O_idx_n_t],axis=0))*\
                        np.prod((1-self.L[np.logical_not(pos_O_idx_n_t)])*np.prod(1-XZ_off[:,np.logical_not(pos_O_idx_n_t)],axis=0))
                    pOt_GIVEN_Xt[1,k,t] = np.prod(1-(1-self.L[pos_O_idx_n_t])*np.prod(1-XZ_on[:,pos_O_idx_n_t],axis=0))*np.prod((1-self.L[np.logical_not(pos_O_idx_n_t)])*np.prod(1-XZ_on[:,np.logical_not(pos_O_idx_n_t)],axis=0))

                    prob = pOt_GIVEN_Xt[:,k,t] / np.sum(pOt_GIVEN_Xt[:,k,t])
                    r = np.random.uniform()
                    drawn_state = np.greater_equal(r, prob[0])
                    Xn[k,t] = drawn_state
                '''


                if n==5 and t==0:
                    import pdb; pdb.set_trace()
                    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nX0:\n',Xn[:,t],'\nZ_pos:\n',Z_pos,'\nXZ_t:\n',XZ_t,'\nprod_other_k:\n',prod_other_k,'\n1-L:\n',(1-self.L[pos_O_idx_n_t]),'\nunmult_off\n',1-(1-self.L[pos_O_idx_n_t])* prod_other_k,'\npOFF:\n',pOt_GIVEN_Xt[0,:,t],'\nZ_factor\n',np.prod(1-self.Z[:,np.logical_not(pos_O_idx_n_t)], axis=1),'\nunmult_ON:\n', (1 - (1-self.L[pos_O_idx_n_t])*(1-Z_pos)*prod_other_k),'\npON\n',np.prod(1-self.Z[:,np.logical_not(pos_O_idx_n_t)], axis=1)*np.prod(1 - (1-self.L[pos_O_idx_n_t])*(1-Z_pos)*prod_other_k, axis=1),'\nnorm:\n',pOt_GIVEN_Xt[:,:,0] / np.sum(pOt_GIVEN_Xt[:,:,0],axis=0), '\nbeta\n', beta, '\n'
                    #print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n:L', self.L[pos_O_idx_n_t], '\nX0', Xn[:,t], '\n'
                    #print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!nX0', Xn[:,t], '\n'
                    #print pOt_GIVEN_Xt[:,:,0] / np.sum(pOt_GIVEN_Xt[:,:,0],axis=0), '\nZ[0:10,0]:', Z_pos[0:10,0], '\nL[0:10]', self.L[pos_O_idx_n_t][0:10],'\n', 'Z_factor:', np.prod(1-self.Z[:,np.logical_not(pos_O_idx_n_t)], axis=1),'\n1-L:', (1-self.L[pos_O_idx_n_t])[0:10], '\n'
                    #print '\nother_k: ', prod_other_k
                    #print 'Z_factor:', np.prod(1-self.Z[:,np.logical_not(pos_O_idx_n_t)], axis=1)
                    #print 'pO', pOt_GIVEN_Xt[:,:,0]

                #(C) Now actually set the beta (finally)
                #we want this loop to go down to zero so we compute pOt_GIVEN_Xt[0]
                #which we need in section (2) to sample the initial X, but obviously
                #we won't want to go down to beta[-1] so we skip this part. Just
                # a little trick to not have to repeat that code to get pOt_GIVEN_Xt[0]
                if t < 1:
                    break
                #beta[:,:,t] = beta[:,:,t] / np.sum(beta[:,:,t],axis=0)
                #pOt_GIVEN_Xt[:,:,t] = pOt_GIVEN_Xt[:,:,t] / np.sum(pOt_GIVEN_Xt[:,:,t], axis=0)
                beta[:,:,t-1] = np.sum(beta[:,:,t] * Psi[:,:,:,t] * \
                    pOt_GIVEN_Xt[:,:,t], axis=1)
                beta[:,:,t-1] = beta[:,:,t-1] / np.sum(beta[:,:,t-1],axis=0)

            #(2)sample X_new
            #(A) Sample starting comorbidities
            pX0_GIVEN_O0 = beta[:,:,0] * \
                np.array([1-self.B0[:,self.S[n,0]],self.B0[:,self.S[n,0]]]) * \
                pOt_GIVEN_Xt[:,:,0]
            X_new[:,0,n] = self.sampleState(pX0_GIVEN_O0)

            #if n==5:
                #print '\nbeta', beta[:,:,0]
                #print 'pS', np.array([self.B0[:,self.S[n,0]],1-self.B0[:,self.S[n,0]]])
            #    print 'pO', pOt_GIVEN_Xt[:,:,0]
                #print 'p', pX0_GIVEN_O0
                #print 'X_new', X_new[:,0,n]

            #(B) Sample rest of X's through time
            for t in xrange(0,self.T[n]-1):
                Xnt = X_new[:,t,n]
                pXt_next = beta[:,:,t+1] * Psi[Xnt,:,0,t+1].T * pOt_GIVEN_Xt[:,:,t+1]
                X_new[:,t+1,n] = self.sampleState(pXt_next)

        return X_new

def evaluate_symbolic_shared(S,B0,B,Z,L):
    f = function([], [S,B0,B,Z,L])
    return f
