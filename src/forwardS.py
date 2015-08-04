from pymc3.core import *
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.theanof import make_shared_replacements
from pymc3.distributions.transforms import logodds
from .transforms import rate_matrix

import theano

class ForwardS(ArrayStepShared):
    """
    Use forward sampling (equation 10) to sample a realization of S_t, t=1,...,T_n
    given Q, B, and X constant.
    """
    def __init__(self, vars, X, model=None):
        model = modelcontext(model)
        vars = inputvars(vars)
        shared = make_shared_replacements(vars, model)

        super(ForwardS, self).__init__(vars, shared)
        
        B0 = logodds.backward(self.shared['B0_logodds'])
        B = logodds.backward(self.shared['B_logodds'])
        Q = rate_matrix.backward(self.shared['Q_ratematrix'])

        #at this point parameters are still symbolic so we
        #must create get_params function to actually evaluate
        #them
        self.get_params = evaluate_symbolic_shared(B0, B, Q)

    def computeBeta(Q_raw, B):
        
        
        return 1
        
    def astep(self, S_current):
        
        #paramaters are now usable
        B0,B,Q=self.get_params()
        
        S_next = S_current
        
        return S_next

def evaluate_symbolic_shared(B0, B, Q):
    f = theano.function([], [B0, B, Q])
    return f