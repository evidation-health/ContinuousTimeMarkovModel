import theano.tensor as TT
from pymc3.distributions.transforms import Transform, ElemwiseTransform
from pymc3.distributions.transforms import interval, logodds
import numpy as np

#__all__ = ['ratematrix','ratematrixoneway','anchoredbeta']

class RateMatrix(Transform):
    name = "ratematrix"
    def __init__(self, lower, upper):
    	self.interval_transform = interval(lower,upper)

    def symbolic_remove_diagonal(self, x):
        x = TT.as_tensor_variable(x)
        flat_x = x.flatten()
        indexes = TT.arange(flat_x.shape[0], dtype='int64')
        diagonal_modulo = indexes % (x.shape[0] + 1)
        off_diagonal_flat_x = flat_x[TT.neq(diagonal_modulo, 0).nonzero()]
        return off_diagonal_flat_x.reshape((x.shape[0], x.shape[1] - 1))

    def symbolic_add_diagonal(self, x):
        diagonal_values = -x.sum(axis=1)
        flat_x = x.flatten()
        result_length = flat_x.shape[0] + x.shape[0]
        indexes = TT.arange(result_length, dtype='int64')
        diagonal_modulo = indexes % (x.shape[0] + 1)
        result = TT.zeros((result_length,), dtype=x.dtype)
        result = TT.set_subtensor(result[TT.eq(diagonal_modulo, 0).nonzero()], diagonal_values)
        result = TT.set_subtensor(result[TT.neq(diagonal_modulo, 0).nonzero()], flat_x)
        return result.reshape((x.shape[0], x.shape[1] + 1))
    
    def backward(self, Q_raw_log):
        Q_raw = self.interval_transform.backward(Q_raw_log)
        Q = self.symbolic_add_diagonal(Q_raw)
        return Q

    def forward(self, Q):
        Q_raw = self.symbolic_remove_diagonal(Q)
        Q_raw_log = self.interval_transform.forward(Q_raw)
        return Q_raw_log

    def jacobian_det(self, x):
    	return self.interval_transform.jacobian_det(x)

def rate_matrix(lower, upper):
	return RateMatrix(lower, upper)


class RateMatrixOneWay(Transform):
    name = "ratematrixoneway"
    def __init__(self, lower, upper):
        self.interval_transform = interval(lower,upper)

    #get the outward rates for the first n-1 states the last state has outward rate of 0
    def symbolic_full_matrix_to_rates(self, x):
        x = TT.as_tensor_variable(x)
        flat_x = x.flatten()
        indexes = TT.arange(flat_x.shape[0], dtype='int64')
        diagonal_modulo = indexes % (x.shape[0] + 1)
        outward_rates = flat_x[TT.eq(diagonal_modulo, 1).nonzero()]
        
        return outward_rates

    def symbolic_rates_to_full_matrix(self, x):
        zero = TT.as_tensor_variable(np.array([0.0]))
        rates = TT.concatenate([x, zero])
        result_length = (rates.shape[0])**2
        indexes = TT.arange(result_length, dtype='int64')
        diagonal_modulo = indexes % (rates.shape[0] + 1)
        #result = TT.zeros((result_length,), dtype=x.dtype)
        result = TT.zeros_like(indexes, dtype=rates.dtype)
        #this line is here because otherwise zeros_like doesn't actually return a vector with zeros
        result = TT.set_subtensor(result[TT.neq(diagonal_modulo, 0).nonzero()], 0.0)
        result = TT.set_subtensor(result[TT.eq(diagonal_modulo, 0).nonzero()], -rates)
        result = TT.set_subtensor(result[TT.eq(diagonal_modulo, 1).nonzero()], rates[:-1])
        return result.reshape((rates.shape[0], rates.shape[0]))
    
    def backward(self, Q_raw_log):
        Q_raw = self.interval_transform.backward(Q_raw_log)
        Q = self.symbolic_rates_to_full_matrix(Q_raw)
        return Q

    def forward(self, Q):
        Q_raw = self.symbolic_full_matrix_to_rates(Q)
        Q_raw_log = self.interval_transform.forward(Q_raw)
        return Q_raw_log

    def jacobian_det(self, x):
        return self.interval_transform.jacobian_det(x)

def rate_matrix_one_way(lower, upper):
    return RateMatrixOneWay(lower, upper)


class AnchoredBeta(Transform):
    name = "anchoredbeta"
    def __init__(self, mask, K, D):
        self.logodds_transform = logodds
        #anchors is list of tuples (D,K)
        #self.anchors = anchors
        self.K = K
        self.D = D

        #mask contains zeros for elements fixed at 10E-6
        self.mask = mask

    def anchored_to_full_matrix(self,Z_raw):
        #Z_raw is of length D*K-a*(K-1)
        Z_full = TT.ones_like(self.mask)*1E-6
        Z_full = TT.set_subtensor(Z_full[self.mask.nonzero()], Z_raw)
        return Z_full

    def full_matrix_to_anchored(self,Z_full):
        Z_full = TT.as_tensor_variable(Z_full)
        Z_raw = Z_full[self.mask.nonzero()]
        return Z_raw

    def backward(self, Z_raw_log):
        #Z_raw_log = TT.as_tensor_variable(Z_raw_log)
        Z_raw = self.logodds_transform.backward(Z_raw_log)
        Z = self.anchored_to_full_matrix(Z_raw)
        return Z

    def forward(self, Z):
        #import pdb; pdb.set_trace()
        #Z = TT.as_tensor_variable(Z)
        Z_raw = self.full_matrix_to_anchored(Z)
        Z_raw_log = self.logodds_transform.forward(Z_raw)
        return Z_raw_log

    def jacobian_det(self, x):
        return self.logodds_transform.jacobian_det(x)

def anchored_betas(mask,K,D,alpha,beta):
    return AnchoredBeta(mask,K,D)
