import theano.tensor as TT
from pymc3.distributions.transforms import Transform, ElemwiseTransform
from pymc3.distributions.transforms import interval

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