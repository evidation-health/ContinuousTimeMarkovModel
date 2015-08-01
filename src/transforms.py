import theano.tensor as tt
from pymc3.distributions.transforms import Transform, ElemwiseTransform

__all__ = ['rate_matrix']

class ElemwiseTransformFlat(Transform):
    def jacobian_det(self, x):
        return tt.as_tensor_variable(0)

class RateMatrix(ElemwiseTransformFlat):
    name = "ratematrix"
    def __init__(self): 
        pass

    def symbolic_remove_diagonal(self, x):
        x = tt.as_tensor_variable(x)
        flat_x = x.flatten()
        indexes = tt.arange(flat_x.shape[0], dtype='int64')
        diagonal_modulo = indexes % (x.shape[0] + 1)
        off_diagonal_flat_x = flat_x[tt.neq(diagonal_modulo, 0).nonzero()]
        return off_diagonal_flat_x.reshape((x.shape[0], x.shape[1] - 1))


    def symbolic_add_diagonal(self, x):
        diagonal_values = -x.sum(axis=1)
        flat_x = x.flatten()
        result_length = flat_x.shape[0] + x.shape[0]
        indexes = tt.arange(result_length, dtype='int64')
        diagonal_modulo = indexes % (x.shape[0] + 1)
        result = tt.zeros((result_length,), dtype=x.dtype)
        result = tt.set_subtensor(result[tt.eq(diagonal_modulo, 0).nonzero()], diagonal_values)
        result = tt.set_subtensor(result[tt.neq(diagonal_modulo, 0).nonzero()], flat_x)
        return result.reshape((x.shape[0], x.shape[1] + 1))
    
    def backward(self, Q_raw_log):
        Q_raw = tt.exp(Q_raw_log)
        Q = self.symbolic_add_diagonal(Q_raw)
        return Q

    def forward(self, Q):
        Q_raw = self.symbolic_remove_diagonal(Q)
        Q_raw_log = tt.log(Q_raw)
        return Q_raw_log

rate_matrix = RateMatrix()