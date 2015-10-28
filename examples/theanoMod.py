import numpy as np
import numpy

import theano
from theano.tensor import basic
from theano.tensor import nlinalg  # noqa
from theano import gof, scalar
from theano.gradient import DisconnectedType
tensor = basic


class DES_DiffOp(theano.Op):
    # See function diff for docstring 
        
    __props__ = ("n", "axis")
                                                                                     
    def __init__(self, n=1, axis=-1):                                                
        self.n = n                                                                   
        self.axis = axis
        # numpy return a view in that case.
        # TODO, make an optimization that remove this op in this case.               
        if n == 0:
            self.view_map = {0: [0]}                                                 
                                                                                     
    def make_node(self, x):
        x = basic.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.diff(x, n=self.n, axis=self.axis)                                  
    
    def grad(self, inputs, outputs_gradients):                                       
        inputs = inputs[0]
        
#        if inputs.ndim != 1:
#            raise NotImplementedError("Grad is not implemented for inputs with"
#                                      "number of dimension other than 1.")
#    
        z = outputs_gradients[0]                                                     
                                                                                     
        def _grad_helper(z):
            #import pdb; pdb.set_trace()
            zeros = z.zeros_like().astype(theano.config.floatX)
            zeros = zeros.sum(axis=self.axis,keepdims=True)
            pre = basic.concatenate([zeros, z],axis=self.axis)
            app = basic.concatenate([z, zeros],axis=self.axis)
            return pre - app

        for k in range(self.n):                                                      
            z = _grad_helper(z)
        return [z]

def DES_diff(x, n=1, axis=-1):
    """Calculate the n-th order discrete difference along given axis.

    The first order difference is given by out[i] = a[i + 1] - a[i]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff.

    Parameters
    ----------
    x
        Input tensor variable.

    n
        The number of times values are differenced, default is 1.

    axis
        The axis along which the difference is taken, default is the last axis.

    .. versionadded:: 0.6

    """
    return DES_DiffOp(n=n, axis=axis)(x)
