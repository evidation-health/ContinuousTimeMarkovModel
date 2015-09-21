import numpy as np
from pymc3 import Model
from ContinuousTimeMarkovModel.distributions import *
from theano import function

#test discrete
M = 6
Tn = 20

S_test = np.array([ 0,  1,  1,  2,  2,  5,
			5,  5,  5,  5,  5,  5,  5,
        	5,  5,  5,  5,  5,  5,  5])
observed_jumps = np.array([2, 4, 2, 2, 4, 2, 2,
			2, 1, 2, 2, 2,
			4, 1, 1, 1, 1, 4, 2])

with Model() as model:
    Q = DiscreteObsMJP_unif_prior('Q', M=M, shape=(M,M))
    S = DiscreteObsMJP('S', Q=Q, observed_jumps=observed_jumps, shape=(Tn))

Q_true_transformed = np.log(np.array([[0.631921,0.000001,0.000001,0.000001,0.000001],
    [0.000001,0.229485,0.000001,0.000001,0.000001],
    [0.000001,0.000001,0.450538,0.000001,0.000001],
    [0.000001,0.000001,0.000001,0.206042,0.000001],
    [0.000001,0.000001,0.000001,0.000001,0.609582],
    [0.000001,0.000001,0.000001,0.000001,0.00001]]))

Q_perturbed_transformed = np.log(np.array([[0.000001,0.000001,0.000001,0.000001,0.000001],
    [0.000001,0.229485,0.000001,0.000001,0.000001],
    [0.000001,0.000001,0.450538,0.000001,0.000001],
    [0.000001,0.000001,0.000001,0.206042,0.000001],
    [0.000001,0.000001,0.000001,0.000001,0.609582],
    [0.000001,0.000001,0.000001,0.000001,0.00001]]))

logp = model.fastlogp

pt_true = {'Q_ratematrix': Q_true_transformed,
	  'S': S_test}

pt_perturbed = {'Q_ratematrix': Q_perturbed_transformed,
	  'S': S_test}

#this number was double checked with R (see ratematrix_logp.R)
np.testing.assert_almost_equal(logp(pt_true), -5.554507, decimal=6,
	err_msg="logp of input Q is incorrect")

#the true value of Q should have the highest likelihood of any Q
assert logp(pt_true) > logp(pt_perturbed), "true value of Q should have a higher likelihood"
