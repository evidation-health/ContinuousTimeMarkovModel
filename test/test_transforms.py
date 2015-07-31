from pymc3.tests.test_transforms import *
import ContinuousTimeMarkovModel.src.transforms as ctrans
from pymc3.tests.test_distributions import Domain

min_rate_matrix = np.zeros(shape=(5,5))
max_rate_matrix = np.zeros(shape=(5,5)) + np.inf
max_rate_matrix = np.fill_diagonal(max_rate_matrix, -np.inf)
rate_matrix_test = np.array([[-6, 2, 2, 1, 1], 
				[1, -4, 0, 1, 2],
				[1, 0, -4, 2, 1],
				[2, 1, 0, -3, 0],
				[1, 1, 1, 1, -4]])

RateMatrixDomain = Domain([min_rate_matrix, rate_matrix_test, max_rate_matrix])

def check_matrix_transform_identity(transform, domain):
    return check_transform_identity(transform, domain, t.dmatrix, test=rate_matrix_test)

def test_rate_matrix():
	check_matrix_transform_identity(ctrans.rate_matrix, RateMatrixDomain)