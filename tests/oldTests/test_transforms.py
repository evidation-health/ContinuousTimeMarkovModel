from pymc3.tests.test_transforms import *
import ContinuousTimeMarkovModel.transforms as ctrans
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
	check_matrix_transform_identity(ctrans.rate_matrix(0,100), RateMatrixDomain)


min_rate_matrix = np.zeros(shape=(6,6))
max_rate_matrix = np.zeros(shape=(6,6)) + np.inf
max_rate_matrix = np.fill_diagonal(max_rate_matrix, -np.inf)
rate_matrix_test = np.array([
[-0.631921, 0.631921, 0.0000000, 0.0000000, 0.0000000, 0.0000000], 
[0.0000000, -0.229485, 0.229485, 0.0000000, 0.0000000, 0.0000000],
[0.0000000, 0.0000000, -0.450538, 0.450538, 0.0000000, 0.0000000],
[0.0000000, 0.0000000, 0.0000000, -0.206042,0.206042, 0.0000000],
[0.0000000, 0.0000000, 0.0000000, 0.0000000, -0.609582, 0.609582],
[0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]])
RateMatrixDomain = Domain([min_rate_matrix, rate_matrix_test, max_rate_matrix])

def check_matrix_transform_identity(transform, domain):
    return check_transform_identity(transform, domain, t.dmatrix, test=rate_matrix_test)

def test_rate_matrix_one_way():
	check_matrix_transform_identity(ctrans.rate_matrix_one_way(0,1), RateMatrixDomain)
