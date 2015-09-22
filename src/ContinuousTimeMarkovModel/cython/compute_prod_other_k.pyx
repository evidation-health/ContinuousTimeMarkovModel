import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def compute(np.ndarray[DTYPE_t, ndim=2] XZ_t, int n_pos_O, int K):
	cdef np.ndarray[DTYPE_t, ndim=2] prod_other_k = np.zeros([K,n_pos_O], dtype=DTYPE)
	cdef DTYPE_t prod_k
	for o in range(n_pos_O):
		for k in range(K):
			prod_k = 1.0
			for kk in range(K):
				if kk != k:
					prod_k *= 1-XZ_t[kk,o]
			prod_other_k[k,o] = prod_k

	return prod_other_k