# ContinuousTimeMarkovModel

Implements the model described in the paper

*Unsupervised Learning of Disease Progression Models* (X. Wang, D. Sontag, F. Wang), KDD'14

http://cs.nyu.edu/~dsontag/papers/WanSonWan_kdd14.pdf

# Instructions

1. cd in to src/cython
2. run the command 

```
python setup.py build_ext --inplace
```

3. This compiles the Cython code snippet. There should be a file now in the Cython directory called "compute_prod_other_k.so" if not, then it's in one of the other directories under cython (most likely cython/ContinuousTimeMarkovModel/src/cython) so move it in to the cython directory

4. Add the directory that you cloned the project in to i.e. the directory that contains "ContinuousTimeMarkovModel" in to the Python path. The python path is the environment variable called $PYTHONPATH 

5. Run the program (from ipython the command is)

```
run main.py
```