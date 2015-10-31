# ContinuousTimeMarkovModel

Implements the model described in the paper

*Unsupervised Learning of Disease Progression Models* (X. Wang, D. Sontag, F. Wang), KDD'14

http://cs.nyu.edu/~dsontag/papers/WanSonWan_kdd14.pdf

# Instructions

1. Install the package with the command

```
pip install --process-dependency-links --trusted-host github.com -e git+https://github.com/evidation-health/ContinuousTimeMarkovModel.git#egg=ContinuousTimeMarkovModel
```

2. cd into the ContinuousTimeMarkovModel directory, something like /usr/local/lib/python2.7/site-packages/src/continuoustimemarkovmodel
    or try running 'import ContinuousTimeMarkovModel; print ContinuousTimeMarkovModel.__path__' in a python script

3. Add your data using the claimsToInputs.py script in the data file (may need slight modification for specific csv format). Example:

```
python claimsToInputs.py mydata.csv -c 32 -o myPickledInputs
```

    mydata.csv should contain rows of index, userid, datetime, claimsCode(i.e. ICD9) with a header pers_uniq_id, date_of_service, primary_diag_cd but the
    script can be easily modified to fit slightly different formatting

4. Run the runSontagModel.py script passing it your inputs (or try out the small_sample default) which in ipython would be run as:

```
run runSontagModel.py -n 1001 -t 100 -d '../data/myPickledInputs'
```
