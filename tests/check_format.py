"""
Checks that datafile format is compliant with description
"""

__author__ = "Luca Foschini"
__maintainer__ = "Luca Foschini"
__email__ = "luca@evidation.com"

import argparse
import sys
import datetime

sys.path.append('../lib')

from etl.data_io import read_file


DATA_DIR = '../data/'
COPD_CODES = ['491', '491.0', '491.1', '491.20', '491.21', '491.22', '491.8', 
              '491.9', '492', '492.0', '492.8', '493.20', '493.21', '493.22', 
              '496', '518.1', '518.2']

def parse_args():
    parser = argparse.ArgumentParser(description='Check format of data file is compliant with description')
    parser.add_argument("--filename",  help = "Filename, expects csv.gz", type = str, default = 'COPD.csv.gz')
    args = parser.parse_args()
    return args.filename

def print_summary(df):
    print
    print "Number of records: ", len(df)
    print "------ SNIPPET --------"
    print df.head()
    print "------ STATS --------"
    print "Number of unique patients: ", len(df.PatientID.unique())
    print " -- Observation Timestamp "
    print df.Timestamp.describe()
    print " -- First Onset Time "
    print df.FirstOnsetTime.describe()
    print " -- Unique ICD9 Codes found: ", len(df.ICD9.unique())
    print " -- Unique ICD9 COPD codes found: ", len(set(df.ICD9.unique().tolist()).intersection(
          set(COPD_CODES)))


def main():
    filename = parse_args()

    now = datetime.datetime.now()
    print >> sys.stderr, "Reading file %s ..." % filename
    df = read_file("%s/%s" % (DATA_DIR, filename))
    print >> sys.stderr, "Done. Time elapsed: %.2f seconds" % \
        (float((datetime.datetime.now() - now).microseconds) / 10**6)

    print_summary(df)
    
if __name__ == '__main__':
    main()
