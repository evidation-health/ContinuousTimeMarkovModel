import pandas as pd

def read_file(full_filename):

    return pd.read_csv(full_filename, 
                     names = ['PatientID', 'Timestamp', 'ICD9', 'FirstOnsetTime'],  
                     compression = 'gzip')
