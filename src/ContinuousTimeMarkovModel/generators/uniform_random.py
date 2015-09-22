"""
Sample uniform random generator
"""

__author__ = "Luca Foschini"
__maintainer__ = "Luca Foschini"
__email__ = "luca@evidation.com"

def parse_args():
    parser = argparse.ArgumentParser(description='Generates a sequence of N random numbers between 1 and K')
    parser.add_argument("--length",  help = "Length of the sequence to generate", type = int, default = 10)
    parser.add_argument("--range",  help = "Numbers are between 1 and K", type = int, default = 10)
    args = parser.parse_args()
    return args.length, args.range


import argparse
import random

def main():
    n,k = parse_args()
    for i in range(n):
        print random.randint(1, k);
    
if __name__ == '__main__':
    main()

