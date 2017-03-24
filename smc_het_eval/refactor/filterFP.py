import math
import numpy as np
import itertools
import json
import argparse
import StringIO
import sys
import metric_behavior as mb
from functools import reduce
#from scoring_harness_optimized import *
from permutations import *
import gc
import traceback

import verify
#import filterFP 
import score
import helpers

import time
import resource
import os
import gzip



def filterFPs(x, mask):
    # EVERYTHING is done in memory
    #   1 - the elements at the indicies specified by mask are "picked" and assembled into a matrix
    #       that grows out of the upper-left corner of the original matrix (the original matrix will
    #       always be bigger than the eventual masked matrix)
    #   2 - the matrix is resized into a 1D array
    #   3 - the elements of the array are "shifted" so that they satisfy the [i,j] indices for
    #       a matrix of the new (masked) size
    #   4 - the array is "shrunk" to discard the difference between the old size and the new size
    #   5 - the array is resized into an actual nxn matrix
    # NOTE: matrix[np.ix_(mask, mask)] is considered advanced indexing and creates a copy, allocating new memory
    #       that's why we don't do it anymore
    if x.shape[0] == x.shape[1]:
        # 1 assemble masked matrix within the original matrix
        for i, m1 in enumerate(mask):
            for j, m2 in enumerate(mask):
                x[i, j] = x[m1, m2]

        old_n = x.shape[0]
        new_n = len(mask)

        # 2 resize into array
        x.resize((old_n**2), refcheck=False)

        # 3 shift elements
        for k in xrange(new_n):
            x[(k*new_n):((k+1)*new_n)] = x[(k*old_n):(k*old_n+new_n)]

        # 4 shrink array
        x.resize((new_n**2), refcheck=False)
        # 5 resize to array
        x.resize((new_n, new_n), refcheck=False)
        return x
    else:
        return x[mask, :]

challengeMapping = {
    '1A' : {
        'val_funcs' : [verify.validate1A],
        'score_func' : score.calculate1A,
        'vcf_func' : None,
        'filter_func' : None
    },
    '1B' : {
        'val_funcs' : [verify.validate1B],
        'score_func' : score.calculate1B,
        'vcf_func' : None,
        'filter_func' : None
    },
    '1C' : {
        'val_funcs' : [verify.validate1C],
        'score_func' : score.calculate1C,
        'vcf_func' : verify.parseVCF1C,
        'filter_func' : None
    },
    # According to Quaid, there is no need to filter false positves with the new method developed for scoring subchallenge 2A
    '2A' : {
        'val_funcs' : [verify.om_validate2A],
        'score_func' : score.om_calculate2A,
        'vcf_func' : verify.parseVCF2and3,
        'filter_func' : None
    },
    '2B' : {
        'val_funcs' : [verify.validate2B],
        'score_func' : score.calculate2,
        'vcf_func' : verify.parseVCF2and3,
        'filter_func' : filterFP.filterFPs
    },
    '3A' : {
        'val_funcs' : [verify.om_validate2A, verify.om_validate3A],
        'score_func' : score.calculate3A,
        'vcf_func' : verify.parseVCF2and3,
        'filter_func' : None
    },
    '3B' : {
        'val_funcs' : [verify.validate2B, verify.validate3B],
        'score_func' : score.calculate3Final,
        'vcf_func' : verify.parseVCF2and3,
        'filter_func' : filterFP.filterFPs
    },
}