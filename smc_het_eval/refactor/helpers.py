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
import filterFP 
import score
#import helpers

import time
import resource
import os
import gzip


def is_gzip(path):
    with open(path,'rb') as handle: 
        # test for gzip
        if (handle.read(2) == b'\x1f\x8b'):
            return True
    return False


def makeMasks(vcfFile, sample_fraction):
    # returns mask dictionary { 'all' : sample_mask, 'truth' : truth_mask }
    #   where sample_mask and truth_mask are both lists of indices
    # we need the truth_mask because the truth file ONLY contains truth lines,
    # whereas the vcf, pred files contain truth and false lines. thus, the truth
    # file line indicies do NOT match up with vcf and pred, so we need to make a
    # separate mask just for the truth file

    f = open(vcfFile)
    vcf = f.read()
    f.close()

    vcf = vcf.split('\n')
    vcf = [x for x in vcf if x != '' and x[0] != '#']
    vcf = [x[-4:] == "True" for x in vcf]

    # can use the combinadics method here..
    vcf_count = len(vcf)
    sample_size = int(np.floor(vcf_count * sample_fraction))
    sample_mask = set()
    for x in np.random.randint(0, vcf_count, sample_size):
        sample_mask.add(x)
    # useless counter to see how many "tries" it does to finish the set
    i = 0

    # finding the remaining random numbers can take a looooooooooooooong time
    # probably better to implement deterministic combinadics
    while len(sample_mask) < sample_size:
        missing = sample_size - len(sample_mask)
        for x in np.random.randint(0, vcf_count, missing):
            sample_mask.add(x)
        i += 1

    truth_mask = []
    truth_index = 0
    for i in xrange(len(vcf)):
        if vcf[i] and i in sample_mask:
            truth_mask.append(truth_index)
        if vcf[i]:
            truth_index += 1

    sample_mask = sorted(sample_mask)

    return { 'samples' : sample_mask, 'truths' : truth_mask }