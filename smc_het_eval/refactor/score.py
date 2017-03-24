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
#import score
import helpers

import time
import resource
import os
import gzip





def scoreChallenge(challenge, predfiles, truthfiles, vcf, sample_fraction=1.0):
    
    
    
    #global err_msgs
    mem('START %s' % challenge)
    masks = helpers.makeMasks(vcf, sample_fraction) if sample_fraction != 1.0 else { 'samples' : None, 'truths' : None}

    if challengeMapping[challenge]['vcf_func']:
        nssms = verify(vcf, "input VCF", challengeMapping[challenge]['vcf_func'], sample_mask=masks['samples'])
        if nssms == None:
            err_msgs.append("Could not read input VCF. Exiting")
            return "NA"
    else:
        nssms = [[], []]

    mem('VERIFY VCF %s' % vcf)

    # total number of predicted lines?
    printInfo('total lines -> ' + str(nssms[0]))
    # total number of truth lines
    printInfo('total truth lines -> ' + str(nssms[1]))

    if len(predfiles) != len(challengeMapping[challenge]['val_funcs']) or len(truthfiles) != len(challengeMapping[challenge]['val_funcs']):
        err_msgs.append("Not enough input files for Challenge %s" % challenge)
        return "NA"

    tout = []
    pout = []
    tpout = []

    for predfile, truthfile, valfunc in zip(predfiles, truthfiles, challengeMapping[challenge]['val_funcs']):
        if helpers.is_gzip(truthfile) and challenge not in ['2B', '3B']:
            err_msgs.append('Incorrect format, must input a text file for challenge %s' % challenge)
            return "NA"
        targs = tout + nssms[1]

        vcfargs = tpout + nssms[0] + nssms[1]
        # an overlapping matrix is created for challenge 2A
        if challenge in ['2A', '3A']:
            if valfunc is om_validate2A:
                try:
                    vout, raw = verify2A(predfile, truthfile, "Combined truth and pred file for Challenge 2A", *vcfargs, filter_mut=nssms[2], mask=masks['truths'], subchallenge="3A")
                except SampleError as e:
                    raise e

                printInfo('OVERLAPPING MATRIX DIMENSIONS -> ', vout.shape)
                tpout.append(vout)
                if vout is None:
                    return "NA"
                if raw[0] != -1:
                    tpout.append(raw)

            elif valfunc is om_validate3A:
                try:
                    vtout = verify(truthfile, "truth file for Challenge %s" % (challenge), valfunc, vcfargs[0].shape[0], mask=masks['truths'])
                    vpout = verify(predfile, "pred file for Challenge %s" % (challenge), valfunc, vcfargs[0].shape[1], mask=masks['truths'])
                except SampleError as e:
                    raise e
                if vpout is None or vtout is None:
                    return "NA"                    
                tpout.append(vpout)
                tpout.append(vtout)

        elif challenge in ['2B']:
            try:
                vout = verify(truthfile, "truth file for Challenge %s" % (challenge), valfunc, *targs, mask=masks['truths'])
            except SampleError as e:
                raise e
   
            printInfo('TRUTH DIMENSIONS -> ', vout.shape)

            if WRITE_2B_FILES:
                np.savetxt('truth2B.txt.gz', vout)

            mem('VERIFY TRUTH %s' % truthfile)
            vout_with_pseudo_counts = add_pseudo_counts(vout)
            tout.append(vout_with_pseudo_counts)
            mem('APC TRUTH %s' % truthfile)
        else:
            tout.append(verify(truthfile, "truth file for Challenge %s" % (challenge), valfunc, *targs, mask=masks['truths']))
            mem('VERIFY TRUTH %s' % truthfile)
        
        if challenge in ['2B', '3B']:
            printInfo('FINAL TRUTH DIMENSIONS -> ', tout[-1].shape)

        # starts reading in predfile here
        if helpers.is_gzip(predfile) and challenge not in ['2B', '3B']:
            err_msgs.append('Incorrect format, must input a text file for challenge %s' % challenge)
            return "NA"

        # read in from pred file
        if challenge not in ['2A', '3A']:
            pargs = pout + nssms[0]

            pout.append(verify(predfile, "prediction file for Challenge %s" % (challenge), valfunc, *pargs, mask=masks['samples']))
            if pout[-1] is None:
                err_msgs.append("Unable to open prediction file")
                return "NA"
            mem('VERIFY PRED %s' % predfile)

        if challenge in ['2B', '3B']:
            printInfo('PRED DIMENSIONS -> ', pout[-1].shape)

        if challenge not in ['2A', '3A']:
            if tout[-1] is None or pout[-1] is None:
                return "NA"

    # if challenge in ['3A'] and WRITE_3B_FILES:
        # np.savetxt('pred3B.txt.gz', pout[-1])
        # np.savetxt('truth3B.txt.gz', tout[-1])
    if challenge not in ['2A', '3A']:
        printInfo('tout sum -> ', np.sum(tout[0]))
        printInfo('pout sum -> ', np.sum(pout[0]))

    if challengeMapping[challenge]['filter_func']:
        pout = [challengeMapping[challenge]['filter_func'](x, nssms[2]) for x in pout]
        printInfo('PRED DIMENSION(S) -> ', [p.shape for p in pout])

        mem('FILTER PRED(S)')

        printInfo('tout sum filtered -> ', np.sum(tout[0]))
        printInfo('pout sum filtered -> ', np.sum(pout[0]))

        if challenge in ['2B']:
            pout = [ add_pseudo_counts(*pout) ]
            mem('APC PRED')
            printInfo('FINAL PRED DIMENSION -> ', pout[-1].shape)

        # if challenge in ['3A']:
            # tout[0] = np.dot(tout[0], tout[0].T)
            # pout[0] = np.dot(pout[0], pout[0].T)
            # mem('3A DOT')

    if challenge in ['2A']:
        return challengeMapping[challenge]['score_func'](*tpout, add_pseudo=True, pseudo_counts=None)
    if challenge in ['3A']:
        return challengeMapping[challenge]['score_func'](*tpout)

    return challengeMapping[challenge]['score_func'](*(pout + tout))

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