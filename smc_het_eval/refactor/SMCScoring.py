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
import helpers

import time
import resource
import os
import gzip

global err_msgs
err_msgs = []

INFO            = False
TIME            = False
MEM             = False
FINAL_MEM       = False
WRITE_2B_FILES  = False
WRITE_3B_FILES  = False

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


if __name__ == '__main__':
    start_time = time.time()
    #global err_msgs
    #err_msgs = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-config", default=None)
    parser.add_argument("--truth-config", default=None)
    parser.add_argument("-c", "--challenge", default=None)
    parser.add_argument("--predfiles", nargs="+")
    parser.add_argument("--truthfiles", nargs="*")
    parser.add_argument("--vcf")
    parser.add_argument("-o", "--outputfile")
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('--approx', nargs=2, type=float, metavar=('sample_fraction', 'iterations'), help='sample_fraction ex. [0.45, 0.8] | iterations ex. [4, 20, 100]')
    parser.add_argument('--approx_seed', nargs=1, type=int, default=[75])
    args = parser.parse_args()

    if args.pred_config is not None and args.truth_config is not None:
        with open(args.pred_config) as handle:
            pred_config = {}
            for line in handle:
                try:
                    v = json.loads(line)
                    if isinstance(v, dict):
                        pred_config = dict(pred_config, **v)
                except ValueError as e:
                    pass
        with open(args.truth_config) as handle:
            truth_config = {}
            for line in handle:
                try:
                    v = json.loads(line)
                    if isinstance(v, dict):
                        truth_config = dict(truth_config, **v)
                except ValueError as e:
                    pass
        out = {}
        print "pred", pred_config
        print "truth", truth_config
        for challenge in pred_config:
            if challenge in truth_config:
                predfile = pred_config[challenge]
                vcf = truth_config[challenge]['vcf']
                truthfiles = truth_config[challenge]['truth']
                if args.v:
                    res = verify.verifyChallenge(challenge, predfile, vcf)
                else:
                    res = score.scoreChallenge(challenge, predfile, truthfiles, vcf)
                out[challenge] = res
        with open(args.outputfile, "w") as handle:
            jtxt = json.dumps(out)
            handle.write(jtxt)
    else:
        # VERIFY
        if args.v:
            #res = adj_final(verifyChallenge(args.challenge, args.predfiles, args.vcf))
            res = verify.verifyChallenge(args.challenge, args.predfiles, args.vcf) 
        # APPROXIMATE
        elif args.approx and args.challenge in ['2A', '2B', '3A', '3B']:
            np.random.seed(args.approx_seed)
            sample_fraction = args.approx[0]
            iterations = int(np.floor(args.approx[1]))
            if sample_fraction >= 1.0 or sample_fraction <= 0.0:
                print('Sample Fraction value must be 0.0 < x < 1.0')
                sys.exit(1)
            results = []
            for i in xrange(iterations):
                print('Running Iteration %d with Sampling Fraction %.2f' % (i + 1, sample_fraction))
                resample = True
                while (resample):
                    try:
                        res = score.scoreChallenge(args.challenge, args.predfiles, args.truthfiles, args.vcf, sample_fraction)
                        print('Score[%d] -> %.5f' % (i + 1, res))
                        results.append(res)
                        resample = False
                    except SampleError as e:
                        # print(e.value)
                        print('resampling..')
                        resample = True
            mean = np.mean(results)
            median = np.median(results)
            std = np.std(results)
            print('')
            print('###################')
            print('## R E S U L T S ##')
            print('###################')
            print('Sampling Fraction\t%.2f' % sample_fraction)
            print('Sample Iterations\t%d' % iterations)
            print('Sampling Seed\t\t%d' % args.approx_seed[0])
            print('Scores\t\t\t%s' % str(results))
            print('Mean\t\t\t%.5f' % mean)
            print('Median\t\t\t%.5f' % median)
            print('Standard Deviation\t%.5f' % std)
            print('')
            res = adj_final(mean)
        # REAL SCORE
        else:
            print('Running Challenge %s' % args.challenge)
            res = score.scoreChallenge(args.challenge, args.predfiles, args.truthfiles, args.vcf)
            print res
            #print('SCORE -> %.16f' % res)

        with open(args.outputfile, "w") as handle:
            #jtxt = json.dumps( { args.challenge : res } )
            jtxt = str(res)
            handle.write(jtxt)

    mem('DONE')

    end_time = time.time() - start_time
    if TIME:
        print("[ T I M E ] %s seconds!" % round(end_time, 2))

    if len(err_msgs) > 0:
        for msg in err_msgs:
            print msg
        raise ValidationError("Errors encountered. If running in Galaxy see stdout for more info. The results of any successful evaluations are in the Job data.")

