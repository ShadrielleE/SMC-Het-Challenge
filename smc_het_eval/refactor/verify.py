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

#import verify
import filterFP 
import score
import helpers

import time
import resource
import os
import gzip

class ValidationError(Exception):
    def __init__(self, value):
        self.value = value
        print('VALIDATION ERROR: %s' % value)
    def __str__(self):
        return repr(self.value)



#### VALIDATE CHALLENGE 1 ###########################################################
def validate1A(data, mask=None):
    data = data.split('\n')
    data = filter(None, data)
    if len(data) < 1:
        raise ValidationError("Input file contains zero lines")
    if len(data) > 1:
        raise ValidationError("Input file contains more than one line")
    data = data[0].strip()
    try:
        numeric = float(data)
    except ValueError:
        raise ValidationError("Data could not be converted to float: %s" % data)
    if math.isinf(numeric):
        raise ValidationError("Non-finite Cellularity")
    if math.isnan(numeric):
        raise ValidationError("Cellularity is NaN")
    if numeric < 0:
        raise ValidationError("Cellularity was < 0: %f" % numeric)
    if numeric > 1:
        raise ValidationError("Cellularity was > 1: %f" % numeric)

    return numeric



def validate1B(data, mask=None):
    data = data.split('\n')
    data = filter(None, data)
    if len(data) != 1:
        if len(data) == 0:
            raise ValidationError("Input file contains zero lines")
        else:
            raise ValidationError("Input file contains more than one line")
    data = data[0].strip()
    try:
        numeric = int(data)
    except ValueError:
        raise ValidationError("Data could not be converted to int: %s" % data)
    if numeric < 1:
        raise ValidationError("Number of lineages was less than 1: %d" % numeric)
    if numeric > 20:
        raise ValidationError("Number of lineages was greater than 20: %d" % numeric)
    return numeric

def validate1C(data, nssms, mask=None):
    data = data.split('\n')
    data = filter(None, data)
    data = [x.strip() for x in data]
    if len(data) < 1:
        raise ValidationError("Number of lines is less than 1")
    elif len(data) > 10:
        raise ValidationError("Number of lines is greater than 10")

    data2 = [x.split('\t') for x in data]

    for i in range(len(data)):
        if len(data2[i]) != 3:
            raise ValidationError("Number of tab separated columns in line %d is not 3" % (i+1))
        try:
            id = int(data2[i][0])
            if id != i+1:
                raise ValidationError("Cluster ID in line %d is not %d" % (i+1, i+1))
        except ValueError:
            raise ValidationError("Cluster ID in line %d can not be cast as an integer: %s" % (i+1, data2[i][0]))
        try:
            nm = int(data2[i][1])
            if nm < 1:
                raise ValidationError("Number of mutations in line %d is less than 1." % (i+1))
        except ValueError:
            raise ValidationError("Number of mutations in line %d can not be cast as an integer: %s" % (i+1, data2[i][1]))
        try:
            if data2[i][2] == 'NA':
                data2[i][2] = 0
            cf = float(data2[i][2])
            if math.isinf(cf):
                raise ValidationError("Cellular Frequency for cluster %d is non-finite" % (i+1))
            if math.isnan(cf):
                raise ValidationError("Cellular Frequency for cluster %d is NaN" % (i+1))
            if cf < 0:
                raise ValidationError("Cellular Frequency for cluster %d is negative: %f" % (i+1, cf))
            if cf > 1:
                raise ValidationError("Cellular Frequency for cluster %d is > 1: %f" % (i+1, cf))

        except ValueError:
            raise ValidationError("Cellular Frequency for cluster %d can not be cast as a float: %s" % (i+1, data2[i][2]))
    reported_nssms = sum([int(x[1]) for x in data2])
    if reported_nssms != nssms:
        raise ValidationError("Total number of reported mutations is %d. Should be %d" % (reported_nssms, nssms))
    return zip([int(x[1]) for x in data2], [float(x[2]) for x in data2])


#### VALIDATE CHALLENGE 2 ###########################################################
def om_validate2A (pred_data, truth_data, nssms_x, nssms_y, filter_mut=None, mask=None, subchallenge="2A"):
    '''
    Creates overlapping matrix for SubChallenge 2 and 3
    :param pred_data: inputed data from prediction file
    :param truth_data: inputed data from truth file
    :param nssms_x: number of mutations prediction file (specified by vcf)
    :param filter_mut: list of mutations to filter in prediction file
    :param mask: mask applied
    :subchallenge: subchallenge scored
    :return: overlapping matrix and (for subchallenge 3) a list which specifies the cluster of each mutation
    '''
    pred_data = pred_data.split('\n')
    pred_data = filter(None, pred_data)
    pred_data = [x for i, x in enumerate(pred_data) if i in mask] if mask else pred_data
     
    if len(pred_data) != nssms_x:
        raise ValidationError("Prediction file contains a different number of lines than the specification file. Input: %s lines. Specification: %s lines" % (len(pred_data), nssms_x))
    pred_cluster_entries = set()

    for i in xrange(len(pred_data)):
        try:
            pred_data[i] = int(pred_data[i])
            pred_cluster_entries.add(pred_data[i])
        except ValueError:
            raise ValidationError("Cluster ID in line %d (ssm %s) can not be cast to an int", (i+1, pred_data[i][0]))

    num_pred_clusters = max(pred_cluster_entries)

    truth_data = truth_data.split('\n')
    truth_data = filter(None, truth_data)
    truth_data = [x for i, x in enumerate(truth_data) if i in mask] if mask else truth_data

    if len(truth_data) != nssms_y:
        raise ValidationError("Truth file contains a different number of lines than the specification file. Input: %s lines. Specification: %s lines" % (len(truth_data), nssms_y))

    truth_cluster_entries = set()
    for i in xrange(len(truth_data)):
        try:
            truth_data[i] = int(truth_data[i])
            truth_cluster_entries.add(truth_data[i])
        except ValueError:
            raise ValidationError("Cluster ID in line %d (ssm %s) can not be cast to an int", (i+1, truth_data[i][0]))

    num_truth_clusters = max(truth_cluster_entries)

    om = np.zeros((num_truth_clusters, num_pred_clusters), dtype=int)

    # print len(filter_mut)
    new_pred_data = []
    if filter_mut is not None:
        for i in range(len(pred_data)):     
            if i in filter_mut:
                new_pred_data.append(pred_data[i])
    else:
        new_pred_data = pred_data

    for i in range(len(new_pred_data)):
        # print(new_pred_data[i]), 
        om[truth_data[i]-1, new_pred_data[i]-1] += 1

    if subchallenge is "3A":
        return om, truth_data

    return om

def validate2B(filename, nssms, mask=None):
  
    ccm = np.zeros((nssms, nssms))
    try:
        if os.path.exists(filename):
            
            if helpers.is_gzip(filename):
                gzipfile = gzip.open(str(filename), 'r')
            else:
                gzipfile = open(str(filename), 'r')
            ccm_i = 0
            for i, line in enumerate(gzipfile):
                if mask == None:
                    ccm[i, ] = np.fromstring(line, sep='\t')
                    #print str(i) + " T" #debug
                elif i in mask:
                    #print "mask" #debug
                    #print str(ccm_i) + " S" #debug
                    matrix_line = line.split(' ')
                    ccm[ccm_i, ] = [x for i, x in enumerate(matrix_line) if i in mask]
                    ccm_i += 1
            gzipfile.close()
            #print "done load " + filename #debug
        else:
            # TODO - optimize with line by line
            data = StringIO.StringIO(filename)
            truth_ccm = np.loadtxt(data, ndmin=2)
            ccm[:nssms, :nssms] = truth_ccm
    except ValueError as e:
        raise ValidationError("Entry in co-clustering matrix could not be cast as a float. Error message: %s" % e.message)

    if ccm.shape != (nssms, nssms):
        raise ValidationError("Shape of co-clustering matrix %s is wrong.  Should be %s" % (str(ccm.shape), str((nssms, nssms))))
    if not np.allclose(ccm.diagonal(), np.ones((nssms))):
        raise ValidationError("Diagonal entries of co-clustering matrix not 1")
    if np.any(np.isnan(ccm)):
        raise ValidationError("Co-clustering matrix contains NaNs")
    if np.any(np.isinf(ccm)):
        raise ValidationError("Co-clustering matrix contains non-finite entries")
    if np.any(ccm > 1):
        raise ValidationError("Co-clustering matrix contains entries greater than 1")
    if np.any(ccm < 0):
        raise ValidationError("Co-clustering matrix contains entries less than 0")
    if not isSymmetric(ccm):
        raise ValidationError("Co-clustering matrix is not symmetric")
    return ccm

def isSymmetric(x):
    '''
    Checks if a matrix is symmetric.
    Better than doing np.allclose(x.T, x) because..
        - it does it in memory without making a new x.T matrix
        - fails fast if not symmetric
    '''
    symmetricity = False
    if (x.shape[0] == x.shape[1]):
        symmetricity = True
        for i in xrange(x.shape[0]):
            symmetricity = symmetricity and np.allclose(x[i, :], x[:, i])
            if (not symmetricity):
                break
    return symmetricity

##### VALIDATE CHALLANGE 3 ##########################################################

def om_validate3A(data_3A, predK, mask=None):
    """Constructs a matrix that describes the relationship between the clusters
    :param data_3A: inputted data for subchallenge 3A
    :param predK: number of clusters
    :param mask: mask used
    :return:
    """
    # read in the data
    data_3A = data_3A.split('\n')
    data_3A = filter(None, data_3A)
    if len(data_3A) != predK:
        raise ValidationError("Input file contains a different number of lines (%d) than expected (%d)")
    data_3A = [x.split('\t') for x in data_3A]
    for i in range(len(data_3A)):
        if len(data_3A[i]) != 2:
            raise ValidationError("Number of tab separated columns in line %d is not 2" % (i+1))
        try:
            data_3A[i][0] = int(data_3A[i][0])
            data_3A[i][1] = int(data_3A[i][1])
        except ValueError:
            raise ValidationError("Entry in line %d could not be cast as integer" % (i+1))

    if [x[0] for x in data_3A] != range(1, predK+1):
        raise ValidationError("First column must have %d entries in acending order starting with 1" % predK)

    for i in range(len(data_3A)):
        if data_3A[i][1] not in set(range(predK+1)):
            raise ValidationError("Parent node label in line %d is not valid." % (i+1))

    # Since cluster zero is not included in file
    ad_cluster = np.zeros((len(data_3A)+1, len(data_3A)+1), dtype=int)
    # file starts at one
    for i in range(len(data_3A)):
        ad_cluster[data_3A[i][1]][data_3A[i][0]] = 1
    # fill in a matrix which tells you whether or not one cluster is a descendant of another
    for i in range(len(data_3A)+1):
        for j in range(len(data_3A)+1):
            if (ad_cluster[j][i] == 1):
                for k in range(len(data_3A)+1):
                    if(ad_cluster[k][j] == 1):
                        ad_cluster[k][i] = 1
                    if(ad_cluster[i][k] == 1):
                        ad_cluster[j][k] = 1

    # check if all nodes are connected. If there are not, we could possibly run the above code again 
    if (not np.array_equal(np.nonzero(ad_cluster[0])[0], map(lambda x: x+1, range(len(data_3A))))):
        for i in range(len(data_3A)+1):
            for j in range(len(data_3A)+1):
                if (ad_cluster[j][i] == 1):
                    for k in range(len(data_3A)+1):
                        if(ad_cluster[k][j] == 1):
                            ad_cluster[k][i] = 1
                        if(ad_cluster[i][k] == 1):
                            ad_cluster[j][k] = 1

    if (not np.array_equal(np.nonzero(ad_cluster[0])[0], map(lambda x: x+1, range(len(data_3A))))):
        raise ValidationError("Root of phylogeny not ancestor of all clusters / Tree is not connected.")

    # print ad_cluster
    ad_cluster = np.delete(ad_cluster, 0, 0)
    ad_cluster = np.delete(ad_cluster, 0, 1)

    return ad_cluster

def validate3B(filename, ccm, nssms, mask=None):
    size = ccm.shape[0]
    
    try:
        if os.path.exists(filename):
            ad = np.zeros((size, size))
            if helpers.is_gzip(filename):
                gzipfile = gzip.open(str(filename), 'r')
            else:
                gzipfile = open(str(filename), 'r')
            ad_i = 0
            for i, line in enumerate(gzipfile):
                if mask == None:
                    ad[i, ] = np.fromstring(line, sep='\t')
                elif i in mask:
                    matrix_line = line.split(' ')
                    ad[ad_i, ] = [x for i, x in enumerate(matrix_line) if i in mask]
                    ad_i += 1
            gzipfile.close()
        else:
            #ad = filename
            # TODO - optimize with line by line 
            data = StringIO.StringIO(filename)
            ad = np.zeros((nssms, nssms))
            cm = np.loadtxt(data, ndmin=2)
            ad[:nssms, :nssms] = cm

    except ValueError:
        raise ValidationError("Entry in AD matrix could not be cast as a float")

    if ad.shape != ccm.shape:
        raise ValidationError("Shape of AD matrix %s is wrong.  Should be %s" % (str(ad.shape), str(ccm.shape)))
    if not np.allclose(ad.diagonal(), np.zeros(ad.shape[0])):
        raise ValidationError("Diagonal entries of AD matrix not 0")
    if np.any(np.isnan(ad)):
        raise ValidationError("AD matrix contains NaNs")
    if np.any(np.isinf(ad)):
        raise ValidationError("AD matrix contains non-finite entries")
    if np.any(ad > 1):
        raise ValidationError("AD matrix contains entries greater than 1")
    if np.any(ad < 0):
        raise ValidationError("AD matrix contains entries less than 0")
    if checkForBadTriuIndices(ad, ad.T, ccm):
        raise ValidationError("For some i, j the sum of AD(i, j) + AD(j, i) + CCM(i, j) > 1.")

    return ad

##### VCF ###################################################################

def parseVCF1C(data, sample_mask=None):
    data = data.split('\n')
    data = [x for x in data if x != '']
    data = [x for x in data if x[0] != '#']
    if len(data) == 0:
        raise ValidationError("Input VCF contains no SSMs")
    return [[len(data)], [len(data)]]

def parseVCF2and3(data, sample_mask=None):

    data = data.split('\n')
    data = [x for x in data if x != '']
    data = [x for x in data if x[0] != '#']

    # apply sample_mask to data, sample_mask exists
    data = [x for i, x in enumerate(data) if i in sample_mask] if sample_mask else data

    if len(data) == 0:
        raise ValidationError("Input VCF contains no SSMs")
    vcf_lines = len(data)
    # check if line is true or false, array of 0/1's
    mask = [x[-4:] == "True" for x in data]
    mask = [i for i, x in enumerate(mask) if x]
    true_lines = len(mask)

##### VERIFY (different from validate) ########################################

def verify(filename, role, func, *args, **kwargs):
    # printInfo('ARGS -> %s | %s | %s | %s | %s' % (filename, role, func, args, kwargs))
    t_start = time.time()
    
    try:
        if func.__name__ in ['validate2B']:
            verified = func(filename,*args, **kwargs)
        elif helpers.is_gzip(filename): #pass compressed files directly to 2B or 3B validate functions
            verified = func(filename, *args, **kwargs)
        #if helpers.is_gzip(filename):
        #    verified = func(filename, *args, **kwargs)
        else:
            # really shouldn't do read() here, stores the whole thing in memory when we could read it in chunks/lines
            f = open(filename)
            data = f.read()
            f.close()
            verified = func(data, *args, **kwargs)
    except (IOError, TypeError) as e:
        traceback.print_exc()
        err_msgs.append("Error opening %s, from function %s using file %s in" %  (role, func, filename))
        return None
    except (ValidationError, ValueError) as e:
        err_msgs.append("%s does not validate: %s" % (role, e.value))
        return None
    except SampleError as e:
        raise e
    t_end = time.time()
    #print (filename +  "  " + str(t_end - t_start)) #debug
    return verified

def verify2A(filename_pred, filename_truth, role, pred_size, truth_size, filter_mut=None, mask=None, subchallenge="2A"):
    try:
        f = open(filename_pred)
        data1 = f.read()
        f = open(filename_truth)
        data2 = f.read()
        f.close()
        if subchallenge is "3A":
            verified, raw = om_validate2A(data1, data2, pred_size, truth_size, filter_mut=filter_mut, mask=mask, subchallenge=subchallenge)
            return verified, raw
        verified = om_validate2A(data1, data2, pred_size, truth_size, filter_mut=filter_mut, mask=mask, subchallenge=subchallenge)
    except (IOError, TypeError) as e:
        traceback.print_exc()
        err_msgs.append("Error opening %s, from function new_validate2A using file %s and %s in" %  (role, filename_pred, filename_truth))
        return None
    except (ValidationError, ValueError) as e:
        err_msgs.append("%s does not validate: %s" % (role, e.value))
        return None
    except SampleError as e:
        raise e
    return verified, [-1]

def verifyChallenge(challenge, predfiles, vcf):
    #global err_msgs
    if challengeMapping[challenge]['vcf_func']:
        nssms = verify(vcf, "input VCF", parseVCF1C)
        if nssms == None:
            err_msgs.append("Could not read input VCF. Exiting")
            return "NA"
    else:
        nssms = [[], []]

    if len(predfiles) != len(challengeMapping[challenge]['val_funcs']):
        err_msgs.append("Not enough input files for Challenge %s" % challenge)
        return "Invalid"

    out = []
    for (predfile, valfunc) in zip(predfiles, challengeMapping[challenge]['val_funcs']):
        args = out + nssms[0]
        out.append(verify(predfile, "prediction file for Challenge %s" % (challenge), valfunc, *args))
        if out[-1] == None:
            return "Invalid"
    return "Valid"


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