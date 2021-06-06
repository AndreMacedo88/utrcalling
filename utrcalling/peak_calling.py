from collections import Counter
import operator

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import single, fcluster, cut_tree
from scipy.spatial.distance import pdist

from .module_log import module_logger

# Set up logging
logger = module_logger()

def filter_peaks(read_mappings):
    read_mappings = read_mappings.filter('>','utr_sizes_from_reads', 0, inplace=False)
    for utr in read_mappings.utr_ids:
        sizes = read_mappings[utr]['utr_sizes_from_reads']
        number_peaks = Counter(sizes)
        read_mappings[utr]['utr_sizes_from_reads'] = \
        [key for key, value in number_peaks.items() if value > 1]


def cluster(utr_positions_counts):
    y = pdist(utr_positions_counts, metric='cityblock')
    Z = single(y)
    cutree = cluster.hierarchy.cut_tree(Z, height=50)

def filter_peaks_2(data, operation, feature, amount, inplace=False):
    if operation in ('>','<','=','>=', '<='):
        result = operator_functions(data, operation, feature, amount, inplace=inplace)
    return result

# Private functions

def operator_functions(data, operation, by, amount, inplace=False):
    ops = { 
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "=": operator.eq,
        }
    if not inplace:
        new = data.copy(deep=True)
    ops_operation = ops[operation]
    rdl = data.read_lengths
    lengths = testss(amount, rdl, ops_operation)
    new.read_lengths = lengths
    if not inplace:
        return new
    else:
        return None

def testss(amount, rdl, ops_operation):
    new_list = []
    for utr in rdl:
        lengths = [item for item in utr if ops_operation(item, amount)]
        new_list.append(lengths)
    return new_list