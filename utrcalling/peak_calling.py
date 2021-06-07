from collections import Counter
import operator

from scipy.cluster.hierarchy import single, fcluster, cut_tree
from scipy.spatial.distance import pdist

from .module_log import module_logger

# Set up logging
logger = module_logger()

def subset_data(utr_positions_counts, utr_reference_regions):
    sub_utr_reference = utr_reference_regions.filter(lambda x: x.name in utr_positions_counts.utr_ids)

    return sub_utr_reference

def filter_peaks(utr_positions_counts, annotation=[]):
    def utr_bigger_than(x, amount):
        return [True if i > amount else False for i in x['utr_sizes_from_reads']]

    filtered = utr_positions_counts.filter(lambda x: x.filter(utr_bigger_than, 148))
    return filtered

def cluster(utr_positions_counts):
    y = pdist(utr_positions_counts, metric='cityblock')
    Z = single(y)
    cutree = cluster.hierarchy.cut_tree(Z, height=50)