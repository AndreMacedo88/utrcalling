from scipy.cluster.hierarchy import single, fcluster, cut_tree
from scipy.spatial.distance import pdist
from collections import Counter

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