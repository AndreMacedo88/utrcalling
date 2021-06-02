from scipy.cluster.hierarchy import single, fcluster, cut_tree

from scipy.spatial.distance import pdist

from .module_log import module_logger

# Set up logging
logger = module_logger()

def filter_peaks():
    pass


def cluster(utr_positions_counts):
    y = pdist(utr_positions_counts, metric='cityblock')
    Z = single(y)
    cutree = cluster.hierarchy.cut_tree(Z, height=50)