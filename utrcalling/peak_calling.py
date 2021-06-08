from scipy.cluster.hierarchy import single, fcluster, cut_tree
from scipy.spatial.distance import pdist

from .module_log import module_logger

# Set up logging
logger = module_logger()

def subset_data(utr_positions_counts, utr_reference_regions):
    sub_utr_reference = utr_reference_regions.filter(
        lambda x: x.name in utr_positions_counts.utr_ids).saveas()
    return sub_utr_reference

def merge_position_counts(utr_positions_counts):
    """
    Before calling peaks from position counts, we should merge the datasets to compare,
    so that peaks get called with the pseudo-bulk data from all samples.
    If you want to call peaks for a single sample, skip this function

    Args:
        utr_positions_counts (list): List with the objects containing the position
        counts for each sample.
    """
    
    def add_counter_to_barcode(x, counter):
        x["read_barcodes"] = [f'{barcode}-{counter}' for barcode in x["read_barcodes"]]
        return x

    sample_counter = 1
    barcode_modified_position_counts = []
    for position_counts in utr_positions_counts:
            changed = position_counts.filter(
                lambda x: x.each(add_counter_to_barcode, sample_counter))
            sample_counter += 1
            barcode_modified_position_counts.append(changed)
    
    position_counts_merged = \
        barcode_modified_position_counts[0].merge(barcode_modified_position_counts[1])

    return position_counts_merged

def filter_positions(utr_positions_counts, utr_reference, annotations=None):
    def filter_func(x, amount, reference):
        condition_1 = [True if i > amount else False for i in x['utr_sizes_from_reads']]
        # condition_2 = [True if i == amount else False for i in x['read_strands']]
        # condition_final = [all(items) for items in zip(condition_1, condition_2)]
        return condition_1

    filtered = utr_positions_counts.filter(
        lambda x: x.filter(filter_func, 148, utr_reference))
    return filtered

def cluster(utr_positions_counts):
    y = pdist(utr_positions_counts, metric='cityblock')
    Z = single(y)
    cutree = cut_tree(Z, height=50)