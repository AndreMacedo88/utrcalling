from functools import reduce
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # silence modin UserWarnings

import modin.pandas as mpd
from modin.utils import to_pandas
import numpy as np
from rapidfuzz import process, distance

from utrcalling.core import helpers, module_log


def merge_samples_utr_sizes(utr_sizes, method="intersection", label_batches=True):
    """
    Before calling peaks from position counts, we should merge the datasets we want to
    compare, so that peaks get called with the pseudo-bulk data from all samples.
    Merging appends both samples information for each UTR. If `method='intersection'`,
    UTRs that only one sample mapped to will be discarded. If `method='union'`, then
    a union merge is performed.
    If you want to call peaks for a single sample, skip this function.

    Args:
        utr_sizes (list): List with the objects containing the position
        counts for each sample.

    Returns:
        mpd.Dataframe: An object with the merged datasets.
    """

    logger = module_log.module_logger()

    if label_batches:
        if not (
            utr_sizes[0]["read_barcodes"]
            .sample(n=20)
            .explode()
            .str.split("-")
            .apply(len)
            > 1
        ).any():  # confirm that the batches haven't already been labelled
            for idx, batch in enumerate(utr_sizes):
                batch.read_barcodes = [
                    [f"{barcode}-{idx+1}" for barcode in barcodes]
                    for barcodes in batch.read_barcodes
                ]

    logger.info("Merging all batches' UTR sizes")
    if method == "union":
        utr_sizes = mpd.concat(utr_sizes)

        try:
            utr_sizes = utr_sizes.groupby(level=0, observed=True).sum()
        except ValueError as err:
            message = (
                f"During sample UTR size merging, the following error arised:\n{err}\n",
                "Bypassing this error may take a while.",
            )
            logger.warning(message)

            utr_sizes = utr_sizes.groupby(level=0).agg(
                {
                    "read_barcodes": lambda grp: [
                        i for lst in grp.to_numpy().flatten() for i in lst
                    ],
                    "utr_sizes_from_reads": lambda grp: [
                        i for lst in grp.to_numpy().flatten() for i in lst
                    ],
                }
            )

    elif method == "intersection":
        utr_sizes = reduce(lambda a, b: (a + b).dropna(axis=0), utr_sizes)
    else:
        raise AttributeError(f"Method {method} not supported.")

    return utr_sizes


def merge_utrs_per_gene(utr_regions, read_mappings):

    def join_unique(x):
        return ",".join(x.unique().astype(str))

    def split_and_then_join_unique(x):
        split = x.str.split(",")
        split = split.explode()
        return ",".join(np.unique(split))

    def explode_and_then_join(x):
        explode = x.explode()
        return explode.to_list()

    def transform_utr_sizes(df):
        strand = df.strand[0]
        if strand == "-":
            df[["start", "end"]] = df[["end", "start"]]
            df = df.sort_values(by=["start", "end"], ascending=False)

        start = df.start[0]  # needs sorted df
        end = df.end[0]
        left_over = 0
        to_add = []

        if strand == "+":
            for row in df.itertuples():
                if row.start == start:
                    to_add.append(left_over)
                    if row.end <= end:
                        continue
                    else:
                        end = row.end
                    continue
                elif row.start <= end:
                    to_add.append(left_over + (row.start - start))
                    if row.end <= end:
                        continue
                    else:
                        end = row.end
                    continue
                elif row.start > end:
                    left_over += end - start
                    to_add.append(left_over)
                    start = row.start
                    end = row.end

        elif strand == "-":
            for row in df.itertuples():
                if row.start == start:
                    to_add.append(left_over)
                    if row.end >= end:
                        continue
                    else:
                        end = row.end
                    continue
                elif row.start >= end:
                    to_add.append(left_over + (start - row.start))
                    if row.end >= end:
                        continue
                    else:
                        end = row.end
                    continue
                elif row.start < end:
                    left_over += start - end
                    to_add.append(left_over)
                    start = row.start
                    end = row.end

        df["to_add"] = to_add
        explode = df.explode("utr_sizes_from_reads")
        explode["utr_sizes_from_reads"] = (
            explode["utr_sizes_from_reads"] + explode["to_add"]
        )
        return explode["utr_sizes_from_reads"].to_list()

    logger = module_log.module_logger()
    mpd_switch = False

    # join the 2 dataframes so that we can do the operations only once
    # note that `how="inner"` is needed so that there are no NaN values in the UTR sizes
    combined_data = utr_regions.join(read_mappings, how="inner")
    # we must also change the genes without name from NaN to a string
    combined_data["gene_name"] = combined_data["gene_name"].fillna(value="None")
    # sorting is necessary to then apply the `transform_utr_sizes` function
    combined_data = combined_data.sort_values(by=["start", "end"])

    grouped = combined_data.groupby("name")
    utr_regions_per_gene = grouped.agg(
        {
            "chrom": "first",
            "start": join_unique,
            "end": join_unique,
            # "name": "first",  # this is the same as the new index, but may be needed for legacy purposes?
            "strand": "first",
            "transcript_id": split_and_then_join_unique,
            "gene_name": split_and_then_join_unique,
            "read_barcodes": explode_and_then_join,
        }
    )

    if isinstance(combined_data, mpd.dataframe.DataFrame):
        # Modin groupBy.apply splits the partitions column-wise
        # so unfortunately for now we must revert back to pandas to use all columns
        mpd_switch = True
        logger.info("Converting Modin dataframe back to pandas as fallback.")
        combined_data = to_pandas(combined_data)
        grouped = combined_data.groupby("name")

    utr_sizes_from_reads = grouped.apply(transform_utr_sizes)
    utr_sizes_from_reads.name = "utr_sizes_from_reads"

    if mpd_switch:
        # then we must re-distribute the dataframe
        logger.info("Re-distributing the pandas dataframe.")
        utr_sizes_from_reads = mpd.DataFrame(utr_sizes_from_reads)

    utr_regions_per_gene = utr_regions_per_gene.join(utr_sizes_from_reads, how="left")
    utr_regions_per_gene.index.name = "utr_name"

    utr_regions = utr_regions_per_gene.loc[
        :,
        [
            "chrom",
            "start",
            "end",
            "strand",
            "transcript_id",
            "gene_name",
        ],
    ]
    utr_regions["name"] = utr_regions.index

    read_mappings = utr_regions_per_gene.loc[
        :,
        [
            "read_barcodes",
            "utr_sizes_from_reads",
        ],
    ]
    return utr_regions, read_mappings


def subset_reference_data(utr_sizes, utr_reference_regions):
    logger = module_log.module_logger()
    logger.info("Subsetting UTR reference data")

    sub_utr_reference = utr_reference_regions.filter(items=utr_sizes.index, axis=0)

    sub_utr_reference_unique = sub_utr_reference.drop_duplicates()

    return sub_utr_reference_unique


def filter_barcodes(utr_sizes, whitelist):
    logger = module_log.module_logger()
    logger.info("Filtering cell barcodes")

    barcodes_to_test = set(helpers.flatten(utr_sizes.read_barcodes))
    barcodes_correct = set()
    barcodes_replace = {}
    for barcode in barcodes_to_test:
        if barcode in whitelist:
            barcodes_correct.add(barcode)
        else:
            try:
                match = process.extract(
                    barcode, whitelist, scorer=distance.Hamming.distance, score_cutoff=1
                )
            except ValueError as err:
                logger.error(
                    f"\nError matching sample barcode/ID: {barcode}\nPlease check if "
                    f"the barcodes in the annotation file are correct or, if using "
                    f"different batches with similar barcodes/IDs, if all the batches "
                    f"are correctly described in the annotations file."
                )
                logger.error(f"Barcodes from the annotation file:\n{whitelist}")
                raise err
            if match:
                barcode_correct = [bc[0] for bc in match]
                for bc in barcode_correct:
                    try:  # check if the batch is the same between barcodes
                        if barcode.split("-")[1] == bc.split("-")[1]:
                            barcodes_correct.add(bc)
                            barcodes_replace[barcode] = bc
                        else:
                            continue
                    except IndexError:  # if working without batches, no need to check
                        barcodes_correct.add(bc)
                        barcodes_replace[barcode] = bc
                        break
            else:
                continue

    utr_sizes = utr_sizes.explode(utr_sizes.columns.to_list())
    utr_sizes.read_barcodes.replace(barcodes_replace, inplace=True)
    utr_sizes = utr_sizes[utr_sizes.read_barcodes.isin(barcodes_correct)]

    return utr_sizes.groupby(level=0).agg(list)
