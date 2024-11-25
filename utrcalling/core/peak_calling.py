"""
peak_calling.py - module with functions to deal with calling peaks
"""

import os
import warnings
from tempfile import mkstemp
import pandas as pd
import modin.pandas as mpd
from modin.utils import to_pandas
import anndata
from utrcalling.core import helpers, module_log
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
# silence modin UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


def generate_peak_counts_and_anno(
    utr_sizes,
    protocol,
):
    logger = module_log.module_logger()
    helpers.init_ray()

    annotations = protocol.annotations
    func = protocol.peak_calling

    logger.info("Generating and counting UTR size peaks")

    if not isinstance(utr_sizes, mpd.DataFrame):
        _, df_path = mkstemp(suffix=".feather", dir=".")
        utr_sizes.index.name = "utr_name"
        utr_sizes = utr_sizes.reset_index()
        utr_sizes.to_feather(df_path)
        utr_sizes = mpd.read_feather(df_path)
        os.remove(df_path)
        utr_sizes = utr_sizes.set_index("utr_name")

    # iterate over each UTR, to call peaks corresponding to the UTR's different sizes.
    logger.info("Calling peaks for each UTR")

    count_matrix_list, peaks_var_list = zip(
        *utr_sizes.apply(call_peaks, axis=1, func_=func)
    )

    logger.info("Finished generating and counting UTR size peaks")

    # Free some RAM by deleting the biggest variables
    del utr_sizes

    # count_matrix_list = [
    #     count_mtx.fillna(0).apply(mpd.to_numeric, downcast="unsigned")
    #     for count_mtx in count_matrix_list
    # ]

    ######################## patchy patch areas ########################

    """
    print(count_matrix_list[0])
    print("passed fillna and downcasting step")

    count_matrix_list = [
        count_mtx.astype(pd.SparseDtype(int, fill_value=0))
        for count_mtx in count_matrix_list
    ]

    print("passed sparsify step")
    try:
        print(f"Type of the first count_mtx: {count_matrix_list[0].head()}")
        print(f"Density of the first count_mtx: {count_matrix_list[0].sparse.density}")
    except:
        pass

    md, hd = "w", True
    for df in tqdm(count_matrix_list, total=len(count_matrix_list)):
        df.reset_index().to_csv("df_all.csv", mode=md, header=hd, index=None)
        md, hd = "a", False
    del count_matrix_list
    count_matrix = pd.read_csv("df_all.csv", index_col=0).T
    # os.remove("df_all.csv")
    print(count_matrix)

    """
    """

    def chunks(lst, n):
        '''Yield successive n-sized chunks from lst.'''
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    import math

    to_concat = []
    chunk_size = 400

    chunk_n = math.ceil(len(count_matrix_list) / chunk_size)
    for chunk in tqdm(chunks(count_matrix_list, chunk_size), total=chunk_n):
        to_concat.append(pd.concat(chunk, axis=1, ignore_index=False, copy=False))

    del count_matrix_list

    print("Filling NAs with 0 and downcasting")
    to_concat = [
        count_mtx.fillna(0).apply(mpd.to_numeric, downcast="unsigned")
        for count_mtx in to_concat
    ]

    print("Storing intermediate count matrices")
    directory = "./intermediate_files/"
    ct = 1
    for peaks_count_matrix in tqdm(to_concat):
        try:
            peaks_count_matrix.to_parquet(
                os.path.join(directory, f"peaks_count_matrix_{ct}.parquet")
            )
        except Exception:
            peaks_count_matrix.to_csv(
                os.path.join(directory, f"peaks_count_matrix_{ct}.csv")
            )
        ct += 1

    del to_concat

    # print(f"Concatenating {chunk_n} chunks into a final data set")
    # count_matrix = pd.concat(to_concat, axis=1, ignore_index=False, copy=False)
    # print("Concatenation complete")
    count_matrix = []  # Remove after solving this issue

    """
    ##################### end of patchy patch area 1 #####################

    """
    print("Starting Concatenation")
    # count_matrix = pd.concat(count_matrix_list, axis=1, ignore_index=False, copy=False)

    # Function to concatenate chunks and save them to disk
    def process_chunks(input_files, output_dir, chunk_size, round_number):
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        # Split input files into groups of `chunk_size`
        file_chunks = [
            input_files[i : i + chunk_size]
            for i in range(0, len(input_files), chunk_size)
        ]

        for i, file_chunk in tqdm(
            enumerate(file_chunks),
            desc=f"Round {round_number} chunk",
            total=len(file_chunks),
        ):
            # Read and concatenate the current chunk
            dfs = [pd.read_parquet(file) for file in file_chunk]
            concatenated_df = pd.concat(dfs, axis=1, ignore_index=False, copy=False)

            # Save the concatenated result
            output_file = os.path.join(
                output_dir, f"round_{round_number}_chunk_{i}.parquet"
            )
            concatenated_df.to_parquet(output_file)
            output_files.append(output_file)
            # print(
            #     f"Round {round_number}: Saved concatenated chunk {i+1}/{len(file_chunks)} to {output_file}"
            # )

        return output_files

    # Directory to save intermediate files
    base_output_dir = "intermediate_files"

    round_number = 0

    # Convert initial DataFrames to Parquet files if they're not already saved
    if isinstance(count_matrix_list[0], pd.DataFrame):
        input_files = []
        os.makedirs(base_output_dir, exist_ok=True)
        for i, df in enumerate(count_matrix_list):
            file_path = os.path.join(base_output_dir, f"initial_chunk_{i}.parquet")
            df.to_parquet(file_path)
            input_files.append(file_path)
    else:
        input_files = count_matrix_list

    count_matrix_list = None

    # Define chunk size for each round
    initial_chunk_sizes = [
        100,
        10,
        5,
        2,
        2,
        2,
        2,
        2,
    ]  # Adjust based on available memory
    current_files = input_files

    # Perform multiple rounds of concatenation
    while len(current_files) > 1:
        round_number += 1
        output_dir = os.path.join(base_output_dir, f"round_{round_number}")

        # Process the current round of files
        current_files = process_chunks(
            current_files,
            output_dir,
            initial_chunk_sizes[round_number - 1],
            round_number,
        )

    count_matrix = current_files[0]
    """
    ##################### end of patchy patch area 2 #####################

    '''
    import numpy as np

    def manual_concat(dataframes, chunk_size=100):
        """
        Manually concatenates a list of DataFrames horizontally (axis=1),
        handling alignment without using Pandas' concat to avoid memory issues.

        Args:
        - dataframes (list): List of Pandas DataFrames to concatenate.
        - chunk_size (int): Number of DataFrames to process in each chunk.

        Returns:
        - pd.DataFrame: The final concatenated DataFrame.
        """
        # Step 1: Gather all unique row indices (master index)
        all_indices = set()
        for df in dataframes:
            all_indices.update(df.index)
        master_index = sorted(all_indices)  # Ensure consistent order

        # Step 2: Prepare for concatenation
        concatenated_values = []
        concatenated_columns = []

        # Process DataFrames in chunks
        for i in tqdm(range(0, len(dataframes), chunk_size), total=len(dataframes)):
            chunk = dataframes[i : i + chunk_size]

            # Align each DataFrame in the chunk
            aligned_values = []
            for df in chunk:
                # Reindex to match the master index, filling missing rows with NaN
                aligned_df = df.reindex(master_index)

                # Fill NaN with 0 and downcast to smallest possible unsigned integer type
                aligned_df = aligned_df.fillna(0).to_numpy(dtype=np.uint16)

                # Append the aligned array to the list
                aligned_values.append(aligned_df)
                concatenated_columns.extend(df.columns)

            # Horizontally stack the aligned values
            chunk_values = np.hstack(aligned_values)
            concatenated_values.append(chunk_values)

            # Free memory for processed chunk
            del aligned_values, chunk_values

        # Step 3: Vertically stack all chunks
        final_values = np.hstack(concatenated_values)

        # Step 4: Reconstruct the final DataFrame
        final_df = pd.DataFrame(
            final_values, index=master_index, columns=concatenated_columns
        )
        return final_df

    # Concatenate manually
    count_matrix = manual_concat(count_matrix_list, chunk_size=1000)
    '''
    ##################### end of patchy patch area 3 #####################

    print("passed fillna and downcasting")

    master_index = set()
    for df in count_matrix_list:
        master_index.update(df.index.to_list())
    master_index = sorted(master_index)  # Ensure consistent order

    print(f"Total cells: {len(master_index)}")
    print(count_matrix_list[0].shape)
    print(count_matrix_list[1].shape)

    print("starting reindexing and sparsify steps")
    count_matrix_list = [
        # Reindex to match the master index, filling missing rows with NaN
        df.reindex(master_index, fill_value=0).astype(pd.SparseDtype(int, fill_value=0))
        for df in count_matrix_list
    ]

    try:
        print(f"First count_mtx: {count_matrix_list[0].shape}")
        print(f"Density of the first count_mtx: {count_matrix_list[0].sparse.density}")
        print(f"Total rows expected: {len(master_index)}")
    except:
        pass

    print("passed sparsify step")

    print("concatenating sparse data frames")
    count_matrix = pd.concat(count_matrix_list, axis=1, ignore_index=False, copy=False)

    # print("Starting final fillna and downcasting")
    # count_matrix = count_matrix.fillna(0).apply(pd.to_numeric, downcast="unsigned")

    peaks_var = pd.concat(peaks_var_list, axis=0, ignore_index=False)

    logger.info("Adding annotations")
    if protocol.barcode_column_name:
        peaks_obs = annotations.set_index(protocol.barcode_column_name)
        peaks_obs = peaks_obs.reindex(index=count_matrix.index)
        return count_matrix, peaks_var, peaks_obs
    else:
        return count_matrix, peaks_var


def call_peaks(utr, func_):
    # create containers to append the data
    count_matrix = {}
    peaks_var = {}

    info = utr.to_frame().T
    info = info.explode(column=info.columns.tolist(), ignore_index=True)
    info = info.sort_values(by="utr_sizes_from_reads")

    utr_sizes_vector = info["utr_sizes_from_reads"].values.reshape(-1, 1)
    # now we call the utr size peaks based on the distribution of individual UTR sizes
    peaks = func_(utr_sizes_vector)
    # store this info in a new key on the original container
    info["peaks"] = peaks
    # group by each peak size and cell and count the number of each peak each cell has
    info_grouped_by_peaks = info.groupby(["peaks", "read_barcodes"])["peaks"].agg(
        "count"
    )

    # append this information to the containers created above
    for utr_size_at_peak, barcodes in info_grouped_by_peaks.groupby(level=0):
        # generate a new name for each peak. Here, it's UTR name (which is gene name
        # plus a counter for each transcript UTR) followed by the peak UTR length
        peak_name = f"{utr.name}-{utr_size_at_peak}"
        # update the peak observations table
        peaks_var[peak_name] = utr_size_at_peak
        # update the peak count matrix
        peak_counts = barcodes.droplevel(0)
        count_matrix[peak_name] = peak_counts

    peaks_var = pd.DataFrame.from_dict(peaks_var, orient="index", columns=["utr_size"])
    count_matrix = pd.DataFrame(count_matrix)

    return count_matrix, peaks_var


def add_to_peaks_var(peak_var, annotations):
    logger = module_log.module_logger()
    logger.info("Adding reference annotations to the final result")

    annotations["utr_name"] = annotations.index

    peak_var_new = peak_var.copy(deep=True)
    peak_var_new["utr_name"] = peak_var_new.index.str.split(pat=r"-").str[0]

    # Modin join does not allow joins not using the index.
    # so unfortunately for now we must revert back to pandas to use all columns
    if isinstance(annotations, mpd.dataframe.DataFrame):
        annotations = to_pandas(annotations)
    if isinstance(peak_var_new, mpd.dataframe.DataFrame):
        annotations = to_pandas(annotations)

    peak_var_new = peak_var_new.join(
        annotations, on="utr_name", how="left", rsuffix="_right"
    )

    peak_var_new = peak_var_new.drop("utr_name_right", axis=1)

    return peak_var_new


def generate_count_matrix_anndata(count_matrix, var, obs=None):
    logger = module_log.module_logger()
    logger.info("Generating Count Matrix")

    count_matrix_anndata = anndata.AnnData(
        count_matrix.to_numpy(), var=var, obs=obs, dtype=count_matrix.values.dtype
    )
    return count_matrix_anndata
