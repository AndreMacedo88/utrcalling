"""
File containing protocols: objects and functions specific for each supported
sequencing platform, version, strategy, etc.

Each protocol should define functions and variables that are expected to be called during
the run.

Note:
1- These functions are defined as private functions at the end of this file.
2- The protocol itself is defined by the `_check_platform_and_version` function.
3- New protocols should, then, append an elif clause in this function.

"""
# built-in packages
from abc import ABC, ABCMeta, abstractmethod
from distutils.util import strtobool
import re
# data wrangling packages
import numpy as np
import pandas as pd
# stats packages
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from bidict import bidict
# internal packages
from utrcalling.core import module_log, helpers


class Meta(type):
    """
    Allows logging at the end of the protocol initialization.
    """
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.protocol_log()
        return instance


class CombinedMeta(Meta, ABCMeta):
    """
    Metaclass that inherits from the abstract class module metaclass and the user-defined
    Meta metaclass, that allows logging at the end of protocol initialization.

    This is necessary for use in abstract protocols that want to inherit behaviour from
    both those metaclasses.
    """
    pass


class Protocol(ABC, metaclass=CombinedMeta):
    """
    Sequencing protocol interface.

    Provides methods for peak calling and fetching read information and sample names.
    """

    def __init__(
        self,
        annotations=None,
        merge_method="union",
        call_peaks=True,
        call_peaks_method="kde",
        label_batches=None,
        alignments=None,
    ) -> None:

        self.logger = module_log.module_logger()

        # set variables from user
        self.annotations = helpers.check_if_path_and_open(
            to_check=annotations, method="csv"
        )
        self.merge_method = merge_method

        if isinstance(label_batches, bool) or label_batches is None:
            self.label_batches = label_batches
        else:
            self.label_batches = bool(strtobool(label_batches))

        self.call_peaks = call_peaks
        if self.call_peaks:
            if call_peaks_method.lower() == "kde":
                self.peak_calling = self._call_peaks_using_kernel_density_estimator
            else:
                raise NotImplementedError(
                    (
                        f"Method {call_peaks_method} for calling consensus peaks is not "
                        f"implemented."
                    )
                )
        else:
            self.peak_calling = self._bypass_calling_peaks

        # set variables to be modified by the different protocols
        self.umis = False

        self.sample_names = None
        self.sample_idx = None

        # set variables that are calculated from the inputs
        self.batches_list, self.batches_range = self.get_batch_information()
        self.barcode_whitelist = self.get_barcode_list()

    # log the protocol:
    def protocol_log(self):
        self.logger.info(
            f'Protocol:\n' +
            f'║ Position counts merging method: {self.merge_method}\n' +
            f'║ Call peaks? {self.call_peaks}\n' +
            f'║ Peak calling method: {self.peak_calling}\n' +
            f'║ Add a number to batch names while merging batches? {self.label_batches}\n'
        )

    # set-up methods
    def get_sample_names(self, alignments):
        sample_names_dict = bidict()
        sample_idx_dict = dict()
        if not isinstance(alignments, list):
            alignments = [alignments]
        try:
            sample_names = self.annotations.loc[:, self.annotations.columns.str.extract(
                r"(.*sample_id.*)", flags=re.IGNORECASE).values[0]]
        except IndexError as err:
            msg = (
                f"Please supply a column for `sample_id` in the annotation file. "
                f"This column should contain the names of each sample, corresponding "
                f"to the name in the bam file."
            )
            self.logger.error(msg)
            raise IndexError(msg) from err

        not_found_trigger = True
        for idx, sample in enumerate(sample_names.iloc[:, 0]):
            sample_bam = [bam for bam in alignments if sample in bam]
            sample_bam_len = len(sample_bam)
            if sample_bam_len > 1:
                msg = (
                    f"The `sample_id` column in the annotation file should contain "
                    f"IDs that are uniquely identifiable to the provided alignment "
                    f"bam files' path/names."
                )
                self.logger.error(msg)
                raise ValueError(msg)
            elif sample_bam_len == 0:
                if not_found_trigger:
                    msg = (
                        f"Some samples IDs not found in the provided alignment bam files."
                        f"The `sample_id` column in the annotation file should contain "
                        f"IDs that match the provided alignment bam files' path/names."
                    )
                    self.logger.warning(msg)
                    not_found_trigger = False
            else:
                sample_names_dict[sample_bam[0]] = sample
                sample_idx_dict[sample_bam[0]] = self.batches_range[idx]

        return sample_names_dict, sample_idx_dict

    def get_batch_information(self):
        try:
            batches_column_name = self.annotations.columns.str.extract(
                r"(.*batch.*)", flags=re.IGNORECASE).dropna().values[0]
        except IndexError:
            msg = (
                f"If you are working with multiple samples/batches, please supply "
                f"a column for `batch` in the annotation file. "
                f"This column should contain the batch ID for each sample."
            )
            self.logger.warning(msg)
            return None, None  # will make the rest of the protocol behave with no batches

        if len(batches_column_name) > 1:
            raise ValueError(
                f"Too many columns in the annotation file with 'barcode' in "
                f"their name."
            )
        else:
            batches_column_name = batches_column_name[0]
        batches_df = self.annotations.loc[:, batches_column_name]
        batches_list = batches_df.to_list()
        batches_range, _ = pd.factorize(batches_df, sort=False)
        batches_range = (batches_range + 1).astype(str)
        return batches_list, batches_range

    def get_barcode_list(self):
        try:
            barcode_column_name = self.annotations.columns.str.extract(
                r"(.*barcode.*)", flags=re.IGNORECASE).dropna().values[0]
        except IndexError:
            msg = (
                f"If you are working with barcodes, please supply "
                f"a column for `barcode` in the annotation file. "
                f"This column should contain the barcode for each sample/cell, "
                f"as appropriate."
            )
            self.logger.warning(msg)
            return None  # will make the rest of the protocol behave with no barcodes

        if len(barcode_column_name) > 1:
            raise ValueError(
                f"Too many columns in the annotation file with 'barcode' in "
                f"their name."
            )
        else:
            barcode_column_name = barcode_column_name[0]

        barcodes_series = self.annotations.loc[:, barcode_column_name]

        if self.label_batches and (self.batches_list is not None):
            try:
                barcodes_series.str.split("-").to_list()[1]
            except IndexError:
                barcodes_series = (barcodes_series + "-" + self.batches_range)

        barcodes_whitelist = barcodes_series.unique()
        return barcodes_whitelist

    """
            batches_dict = dict()
            try:
                batches_df = annotations.loc[:, [annotations.columns.str.extract(
                    r"(.*batch.*)", flags=re.IGNORECASE).dropna().values[0][0],
                    sample_names.columns[0]]]
            except IndexError as err:
                msg = (
                    f"If you are working with multiple samples/batches, please supply "
                    f"a column for `batch` in the annotation file. "
                    f"This column should contain the batch ID for each sample."
                )
                self.logger.error(msg)
                raise err
            for idx, row in enumerate(batches_df.itertuples()):
                try:
                    batches_dict[sample_names_dict.inverse[row[1]]] = {row[0]: idx}
                except KeyError:
                    msg = (
                        f"Some batches not found for the provided alignment bam files."
                        f"The annotation file should contain only information for the"
                        f"supplied data (bam files)."
                    )
                    self.logger.warning(msg)
                    continue

            return sample_names_dict, batches_dict
        else:
            return None

    def get_batch_codes(self, annotations):
        if annotations is not None:
            try:
                batches_keys = np.unique(annotations.loc[:, annotations.columns.str.extract(
                    r"(.*batch.*)", flags=re.IGNORECASE).dropna().values[0]].values)
                batches = {batches_keys[i]: i + 1 for i in range(len(batches_keys))}
            except IndexError as err:
                msg = (
                    f"If you are working with multiple samples/batches, please supply "
                    f"a column for `batch` in the annotation file. "
                    f"This column should contain the batch ID for each sample."
                )
                self.logger.error(msg)
                raise err
            return batches
        else:
            return None
    """

    # Protocol-specific methods
    @ staticmethod
    # @ abstractmethod
    def define_protocol(alignment, **kwargs):
        pass

    # Methods to get read information from the read alignments
    @ staticmethod
    @ abstractmethod
    def get_reads_identity(alignment, **kwargs):
        pass

    @ staticmethod
    @ abstractmethod
    def check_strand_matches(alignment, **kwargs):
        pass

    @ staticmethod
    @ abstractmethod
    def get_utr_size_from_read(read_start_position, interval):
        pass

    # Methods to process UTR sizes
    @ staticmethod
    @ abstractmethod
    def filter_utr_sizes(alignment, **kwargs):
        pass

    # Methods to aggregate individual molecules' UTR sizes to consensus peaks:
    @ staticmethod
    def _call_peaks_using_kernel_density_estimator(utr_sizes_vector, bandwidth=12):
        """
        Fit a kernel density model to predict the peak's ditribution density function

        Args:
            utr_sizes_vector (np.ndarray): column vector with the UTR sizes to call peaks.

        Returns:
            np.ndarray: The vector with the peaks.
        """

        if len(utr_sizes_vector) <= 3:
            peaks = np.repeat(round(np.median(utr_sizes_vector)), len(utr_sizes_vector))
            return peaks

        if np.ndim(utr_sizes_vector) == 1:
            utr_sizes_vector = utr_sizes_vector.reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(utr_sizes_vector)

        # calculate an appropriate window to check the model
        minimum_size, maximum_size = utr_sizes_vector.min(), utr_sizes_vector.max()
        padding = max((maximum_size - minimum_size) * 0.1, 10)
        window = np.arange(round(minimum_size - padding),
                           round(maximum_size + padding) + 1)
        # get the model predictions for that window. Specifically, the extrema.
        evaluation = kde.score_samples(window.reshape(-1, 1))
        minima = argrelextrema(evaluation, np.less_equal, order=1)[0][1:]
        maxima = argrelextrema(evaluation, np.greater_equal, order=1)[0]
        # Note: [1:] because less_equal always outputs index 0,
        # since padding guarantees that kde rises until first value.

        minima = [minimum for minimum in minima if minimum - 1 not in minima]
        maxima = [maximum for maximum in maxima if maximum - 1 not in maxima]

        if minima[0] < maxima[0]:
            raise RuntimeError("First minima is larger than the first maxima. \
                Peak calling using this method is incorrect for this UTR.")

        # create a container to hold the UTR peak sizes
        peaks = np.zeros(len(utr_sizes_vector))
        # attribute a peak to each cell according to their UTR sizes and the kde model:
        # first iterate over the function' minima
        idx_minimum = 0  # for the unbound warning... change it in the future!
        for idx_minimum, val in enumerate(window[minima]):
            # get the maximum upstream that minimum (the peak) and store it
            try:
                max_ = round(window[maxima][idx_minimum])
            except IndexError:
                continue
            for idx_maximum, cond in enumerate(utr_sizes_vector < val):
                if cond and peaks[idx_maximum] == 0:
                    peaks[idx_maximum] = max_
        # get the last maximum, just downstream the last minimum.
        try:
            max_ = round(window[maxima][idx_minimum + 1])
        except IndexError:  # for those cases with just one peak, there will be no minimum
            # so we take the median as the peak
            max_ = round(np.median(utr_sizes_vector))

        # place the last maximum in those molecules that haven't yet had their
        # peaks scored
        peaks[peaks == 0] = max_
        return peaks

    @ staticmethod
    def _bypass_calling_peaks(utr_sizes_vector):
        if np.issubdtype(utr_sizes_vector.dtype, np.integer):
            return utr_sizes_vector
        else:
            try:
                return np.rint(utr_sizes_vector)
            except TypeError:
                utr_sizes_vector = utr_sizes_vector.astype(float)
                return np.rint(utr_sizes_vector)


class ProtocolThreePrime(Protocol):

    def filter_utr_sizes(self, utr_sizes, threshold=1):
        self.logger.info(f"Filtering UTR sizes using threshold of {threshold} counts.")
        utr_sizes_exploded = utr_sizes.explode(utr_sizes.columns.to_list(),
                                               ignore_index=False)
        utr_sizes_exploded = utr_sizes_exploded[
            utr_sizes_exploded.utr_sizes_from_reads > threshold
        ]
        utr_sizes = utr_sizes_exploded.groupby(level=0).agg(list)
        return utr_sizes

    @ staticmethod
    def check_strand_matches(read_strand, utr_strand):
        return read_strand != utr_strand

    @ staticmethod
    def get_utr_size_from_read(read_start_position, interval):
        return abs(read_start_position - interval.start_d)


class ProtocolFivePrime(Protocol):

    @ staticmethod
    def check_strand_matches(read_strand, utr_strand):
        return read_strand == utr_strand

    @ staticmethod
    def filter_utr_sizes(utr_sizes, **kwargs):
        logger = module_log.module_logger()
        logger.info("""
        For 5'UTR protocols, no UTR size filtering is performed because having transcripts
        with 5'UTR size 0 means these transcripts start at the ATG site.
        """)
        return utr_sizes

    @ staticmethod
    def get_utr_size_from_read(read_start_position, interval):
        return abs(read_start_position - interval.end_d)


class ProtocolTenx(ProtocolThreePrime):
    def __init__(self, merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_batches=None, alignments=None, annotations=None) -> None:

        super().__init__(
            merge_method=merge_method,
            call_peaks=call_peaks,
            call_peaks_method=call_peaks_method,
            label_batches=label_batches,
            alignments=alignments,
            annotations=annotations,
        )

        if self.label_batches is None:
            self.label_batches = True


class ProtocolTenx3(ProtocolTenx):

    @ staticmethod
    def get_reads_identity(alignment, **_):
        read_seq = alignment.read.seq
        read_barcode = helpers.to_string(read_seq[:16])
        read_umi = helpers.to_string(read_seq[16:28])  # 12 bp UMI
        return read_umi, read_barcode


class ProtocolTenx2(ProtocolTenx):

    @ staticmethod
    def get_reads_identity(alignment, **_):
        """
        Get read UMI and barcode from 10x version 2

        Args:
            alignment (HTSeq.SAM_Alignment): The genome alignment line with the read
            information.

        Returns:
            tuple: a tuple of strings, containing the the read UMI and barcode in
            position 0 and 1, respectively.
        """
        read_seq = alignment.read.seq
        read_barcode = helpers.to_string(read_seq[:16])
        read_umi = helpers.to_string(read_seq[16:26])  # 10 bp UMI
        return read_umi, read_barcode


class ProtocolSmartseq(ProtocolThreePrime):
    def __init__(self, merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_batches=None, alignments=None, annotations=None) -> None:

        super().__init__(
            merge_method=merge_method,
            call_peaks=call_peaks,
            call_peaks_method=call_peaks_method,
            label_batches=label_batches,
            alignments=alignments,
            annotations=annotations,
        )

        if self.label_batches is None:
            self.label_batches = False
        self.sample_names, self.sample_idx = self.get_sample_names(
            alignments=alignments
        )


class ProtocolSmartseq3(ProtocolSmartseq):

    def get_reads_identity(self, alignment, **kwargs):
        bam_name = kwargs.get("bam_name")
        read_sample_idx = self.sample_idx[bam_name]

        read_seq = alignment.read.seq
        if self.label_batches:
            read_barcode = f"{helpers.to_string(read_seq[:6])}-{read_sample_idx}"
        else:
            read_barcode = helpers.to_string(read_seq[:6])
        read_umi = helpers.to_string(read_seq[6:14])  # 8 bp UMI
        return read_umi, read_barcode


class ProtocolCapseq(ProtocolFivePrime):
    def __init__(self, merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_batches=None, alignments=None, annotations=None) -> None:

        super().__init__(
            merge_method=merge_method,
            call_peaks=call_peaks,
            call_peaks_method=call_peaks_method,
            label_batches=label_batches,
            alignments=alignments,
            annotations=annotations,
        )

        if self.label_batches is None:
            self.label_batches = False
        self.sample_names, self.sample_idx = self.get_sample_names(
            alignments=alignments
        )


class ProtocolCapseqSmartseq3(ProtocolCapseq):

    @ staticmethod
    def get_reads_identity(alignment, **kwargs):
        read_batch_id = kwargs.get('bam_name', 'S')
        read_seq = helpers.to_string(alignment.read.seq)

        ss3_regex = "(ATTGCGCAATG)[NGCAT]{8}"
        umi_match = re.search(ss3_regex, read_seq)
        if umi_match:
            read_umi = umi_match.group(0)[11:19]  # 8 bp UMI
            assert(len(read_umi) == 8)  # 8 bp UMI
        else:
            return None, read_batch_id

        return read_umi, read_batch_id


class ProtocolCapseqSmartseq3ReadName(ProtocolCapseq):

    @ staticmethod
    def get_reads_identity(alignment, **kwargs):
        read_batch_id = kwargs.get('bam_name', 'S')

        read_seq = helpers.to_string(alignment.read.name)
        read_umi = read_seq.split('_')[1]
        assert(len(read_umi) == 8)  # 8 bp UMI

        return read_umi, read_batch_id


def protocol(platform, version, **kwargs):
    error_message = f"Protocol for {platform} version {version} not implemented."

    if platform.lower() in ("10x", "10_x", "tenx", "ten_x"):
        if version == 2:
            return ProtocolTenx2(**kwargs)
        if version == 3:
            return ProtocolTenx3(**kwargs)
        else:
            raise NotImplementedError(error_message)
    elif platform.lower() in ('smart-seq', 'smartseq'):
        if version == 3:
            return ProtocolSmartseq3(**kwargs)
        else:
            raise NotImplementedError(error_message)
    elif platform.lower() in ('smart-seq_cap', 'smartseq_cap'):
        if version == 3:
            return ProtocolCapseqSmartseq3(**kwargs)
        else:
            raise NotImplementedError(error_message)
    elif platform.lower() in ('smart-seq_cap-readname', 'smartseq_cap-readname'):
        if version == 3:
            return ProtocolCapseqSmartseq3ReadName(**kwargs)
        else:
            raise NotImplementedError(error_message)
    else:
        raise NotImplementedError(error_message)
