"""
File containing protocols: objects and functions specific for each supported
sequencing platform, version, strategy, etc.

Each protocol should define functions and variables that are expected to be called during
the run.

Note:
1- These functions are defined as class extensions at the end of this file.
2- New protocols should, then, extend the necessary class in this file.

"""

# built-in packages
from abc import ABC, ABCMeta, abstractmethod
from distutils.util import strtobool
import re

# data wrangling packages
import numpy as np
import pandas as pd
from bidict import bidict

# internal packages
from utrcalling.core import module_log, helpers
from utrcalling.core.protocols import peak_calling as pc


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
        alignments=None,  # TODO: either remove this or use alignments to improve protocol
    ) -> None:

        self.logger = module_log.module_logger()

        # set variables from the arguments
        self.annotations = helpers.check_if_path_and_open(
            to_check=annotations, method="csv"
        )
        try:
            self.annotations = self.annotations.dropna(axis=0, how="all")
        except TypeError:
            pass

        self.merge_method = merge_method

        if isinstance(label_batches, bool) or label_batches is None:
            self.label_batches = label_batches
        else:
            self.label_batches = bool(strtobool(label_batches))

        self.call_peaks = call_peaks

        if self.call_peaks:
            self.peak_calling = pc.get_caller(call_peaks_method)
        else:
            self.peak_calling = pc.bypass_calling_peaks

        # set variables to be modified by the different protocols
        self.umis = False
        self.barcode_whitelist = None
        self.sample_names = None
        self.sample_idx = None

        # set variables that are calculated from the inputs
        self.barcode_column_name = self._get_barcode_column()
        self.batch_column_name = self._get_batch_column()

        self.batches_list, self.batches_range = self._get_batch_information()

        if self.barcode_column_name is not None:
            self.annotations = self._resolve_barcode_reuse_by_batch()
            self.barcode_whitelist = self._get_barcode_list()

    # log the protocol:

    def protocol_log(self):
        self.logger.info(
            "Protocol:\n"
            + f"║ Position counts merging method: {self.merge_method}\n"
            + f"║ Call peaks? {self.call_peaks}\n"
            + f"║ Peak calling method: {self.peak_calling}\n"
            + f"║ Add a number to batch names while merging batches? {self.label_batches}\n"
        )

    # set-up methods
    def get_sample_names(self, alignments):
        sample_names_dict = bidict()
        sample_idx_dict = dict()
        if not isinstance(alignments, list):
            alignments = [alignments]

        sample_column_name = self._get_sample_column()
        sample_names = self.annotations.loc[:, sample_column_name]

        not_found_trigger = True
        for idx, sample in enumerate(sample_names):
            # TODO: the next 2 lines break if the sample_id column in anno file contains
            # a substring of more than one alignment
            sample_bam = [bam for bam in alignments if sample in bam]
            sample_bam_len = len(sample_bam)
            if sample_bam_len > 1:
                msg = (
                    "The `sample_id` column in the annotation file should contain "
                    "IDs that are uniquely identifiable to the provided alignment "
                    "bam files' path/names."
                )
                self.logger.error(msg)
                raise ValueError(msg)
            elif sample_bam_len == 0:
                if not_found_trigger:
                    msg = (
                        "Some samples IDs not found in the provided alignment bam files."
                        "The `sample_id` column in the annotation file should contain "
                        "IDs that match the provided alignment bam files' path/names."
                    )
                    self.logger.warning(msg)
                    not_found_trigger = False
            else:
                sample_names_dict[sample_bam[0]] = sample
                sample_idx_dict[sample_bam[0]] = self.batches_range[idx]

        return sample_names_dict, sample_idx_dict

    def _get_sample_column(self):
        try:
            sample_column_name = (
                self.annotations.columns.str.extract(
                    r"(.*sample_id.*)", flags=re.IGNORECASE
                )
                .dropna()
                .values
            )
        except IndexError as exc:
            msg = (
                "Please supply a column for `sample_id` in the annotation file. "
                "This column should contain the names of each sample, corresponding "
                "to the name in the bam file."
            )
            self.logger.error(msg)
            raise IndexError(msg) from exc

        if len(sample_column_name) > 1:
            raise ValueError(
                "Too many columns in the annotation file with 'sample_id' in "
                "their name."
            )
        else:
            sample_column_name = sample_column_name[0][0]
        return sample_column_name

    def _get_batch_column(self):
        batch_column_name = (
            self.annotations.columns.str.extract(r"(.*batch.*)", flags=re.IGNORECASE)
            .dropna()
            .values
        )
        if not batch_column_name:
            msg = (
                "If you are working with multiple samples/batches, please supply "
                "a column for `batch` in the annotation file. "
                "This column should contain the batch ID for each sample."
            )
            self.logger.warning(msg)
            return None

        if len(batch_column_name) > 1:
            raise ValueError(
                "Too many columns in the annotation file with 'batch' in " "their name."
            )
        else:
            batch_column_name = batch_column_name[0][0]
        return batch_column_name

    def _get_barcode_column(self):
        barcode_column_name = (
            self.annotations.columns.str.extract(r"(.*barcode.*)", flags=re.IGNORECASE)
            .dropna()
            .values
        )
        if not barcode_column_name:
            msg = (
                "If you are working with barcodes, please supply "
                "a column for `barcode` in the annotation file. "
                "This column should contain the barcode for each sample/cell, "
                "as appropriate."
            )
            self.logger.warning(msg)
            return None

        if len(barcode_column_name) > 1:
            raise ValueError(
                "Too many columns in the annotation file with 'barcode' in "
                "their name."
            )
        else:
            barcode_column_name = barcode_column_name[0][0]
        return barcode_column_name

    def _get_batch_information(self):
        if self.batch_column_name is None:
            return (
                None,
                None,
            )  # will make the rest of the protocol behave without batches

        batches_series = self.annotations.loc[:, self.batch_column_name]
        batches_list = batches_series.to_list()
        if np.issubdtype(batches_series.dtype, np.integer):
            batches_range = batches_series.astype(str)
        else:
            batches_range, _ = pd.factorize(batches_series, sort=False)
            batches_range = (batches_range + 1).astype(str)
        return batches_list, batches_range

    def _resolve_barcode_reuse_by_batch(self):
        annotations = self.annotations.copy()
        if self.label_batches and (self.batches_list is not None):
            barcodes = annotations.loc[:, self.barcode_column_name]
            if barcodes.str.match(".*-[\d]$").all():
                pass
            else:
                barcodes_suffix = [f"-{val}" for val in self.batches_range]
                annotations.loc[:, self.barcode_column_name] = (
                    barcodes.astype(str) + barcodes_suffix
                )
            return annotations
        elif self.label_batches:
            msg = (
                "Please use a column in your annotation file with the "
                "batch information"
            )
            self.logger.error(msg)
            raise KeyError(msg)
        else:
            return annotations

    def _get_barcode_list(self):
        barcodes_series = self.annotations.loc[:, self.barcode_column_name]
        barcodes_whitelist = barcodes_series.unique()
        return barcodes_whitelist

    # Methods to get read information from the read alignments
    @staticmethod
    @abstractmethod
    def get_reads_identity(alignment, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def check_strand_matches(alignment, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_utr_size_from_read(read_start_position, interval):
        pass

    # Methods to process UTR sizes
    @staticmethod
    @abstractmethod
    def filter_utr_sizes(alignment, **kwargs):
        pass


class ProtocolThreePrime(Protocol):

    def filter_utr_sizes(self, utr_sizes, threshold=1):
        self.logger.info("Filtering UTR sizes using threshold of %s counts.", threshold)
        utr_sizes_exploded = utr_sizes.explode(
            utr_sizes.columns.to_list(), ignore_index=False
        )
        utr_sizes_exploded = utr_sizes_exploded[
            utr_sizes_exploded.utr_sizes_from_reads > threshold
        ]
        utr_sizes = utr_sizes_exploded.groupby(level=0).agg(list)
        return utr_sizes

    @staticmethod
    def check_strand_matches(read_strand, utr_strand):
        return read_strand != utr_strand

    @staticmethod
    def get_utr_size_from_read(read_start_position, interval):
        return abs(read_start_position - interval.start_d)


class ProtocolFivePrime(Protocol):

    @staticmethod
    def check_strand_matches(read_strand, utr_strand):
        return read_strand == utr_strand

    @staticmethod
    def filter_utr_sizes(utr_sizes, **kwargs):
        logger = module_log.module_logger()
        logger.info(
            """
        For 5'UTR protocols, no UTR size filtering is performed because having transcripts
        with 5'UTR size 0 means these transcripts start at the ATG site.
        """
        )
        return utr_sizes

    @staticmethod
    def get_utr_size_from_read(read_start_position, interval):
        return abs(read_start_position - interval.end_d)


class ProtocolTenx(ProtocolThreePrime):
    def __init__(
        self,
        merge_method="union",
        call_peaks=True,
        call_peaks_method="kde",
        label_batches=None,
        alignments=None,
        annotations=None,
    ) -> None:

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

    @staticmethod
    def get_reads_identity(alignment, **_):
        read_seq = alignment.read.seq
        read_barcode = helpers.to_string(read_seq[:16])
        read_umi = helpers.to_string(read_seq[16:28])  # 12 bp UMI
        return read_umi, read_barcode


class ProtocolTenx2(ProtocolTenx):

    @staticmethod
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
    def __init__(
        self,
        merge_method="union",
        call_peaks=True,
        call_peaks_method="kde",
        label_batches=None,
        alignments=None,
        annotations=None,
    ) -> None:

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
        try:
            self.sample_names, self.sample_idx = self.get_sample_names(
                alignments=alignments
            )
        except TypeError as err:
            if alignments is None:
                self.sample_names, self.sample_idx = None, None
            else:
                raise err


class ProtocolSmartseq3(ProtocolSmartseq):

    def get_reads_identity(self, alignment, **kwargs):
        bam_name = kwargs.get("bam_name")
        read_seq = alignment.read.seq

        if self.label_batches:
            read_sample_idx = self.sample_idx[bam_name]
            read_barcode = f"{helpers.to_string(read_seq[:6])}-{read_sample_idx}"
        else:
            read_barcode = helpers.to_string(read_seq[:6])

        read_umi = helpers.to_string(read_seq[6:14])  # 8 bp UMI
        return read_umi, read_barcode


class ProtocolCapseq(ProtocolFivePrime):
    def __init__(
        self,
        merge_method="union",
        call_peaks=True,
        call_peaks_method="kde",
        label_batches=None,
        alignments=None,
        annotations=None,
    ) -> None:

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
        try:
            self.sample_names, self.sample_idx = self.get_sample_names(
                alignments=alignments
            )
        except TypeError as err:
            if alignments is None:
                self.sample_names, self.sample_idx = None, None
            else:
                raise err


class ProtocolCapseqSmartseq3(ProtocolCapseq):

    @staticmethod
    def get_reads_identity(alignment, **kwargs):
        read_batch_id = kwargs.get("bam_name", "S")
        read_seq = helpers.to_string(alignment.read.seq)

        ss3_regex = "(ATTGCGCAATG)[NGCAT]{8}"
        umi_match = re.search(ss3_regex, read_seq)
        if umi_match:
            read_umi = umi_match.group(0)[11:19]  # 8 bp UMI
            assert len(read_umi) == 8  # 8 bp UMI
        else:
            return None, read_batch_id

        return read_umi, read_batch_id


class ProtocolCapseqSmartseq3ReadName(ProtocolCapseq):

    @staticmethod
    def get_reads_identity(alignment, **kwargs):
        read_batch_id = kwargs.get("bam_name", "S")

        read_seq = helpers.to_string(alignment.read.name)
        read_umi = read_seq.split("_")[1]
        assert len(read_umi) == 8  # 8 bp UMI

        return read_umi, read_batch_id


class ProtocolFiveTenx(ProtocolFivePrime):
    def __init__(
        self,
        merge_method="union",
        call_peaks=True,
        call_peaks_method="kde",
        label_batches=None,
        alignments=None,
        annotations=None,
    ) -> None:

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

    @staticmethod
    def get_reads_identity(alignment, **_):
        """
        Get read UMI and barcode from 5'10x version 1.1

        Args:
            alignment (HTSeq.SAM_Alignment): The genome alignment line with the read
            information.

        Returns:
            tuple: a tuple of strings, containing the the read UMI and barcode in
            position 0 and 1, respectively.
        """
        # The cell barcode CB tag includes a suffix with a dash separator followed by
        # a number. This number denotes the GEM well, and is used to virtualize
        # barcodes in order to achieve a higher effective barcode diversity when
        # combining samples generated from separate GEM well channel runs. Normally,
        # this number will be "1" across all barcodes when analyzing a sample generated
        # from a single GEM well channel. It can either be left in place and treated as
        # part of a unique barcode identifier, or explicitly parsed out to leave only
        # the barcode sequence itself.
        try:
            read_barcode = alignment.optional_field("CB")  # 16bp + "-<GEM well number>"
        except KeyError:
            read_barcode = False

        try:
            read_umi = alignment.optional_field("UB")  # 10bp
        except KeyError:
            read_umi = False
        return read_umi, read_barcode


def create(platform, version, **kwargs):
    error_message = f"Protocol for {platform} version {version} not implemented."

    if platform.lower() in ("10x", "10_x", "tenx", "ten_x"):
        if version == 2:
            return ProtocolTenx2(**kwargs)
        if version == 3:
            return ProtocolTenx3(**kwargs)
        raise NotImplementedError(error_message)
    if platform.lower() in ("smart-seq", "smartseq"):
        if version == 3:
            return ProtocolSmartseq3(**kwargs)
        raise NotImplementedError(error_message)
    if platform.lower() in ("smart-seq_cap", "smartseq_cap"):
        if version == 3:
            return ProtocolCapseqSmartseq3(**kwargs)
        raise NotImplementedError(error_message)
    if platform.lower() in ("smart-seq_cap-readname", "smartseq_cap-readname"):
        if version == 3:
            return ProtocolCapseqSmartseq3ReadName(**kwargs)
        raise NotImplementedError(error_message)
    if platform.lower() in ("5'10x", "5'10_x", "5'tenx", "5prime10x"):
        if version == 1:
            return ProtocolFiveTenx(**kwargs)
        raise NotImplementedError(error_message)
    raise NotImplementedError(error_message)
