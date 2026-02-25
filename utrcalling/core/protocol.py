"""
File containing protocols: objects and functions specific for each supported
sequencing platform, version, strategy, etc.

Each protocol should define functions and variables that are expected to be called during
the run.

Note:
1- These functions are defined as class extensions at the end of this file.
2- New protocols should, then, extend the necessary class in this file.

"""

from __future__ import annotations

# built-in packages
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
import os
import re

# data wrangling packages
import numpy as np
import pandas as pd
from bidict import bidict

# internal packages
from utrcalling.core import module_log, helpers
from utrcalling.core.protocols import peak_calling as pc


class MergeMethod(Enum):
    """Method for merging UTRs across samples."""

    UNION = "union"
    INTERSECTION = "intersection"

    @classmethod
    def from_str(cls, value: str | None) -> "MergeMethod":
        """Create from string, defaulting to UNION if None."""
        if value is None:
            return cls.UNION
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid merge method {value!r}. "
                f"Must be one of: {', '.join(m.value for m in cls)}"
            )


class PeakCallingMethod(Enum):
    """Algorithm for calling consensus peaks."""

    KDE = "kde"
    BYPASS = "bypass"

    @classmethod
    def from_str(cls, value: str) -> "PeakCallingMethod":
        """Create from string."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid peak calling method {value!r}. "
                f"Must be one of: {', '.join(m.value for m in cls)}"
            )


def _strtobool(val: str | bool) -> bool:
    """Convert a string representation of truth to a bool.

    Replacement for the deprecated distutils.util.strtobool (removed in Python 3.12).
    """
    val = str(val).lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


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

    # Default label_batches value — subclasses override this class attribute
    _default_label_batches: bool | None = None

    def __init__(
        self,
        annotations: str | pd.DataFrame | None = None,
        merge_method: str = "union",
        call_peaks: bool = True,
        call_peaks_method: str = "kde",
        label_batches: bool | str | None = None,
        alignments: str | list[str] | None = None,
    ) -> None:
        """Initialize the protocol with pipeline configuration.

        Args:
            annotations: Path to a CSV annotations file, or a pre-loaded DataFrame.
            merge_method: How to merge UTRs across samples ('union' or 'intersection').
            call_peaks: Whether to call consensus peaks on UTR size distributions.
            call_peaks_method: Peak calling algorithm ('kde' for kernel density estimator).
            label_batches: Whether to append batch indices to observation IDs.
                Accepts bool or string representations of bool.
            alignments: BAM file path(s) for sample name resolution.
        """
        self.logger = module_log.module_logger()

        # set variables from the arguments
        self.annotations = helpers.check_if_path_and_open(
            to_check=annotations, method="csv"
        )
        try:
            self.annotations = self.annotations.dropna(axis=0, how="all")
        except TypeError:
            pass

        # Validate and store merge method
        self._merge_method_enum = MergeMethod.from_str(merge_method)
        self.merge_method = self._merge_method_enum.value

        if isinstance(label_batches, bool) or label_batches is None:
            self.label_batches = label_batches
        else:
            self.label_batches = _strtobool(label_batches)

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

        # Apply subclass default if label_batches was not explicitly set.
        # Only apply when batch information is actually available, to avoid
        # raising errors when creating protocols without batch annotations.
        if self.label_batches is None:
            if self.batch_column_name is not None:
                self.label_batches = self._default_label_batches
            else:
                self.label_batches = False

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
            # Search the full path (filename *and* all parent directories) so
            # the sample ID is found whether it is embedded in the filename or
            # is the parent directory name (e.g. STAR output).  Token-boundary
            # delimiters (/, _, .) prevent prefix-collision false matches.
            sample_bam = [
                bam for bam in alignments if helpers.sample_in_path(sample, bam)
            ]
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
            self.annotations.reset_index()
            .columns.str.extract(r"(.*barcode.*)", flags=re.IGNORECASE)
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

    # Observation identity
    def get_observation_id(self, barcode: str, batch_idx: str | None = None) -> str:
        """Return the final observation ID, applying batch suffix if needed.

        Args:
            barcode: The cell/sample barcode string.
            batch_idx: Optional batch index to append as a suffix.

        Returns:
            The observation ID, optionally suffixed with '-{batch_idx}'.
        """
        if self.label_batches and batch_idx is not None:
            return f"{barcode}-{batch_idx}"
        return barcode

    # Methods to get read information from the read alignments
    @staticmethod
    @abstractmethod
    def get_reads_identity(
        alignment: object, **kwargs: str
    ) -> tuple[str | None, str | None]:
        """Extract UMI and barcode from a read alignment.

        Args:
            alignment: A read alignment object (pysam adapter).
            **kwargs: Protocol-specific keyword arguments (e.g., bam_name).

        Returns:
            A (umi, barcode) tuple. Either value may be None if extraction fails.
        """

    @staticmethod
    @abstractmethod
    def check_strand_matches(read_strand: str, utr_strand: str) -> bool:
        """Check whether a read's strand matches the UTR interval strand.

        Args:
            read_strand: Strand of the read alignment ('+' or '-').
            utr_strand: Strand of the UTR interval ('+' or '-').

        Returns:
            True if the strand combination is valid for this protocol.
        """

    @staticmethod
    @abstractmethod
    def get_utr_size_from_read(read_start_position: int, interval: object) -> int:
        """Compute the UTR size from a read's start position and the UTR interval.

        Args:
            read_start_position: Directional start position of the read.
            interval: An _IntervalData object for the UTR region.

        Returns:
            The computed UTR size in base pairs.
        """

    # Methods to process UTR sizes
    @staticmethod
    @abstractmethod
    def filter_utr_sizes(utr_sizes: pd.DataFrame, **kwargs: int) -> pd.DataFrame:
        """Filter UTR size observations based on protocol-specific criteria.

        Args:
            utr_sizes: Flat DataFrame with columns (utr_name, observation_id, utr_size).
            **kwargs: Protocol-specific arguments (e.g., threshold).

        Returns:
            Filtered DataFrame.
        """


class ProtocolThreePrime(Protocol):
    """Protocol base for 3' UTR sequencing platforms."""

    def filter_utr_sizes(
        self, utr_sizes: pd.DataFrame, threshold: int = 1
    ) -> pd.DataFrame:
        self.logger.info("Filtering UTR sizes using threshold of %s counts.", threshold)
        return utr_sizes[utr_sizes["utr_size"] > threshold].copy()

    @staticmethod
    def check_strand_matches(read_strand: str, utr_strand: str) -> bool:
        return read_strand != utr_strand

    @staticmethod
    def get_utr_size_from_read(read_start_position: int, interval: object) -> int:
        return abs(read_start_position - interval.start_d)


class ProtocolFivePrime(Protocol):
    """Protocol base for 5' UTR sequencing platforms."""

    @staticmethod
    def check_strand_matches(read_strand: str, utr_strand: str) -> bool:
        return read_strand == utr_strand

    @staticmethod
    def filter_utr_sizes(utr_sizes: pd.DataFrame, **kwargs: int) -> pd.DataFrame:
        logger = module_log.module_logger()
        logger.info(
            """
        For 5'UTR protocols, no UTR size filtering is performed because having transcripts
        with 5'UTR size 0 means these transcripts start at the ATG site.
        """
        )
        return utr_sizes

    @staticmethod
    def get_utr_size_from_read(read_start_position: int, interval: object) -> int:
        return abs(read_start_position - interval.end_d)


class ProtocolTenx(ProtocolThreePrime):
    _default_label_batches = True


class ProtocolTenx3(ProtocolTenx):
    """10x Chromium v3 protocol (16 bp barcode + 12 bp UMI from read sequence)."""

    @staticmethod
    def get_reads_identity(alignment: object, **_: str) -> tuple[str, str]:
        read_seq = alignment.read.seq
        read_barcode = helpers.to_string(read_seq[:16])
        read_umi = helpers.to_string(read_seq[16:28])  # 12 bp UMI
        return read_umi, read_barcode


class ProtocolTenx2(ProtocolTenx):
    """10x Chromium v2 protocol (16 bp barcode + 10 bp UMI from read sequence)."""

    @staticmethod
    def get_reads_identity(alignment: object, **_: str) -> tuple[str, str]:
        """Get read UMI and barcode from 10x version 2.

        Args:
            alignment: A read alignment object (pysam adapter).

        Returns:
            A (umi, barcode) tuple of strings.
        """
        read_seq = alignment.read.seq
        read_barcode = helpers.to_string(read_seq[:16])
        read_umi = helpers.to_string(read_seq[16:26])  # 10 bp UMI
        return read_umi, read_barcode


class ProtocolSmartseq(ProtocolThreePrime):
    _default_label_batches = False

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
    """Smart-seq3 protocol (6 bp barcode + 8 bp UMI from read sequence)."""

    @staticmethod
    def get_reads_identity(alignment: object, **_: str) -> tuple[str, str]:
        read_seq = alignment.read.seq
        read_barcode = helpers.to_string(read_seq[:6])
        read_umi = helpers.to_string(read_seq[6:14])  # 8 bp UMI
        return read_umi, read_barcode


class ProtocolCapseq(ProtocolFivePrime):
    _default_label_batches = False

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
    """CapSeq + Smart-seq3 protocol (regex UMI extraction, BAM path as identity)."""

    def get_observation_id(self, barcode: str, batch_idx: str | None = None) -> str:
        """CapSeq uses the BAM-derived sample name as identity directly."""
        return barcode

    @staticmethod
    def get_reads_identity(alignment: object, **kwargs: str) -> tuple[str | None, str]:
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
    """CapSeq + Smart-seq3 protocol (UMI from read name, BAM path as identity)."""

    def get_observation_id(self, barcode: str, batch_idx: str | None = None) -> str:
        """CapSeq uses the BAM-derived sample name as identity directly."""
        return barcode

    @staticmethod
    def get_reads_identity(alignment: object, **kwargs: str) -> tuple[str, str]:
        read_batch_id = kwargs.get("bam_name", "S")

        read_seq = helpers.to_string(alignment.read.name)
        read_umi = read_seq.split("_")[1]
        assert len(read_umi) == 8  # 8 bp UMI

        return read_umi, read_batch_id


class ProtocolFiveTenx(ProtocolFivePrime):
    _default_label_batches = True

    @staticmethod
    def get_reads_identity(
        alignment: object, **_: str
    ) -> tuple[str | None, str | None]:
        """Get read UMI and barcode from 5' 10x version 1.1 via BAM tags.

        Args:
            alignment: A read alignment object (pysam adapter) with CB/UB tags.

        Returns:
            A (umi, barcode) tuple. Either value is None if the BAM tag is missing.
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
            read_barcode = None

        try:
            read_umi = alignment.optional_field("UB")  # 10bp
        except KeyError:
            read_umi = None
        return read_umi, read_barcode


def create(platform: str, version: int, **kwargs) -> Protocol:
    """Create a Protocol instance for the given sequencing platform and version.

    Args:
        platform: Sequencing platform identifier (e.g., '10x', 'smartseq').
        version: Protocol version number.
        **kwargs: Passed to the Protocol constructor (annotations, merge_method, etc.).

    Returns:
        A Protocol subclass instance configured for the specified platform.

    Raises:
        NotImplementedError: If the platform/version combination is not supported.
    """
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
