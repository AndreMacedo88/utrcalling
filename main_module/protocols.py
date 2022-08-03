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

from abc import ABC, ABCMeta, abstractmethod
import re
from distutils.util import strtobool

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from main_module import module_log, helpers


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

    def __init__(self,
                 merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_samples=None, sample_names=None
                 ) -> None:
        self.logger = module_log.module_logger()

        self.merge_method = merge_method
        self.sample_names = sample_names
        self.label_samples = label_samples
        if isinstance(call_peaks, bool):
            self.call_peaks = call_peaks
        else:
            self.call_peaks = bool(strtobool(call_peaks))

        if self.call_peaks:
            if call_peaks_method.lower() == "kde":
                self.peak_calling = self._call_peaks_using_kernel_density_estimator
            else:
                raise NotImplementedError(
                    f"Method {call_peaks_method} for calling consensus peaks is not implemented."
                )
        else:
            self.peak_calling = self._bypass_calling_peaks

    # log the protocol:
    def protocol_log(self):
        self.logger.info(f'Protocol args:\n')
        self.logger.info(f'Position counts merging method: {self.merge_method}\n')
        self.logger.info(f'Call peaks? {self.call_peaks}\n')
        self.logger.info(f'Peak calling method: {self.peak_calling}\n')
        self.logger.info(
            f'Add a number to sample names while merging samples? {self.label_samples}\n'
        )

    # Methods to get read information from the read alignments
    @staticmethod
    @abstractmethod
    def get_reads_identity(alignment, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def check_strand_matches(alignment, **kwargs):
        pass

    # Methods to process UTR sizes
    @staticmethod
    @abstractmethod
    def filter_utr_sizes(alignment, **kwargs):
        pass

    # Methods to aggregate individual molecules' UTR sizes to consensus peaks:
    @staticmethod
    def _call_peaks_using_kernel_density_estimator(utr_sizes_vector, bandwidth=1):
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
        padding = (maximum_size - minimum_size) * 0.1
        window = np.arange(round(minimum_size - padding),
                           round(maximum_size + padding) + 1)
        # get the model predictions for that window. Specifically, the maxima.
        evaluation = kde.score_samples(window.reshape(-1, 1))
        minima = argrelextrema(evaluation, np.less)[0]
        maxima = argrelextrema(evaluation, np.greater)[0]

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

    @staticmethod
    def _bypass_calling_peaks(utr_sizes_vector):
        return np.rint(utr_sizes_vector)


class ProtocolThreePrime(Protocol):

    @staticmethod
    def check_strand_matches(read_strand, utr_strand):
        return read_strand != utr_strand

    @staticmethod
    def filter_utr_sizes(utr_sizes, amount=1):

        def filter_func(x, amount_):
            condition = [True if i >
                         amount_ else False for i in x['utr_sizes_from_reads']]
            return condition

        logger = module_log.module_logger()
        logger.info('Filtering samples UTR sizes')
        filtered = utr_sizes.filter(
            lambda x: x.filter(filter_func, amount))
        return filtered


class ProtocolFivePrime(Protocol):

    @staticmethod
    def check_strand_matches(read_strand, utr_strand):
        return read_strand == utr_strand

    @staticmethod
    def filter_utr_sizes(utr_sizes, **kwargs):
        logger = module_log.module_logger()
        logger.info("For 5'UTR protocols, no UTR size filtering is performed.")
        return utr_sizes


class ProtocolTenx(ProtocolThreePrime):
    def __init__(self, merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_samples=None, sample_names=None) -> None:
        super().__init__(merge_method, call_peaks, call_peaks_method, label_samples,
                         sample_names)

        if self.label_samples is None:
            self.label_samples = True


class ProtocolTenx3(ProtocolTenx):

    @staticmethod
    def get_reads_identity(alignment, **_):
        read_barcode = helpers.to_string(alignment.read.seq[:16])
        read_umi = helpers.to_string(alignment.read.seq[16:28])  # 12 bp UMI
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
        read_barcode = helpers.to_string(alignment.read.seq[:16])
        read_umi = helpers.to_string(alignment.read.seq[16:26])  # 10 bp UMI
        return read_umi, read_barcode


class ProtocolSmartseq(ProtocolThreePrime):
    def __init__(self, merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_samples=None, sample_names=None) -> None:
        super().__init__(merge_method, call_peaks, call_peaks_method, label_samples,
                         sample_names)

        if self.label_samples is None:
            self.label_samples = False


class ProtocolSmartseq3(ProtocolSmartseq):

    @staticmethod
    def get_reads_identity(alignment, **kwargs):
        read_sample_id = kwargs.get('sample', 'S')
        read_umi = helpers.to_string(alignment.read.seq[6:14])  # 8 bp UMI
        return read_umi, read_sample_id


class ProtocolCapseq(ProtocolFivePrime):
    def __init__(self, merge_method="union", call_peaks=True, call_peaks_method="kde",
                 label_samples=None, sample_names=None) -> None:
        super().__init__(merge_method, call_peaks, call_peaks_method, label_samples,
                         sample_names)

        if self.label_samples is None:
            self.label_samples = False


class ProtocolCapseqSmartseq3(ProtocolCapseq):

    @staticmethod
    def get_reads_identity(alignment, **kwargs):
        read_sample_id = kwargs.get('sample', 'S')

        read_seq = helpers.to_string(alignment.read.name)
        read_umi = read_seq.split('_')[1]
        assert(len(read_umi) == 8)  # 8 bp UMI

        return read_umi, read_sample_id



def protocol(platform, version, *args, **kwargs):
    error_message = f"Protocol for {platform} version {version} not implemented."

    if platform.lower() in ("10x", "10_x", "tenx", "ten_x"):
        if version == 2:
            return ProtocolTenx2(*args, **kwargs)
        if version == 3:
            return ProtocolTenx3(*args, **kwargs)
        else:
            raise NotImplementedError(error_message)
    elif platform.lower() in ('smart-seq', 'smartseq'):
        if version == 3:
            return ProtocolSmartseq3(*args, **kwargs)
        else:
            raise NotImplementedError(error_message)
    elif platform.lower() in ('smart-seq_cap', 'smartseq_cap'):
        if version == 3:
            return ProtocolCapseqSmartseq3(*args, **kwargs)
        else:
            raise NotImplementedError(error_message)
    else:
        raise NotImplementedError(error_message)
