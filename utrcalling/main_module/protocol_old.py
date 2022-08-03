import re
from distutils.util import strtobool

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from utrcalling.main_module import module_log, helpers

class Protocol:
    def __init__(self,
                 platform, version, merge_method=None, call_peaks=True,
                 label_samples=None, sample_names=None
                 ) -> None:
        self.logger = module_log.module_logger()

        # Variables that can be supplied directly in init, or modified via specific protocol functions
        self.merge_method = merge_method
        self.sample_names = sample_names
        self.label_samples = label_samples
        if isinstance(call_peaks, bool):
            self.call_peaks = call_peaks
        else:
            self.call_peaks = bool(strtobool(call_peaks))

        # Objects (functions) to be defined by this class only
        self.get_reads_identity = None
        self.check_strand_matches = None
        self.filter_utr_sizes = None

        # Since we only have one peak calling algorithm so far, we can define it here.
        if self.call_peaks:
            self.peak_calling = self._call_peaks_using_kernel_density_estimator
        else:
            self.peak_calling = self._bypass_calling_peaks

        # Call the protocol that will define and/or change the objects
        self._check_platform_and_version(platform=platform, version=version)

        self.logger.info(f'Protocol args:\n')
        self.logger.info(f'Function to get read information: {self.get_reads_identity}\n')
        self.logger.info(
            f'Function to check if strand matches: {self.check_strand_matches}\n')
        self.logger.info(f'Position counts merging method: {self.merge_method}\n')
        self.logger.info(f'Peak calling method: {self.peak_calling}\n')
        self.logger.info(
            f'Add number to sample names while merging samples? {self.label_samples}\n'
        )

    def _check_platform_and_version(self, platform, version):
        if platform.lower() in ("10x", "10_x", "tenx", "ten_x"):
            self.check_strand_matches = self._check_strand_matches_3prime
            self.filter_utr_sizes = self._filter_utr_sizes
            self.label_samples = True
            if self.merge_method is None:
                self.merge_method = 'intersection'
            if version == 2:
                self.get_reads_identity = self._get_reads_identity_tenx_2
            elif version in (3, 3.1):
                self.get_reads_identity = self._get_reads_identity_tenx_3
            else:
                version_fail = f'10x genomics protocol version {version} currently not supported.'
                self.logger.error(version_fail)
                raise RuntimeError(version_fail)

        elif platform.lower() in ('smart-seq', 'smartseq'):
            self.check_strand_matches = self._check_strand_matches_3prime
            self.filter_utr_sizes = self._filter_utr_sizes
            self.label_samples = False
            if self.merge_method is None:
                self.merge_method = 'intersection'
            if version == 3:
                self.get_reads_identity = self._get_reads_identity_smart_seq_3
            else:
                version_fail = f'Smart-seq protocol version {version} \
                    currently not supported'
                self.logger.error(version_fail)
                raise RuntimeError(version_fail)

        elif platform.lower() in ('smart-seq_cap', 'smartseq_cap'):
            self.check_strand_matches = self._check_strand_matches_5prime
            self.filter_utr_sizes = self._filter_utr_sizes_smart_seq_3_cap
            self.label_samples = False
            if self.merge_method is None:
                self.merge_method = 'union'
            if version == 3:
                self.get_reads_identity = self._get_reads_identity_smart_seq_3_cap
            else:
                version_fail = f'Smart-seq protocol version {version} \
                    currently not supported'
                self.logger.error(version_fail)
                raise RuntimeError(version_fail)

        elif platform.lower() in ("cellranger_vdj", "cr_vdj", "cellranger-vdj", "cr-vdj"):
            self.check_strand_matches = self._check_strand_matches_5prime
            self.filter_utr_sizes = self._filter_utr_sizes_smart_seq_3_cap
            self.label_samples = False
            if self.merge_method is None:
                self.merge_method = "intersection"
            self.get_reads_identity = self._get_reads_identity_cellranger

        # Implement any new protocol here, as a new elif clause.
        # TODO: refactor this whole class. Make individual protocols inherit an
        # AbstractProtocol that provides the interface needed for a valid
        # protocol and then define the necessary stuff in each subclass

        else:
            platform_fail = f'Sequencing platform - {platform} - currently not supported.'
            self.logger.error(platform_fail)
            raise RuntimeError(platform_fail)

    # Implement your private functions below.

    # 1- Methods to get read information from the alignment file:

    @staticmethod
    def _get_reads_identity_cellranger(alignment, **_):
        """
        Get read UMI and barcode from a cellranger output bam.

        Cellranger bam files contain sam tags with the error-corrected read barcodes and UMIs
        """
        read_barcodes = filter(lambda x: x[0] == "CB", alignment.optional_fields)
        read_umis = filter(lambda x: x[0] == "UB", alignment.optional_fields)
        # these tags can be duplicated, we only want the first one
        try:
            return next(read_umis)[1], next(read_barcodes)[1]
        except StopIteration:
            return None, None

    @staticmethod
    def _get_reads_identity_tenx_2(alignment, **_):
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

    @staticmethod
    def _get_reads_identity_tenx_3(alignment, **_):
        read_barcode = helpers.to_string(alignment.read.seq[:16])
        read_umi = helpers.to_string(alignment.read.seq[16:28])  # 12 bp UMI
        return read_umi, read_barcode

    @staticmethod
    def _get_reads_identity_smart_seq_3(alignment, **kwargs):
        read_sample_id = kwargs.get('sample', 'S')
        read_umi = helpers.to_string(alignment.read.seq[6:14])  # 8 bp UMI
        return read_umi, read_sample_id

    @staticmethod
    def _get_reads_identity_smart_seq_3_cap(alignment, **kwargs):
        read_sample_id = kwargs.get('sample', 'S')

        read_seq = helpers.to_string(alignment.read.name)
        read_umi = read_seq.split('_')[1]
        assert(len(read_umi) == 8)  # 8 bp UMI

        return read_umi, read_sample_id

    @staticmethod
    def _check_strand_matches_3prime(read_strand, utr_strand):
        return read_strand != utr_strand

    @staticmethod
    def _check_strand_matches_5prime(read_strand, utr_strand):
        return read_strand == utr_strand

    # 2- Methods to process UTR sizes:
    @staticmethod
    def _filter_utr_sizes(utr_sizes, amount=1):

        def filter_func(x, amount_):
            condition = [True if i >
                         amount_ else False for i in x['utr_sizes_from_reads']]
            return condition

        logger = module_log.module_logger()
        logger.info('Filtering samples UTR sizes')
        filtered = utr_sizes.filter(
            lambda x: x.filter(filter_func, amount))
        return filtered

    @staticmethod
    def _filter_utr_sizes_smart_seq_3_cap(utr_sizes, amount=0):
        logger = module_log.module_logger()
        logger.info('For this protocol, no UTR size filtering is performed.')
        return utr_sizes

    # 3- Methods to aggregate individual molecules' UTR sizes to consensus peaks:

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
        return np.around(utr_sizes_vector, decimals=0)
