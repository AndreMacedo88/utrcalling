"""
File containing objects and functions specific for each supported sequencing platform

"""

from os import read

from .module_log import module_logger
from .helpers import to_string


class Platform:
    def __init__(self, platform, version, merge_method='intercept') -> None:
        self.logger = module_logger()

        self.get_reads_identity = None
        self.merge_method = merge_method

        self.check_platform_and_version(platform=platform, version=version)

    def check_platform_and_version(self, platform, version):
        if platform.lower() in ("10x", "10_x", "tenx", "ten_x"):
            if version == 2:
                self.get_reads_identity = self.get_reads_identity_tenx_2
                self.merge_method = 'intercept'
            elif version in (3, 3.1):
                self.get_reads_identity = self.get_reads_identity_tenx_3
                self.merge_method = 'intercept'
            else:
                version_fail = f'10x genomics protocol version {version} currently not supported.'
                self.logger.error(version_fail)
                raise RuntimeError(version_fail)

        elif platform.lower() in ('smart-seq_cap', 'smartseq_cap'):
            if version == 3:
                self.get_reads_identity = self.get_reads_identity_smartseq_3_bulk
                self.merge_method = 'union'
            else:
                version_fail = f'Smart-seq protocol version {version} \
                    currently not supported'
                self.logger.error(version_fail)
                raise RuntimeError(version_fail)

        else:
            platform_fail = f'Sequencing platform - {platform} - currently not supported.'
            self.logger.error(platform_fail)
            raise RuntimeError(platform_fail)

    @staticmethod
    def get_reads_identity_tenx_2(alignment):
        read_barcode = to_string(alignment.read.seq[:16])
        read_umi = to_string(alignment.read.seq[17:26])  # 10 bp UMI
        return read_umi, read_barcode

    @staticmethod
    def get_reads_identity_tenx_3(alignment):
        read_barcode = to_string(alignment.read.seq[:16])
        read_umi = to_string(alignment.read.seq[17:28])  # 12 bp UMI
        return read_umi, read_barcode

    @staticmethod
    def get_reads_identity_smartseq_3_bulk(alignment):
        read_sample_id = 'S'

        read_seq = to_string(alignment.read.name)
        read_umi = read_seq.split('_')[1]
        assert(len(read_umi) == 8)  # 8 bp UMI

        return read_umi, read_sample_id
