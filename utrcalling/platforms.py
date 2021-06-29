"""
File containing objects and functions specific for each supported sequencing platform

"""

from .module_log import module_logger
from .helpers import to_string


class Platform:
    def __init__(self, platform, version) -> None:
        self.logger = module_logger()

        self.get_reads_identity = None

        self.check_platform_and_version(platform=platform, version=version)

    def check_platform_and_version(self, platform, version):
        if platform.lower() in ("10x", "10_x", "tenx", "ten_x"):
            if version == 2:
                self.get_reads_identity = self.get_reads_identity_tenx_2
            elif version in (3, 3.1):
                self.get_reads_identity = self.get_reads_identity_tenx_3
            else:
                version_fail = f'10x genomics protocol version {version} currently not supported.'
                self.logger.error(version_fail)
                raise RuntimeError(version_fail)

        elif platform.lower() in ('smart-seq', 'smartseq'):
            if version == 3:
                self.get_reads_identity = self.get_reads_identity_smartseq_3
            elif version == 2:
                pass
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
    def get_reads_identity_smartseq_3(alignment):
        read_barcode = "test"
        read_umi = to_string(alignment.read.seq[17:24])  # 8 bp UMI
        return read_umi, read_barcode
