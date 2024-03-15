import os
from tempfile import mkstemp


class GtfEnsembl:
    def __init__(self) -> None:
        self.fd, self.file_path = mkstemp(suffix=".gtf", dir="./test_data/")

    def __enter__(self):
        """
        Note: GTF files are made of fields separated by tabs, but then the last field,
        the attributes, which can be composed of multiple attributes, must separate the
        multiple attributes by one space.
        """

        header = (
            "#!genome-build test_genome.p6\n"
            "#!genome-version test_genome\n"
            "#!genome-date 2023-05\n"
            "#!genome-build-accession NCBI:test_accession.8\n"
            "#!genebuild-last-updated 2023-05\n"
        )

        feature_1 = (
            "1	ensembl	three_prime_utr	1000	1100	.	+	0	"
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST1"; transcript_version "4"; '
            'gene_name "Test1"; transcript_name "Test1-201"; '
            'protein_id "ENSMUSP1"; protein_version "6";\n'
        )

        feature_2 = (
            "1	ensembl	three_prime_utr	1200	1350	.	+	0	"
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST1"; transcript_version "4"; '
            'gene_name "Test1"; transcript_name "Test1-201"; '
            'protein_id "ENSMUSP1"; protein_version "6";\n'
        )

        feature_3 = (
            "1	ensembl	three_prime_utr	1000	1100	.	+	0	"
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST2"; transcript_version "1"; '
            'gene_name "Test1"; transcript_name "Test1-202"; '
            'protein_id "ENSMUSP2"; protein_version "2";\n'
        )

        feature_4 = (
            "1	ensembl	three_prime_utr	1150	1300	.	+	0	"
            'gene_id "ENSMUSG1"; gene_version "5"; '
            'transcript_id "ENSMUST2"; transcript_version "1"; '
            'gene_name "Test1"; transcript_name "Test1-202"; '
            'protein_id "ENSMUSP2"; protein_version "2";\n'
        )

        feature_5 = (
            "1	ensembl	three_prime_utr	2400	2600	.	-	0	"
            'gene_id "ENSMUSG2"; gene_version "2"; '
            'transcript_id "ENSMUST3"; transcript_version "4"; '
            'gene_name "Test2"; transcript_name "Test2-201"; '
            'protein_id "ENSMUSP3"; protein_version "3";\n'
        )

        feature_6 = (
            "1	ensembl	three_prime_utr	2000	2200	.	-	0	"
            'gene_id "ENSMUSG2"; gene_version "2"; '
            'transcript_id "ENSMUST3"; transcript_version "4"; '
            'gene_name "Test2"; transcript_name "Test2-201"; '
            'protein_id "ENSMUSP3"; protein_version "3";\n'
        )

        feature_7 = (
            "1	ensembl	three_prime_utr	2000	2100	.	-	0	"
            'gene_id "ENSMUSG2"; gene_version "2"; '
            'transcript_id "ENSMUST4"; transcript_version "3"; '
            'gene_name "Test2"; transcript_name "Test2-202"; '
            'protein_id "ENSMUSP4"; protein_version "2";\n'
        )

        feature_8 = (
            "1	ensembl	three_prime_utr	3000	3200	.	+	0	"
            'gene_id "ENSMUSG3"; gene_version "3"; '
            'transcript_id "ENSMUST5"; transcript_version "5"; '
            'gene_name "Test3"; transcript_name "Test3-201"; '
            'protein_id "ENSMUSP5"; protein_version "1";\n'
        )

        feature_9 = (
            "1	ensembl	three_prime_utr	500	600	.	+	0	"
            'gene_id "ENSMUSG4"; gene_version "1"; '
            'transcript_id "ENSMUST6"; transcript_version "2"; '
            'gene_name "Test4"; transcript_name "Test4-201"; '
            'protein_id "ENSMUSP6"; protein_version "1";\n'
        )

        feature_10 = (
            "1	ensembl	three_prime_utr	1200	1300	.	+	0	"
            'gene_id "ENSMUSG4"; gene_version "1"; '
            'transcript_id "ENSMUST6"; transcript_version "2"; '
            'gene_name "Test4"; transcript_name "Test4-201"; '
            'protein_id "ENSMUSP6"; protein_version "1";\n'
        )

        feature_11 = (
            "1	ensembl	three_prime_utr	500	650	.	-	0	"
            'gene_id "ENSMUSG5"; gene_version "7"; '
            'transcript_id "ENSMUST7"; transcript_version "7"; '
            'gene_name "Test5"; transcript_name "Test5-201"; '
            'protein_id "ENSMUSP7"; protein_version "1";\n'
        )

        feature_12 = (
            "1	ensembl	three_prime_utr	1200	1300	.	-	0	"
            'gene_id "ENSMUSG5"; gene_version "7"; '
            'transcript_id "ENSMUST7"; transcript_version "7"; '
            'gene_name "Test5"; transcript_name "Test5-201"; '
            'protein_id "ENSMUSP7"; protein_version "1";\n'
        )

        features = (
            feature_1
            + feature_2
            + feature_3
            + feature_4
            + feature_5
            + feature_6
            + feature_7
            + feature_8
            + feature_9
            + feature_10
            + feature_11
            + feature_12
        )

        with os.fdopen(self.fd, "w") as f:
            f.writelines(header + features)

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.file_path)
