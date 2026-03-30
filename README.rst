##########
utrcalling
##########

.. image:: https://img.shields.io/badge/python-%3E%3D3.11-blue
   :alt: Python >= 3.11

.. image:: https://img.shields.io/badge/license-BSD--3--Clause-green
   :alt: License: BSD-3-Clause

.. image:: https://img.shields.io/badge/version-3.1.0-orange
   :alt: Version 3.1.0

A bioinformatics toolkit for calculating molecule UTR (Untranslated Region)
sizes from RNA sequencing reads. **utrcalling** processes BAM alignment files
against GTF gene annotations to produce AnnData (``.h5ad``) count matrices of
UTR-length usage per cell or sample --- enabling the study of alternative
polyadenylation (APA) and alternative transcription initiation (ATI) at
single-cell resolution.

--------
Features
--------

- **Multi-protocol support** --- works with modified and standard versions of
  10x Chromium and SmartSeq3 sequencing, as well as CapSeq and 5' 10x
  protocols.
- **3'UTR and 5'UTR calling** --- analyse either end of the transcript.
- **Flexible summarisation** --- aggregate results at UTR, transcript, or gene
  level.
- **KDE peak calling** --- Gaussian kernel density estimation with configurable
  bandwidth to identify consensus UTR isoform peaks from per-molecule size
  distributions.
- **Batch-aware processing** --- union or intersection merge strategies for
  multi-sample experiments.
- **Sparse output** --- count matrices are stored as sparse arrays in AnnData
  objects, ready for downstream single-cell analysis with scanpy.
- **Downstream utilities** --- filter peaks, select APA/ATI events, compute
  mean UTR sizes, convert counts to percentages, concatenate or subset
  experiments, and more.

---------
Protocols
---------

utrcalling supports several sequencing protocols, each with tailored read
parsing, deduplication, and UTR size estimation logic:

.. list-table::
   :header-rows: 1
   :widths: 22 14 10 10 44

   * - Platform (``--protocol_platform``)
     - Read used
     - UMI
     - Dedup
     - Description
   * - ``10x``
     - Read 1
     - Yes
     - keep first
     - Modified 10x Chromium (extended Read 1 captures the 3'UTR terminus).
   * - ``standard_10x``
     - Read 2
     - Yes
     - keep max
     - Standard 3' 10x Chromium. Requires Cell Ranger BAM output with CB/UB
       tags. Keeps the largest UTR size per (barcode, UMI) pair.
   * - ``5prime10x``
     - Read 1
     - Yes
     - keep first
     - 5' 10x Chromium.
   * - ``smartseq``
     - Read 1
     - Yes
     - keep first
     - Modified SmartSeq3 (extended Read 1 captures the 3'UTR terminus).
   * - ``standard_smartseq``
     - Any
     - No
     - none
     - Standard SmartSeq3. One BAM per cell, no barcodes/UMIs. Accepts reads
       from both strands and uses the distal read position (closest to polyA).
   * - ``smart-seq_cap``
     - Read 1
     - Yes
     - keep first
     - CapSeq SmartSeq3.
   * - ``smart-seq_cap-readname``
     - Read 1
     - Yes
     - keep first
     - CapSeq SmartSeq3 with read-name UMI extraction.

Each platform is used with a ``--protocol_version`` (typically ``2`` or ``3``).

-----
Input
-----

1. **GTF annotation file** --- genome annotations from `Ensembl
   <http://ftp.ensembl.org/pub/>`_ (plain or gzipped).
2. **BAM alignment file(s)** --- coordinate-sorted and indexed. For
   ``standard_10x``, these must be Cell Ranger output BAMs containing ``CB``
   and ``UB`` tags.
3. **Annotations file** --- CSV with experiment metadata (cell barcodes, sample
   IDs, batch information). Required columns depend on the protocol.

------
Output
------

- **Primary output** --- ``.h5ad`` (AnnData) count matrix with cells/samples as
  observations and UTR peaks as variables.
- **Raw sizes mode** (``--raw_sizes``) --- produces two files: raw UTR sizes
  and KDE-estimated peaks (``*_kde.h5ad``).
- **Intermediate files** (``--store_intermediate``) --- ``.bed`` UTR reference
  and ``.parquet`` read mappings for resumable workflows.
- **Log files** --- automatically written to the output directory (suppress with
  ``--no_log_file``).

------------
Installation
------------

Prerequisites
=============

- Python >= 3.11 (tested on 3.12)
- `BEDTools <https://github.com/arq5x/bedtools2>`_ must be installed and
  available on your ``PATH``.

With conda (recommended)
========================

The provided ``environment.yml`` creates a conda environment with all
dependencies, including BEDTools:

.. code-block:: bash

   git clone https://github.com/AndreMacedo88/utrcalling.git
   cd utrcalling
   conda env create -f environment.yml
   conda activate utrcalling

This installs the package in editable mode automatically.

From source
===========

.. code-block:: bash

   git clone https://github.com/AndreMacedo88/utrcalling.git
   cd utrcalling
   pip install .

For development (editable install):

.. code-block:: bash

   pip install -e .

From PyPI
=========

Not yet available on PyPI.

-----
Usage
-----

utrcalling provides a command-line interface with multiple subcommands. Run
``utrcalling --help`` for a full list.

Full pipeline
=============

Run the entire pipeline --- from GTF parsing through read mapping to peak
calling --- in a single command:

.. code-block:: bash

   utrcalling run \
       genome_annotations.gtf.gz \
       --alignments sample1.bam sample2.bam \
       --annotations_file experiment_annotations.csv \
       --protocol_platform 10x \
       --protocol_version 3 \
       --out results.h5ad

Standard 3' 10x (Cell Ranger)
==============================

For standard 10x Chromium data processed through Cell Ranger:

.. code-block:: bash

   utrcalling run \
       genome_annotations.gtf.gz \
       --alignments cellranger_output/possorted_genome_bam.bam \
       --annotations_file annotations.csv \
       --protocol_platform standard_10x \
       --protocol_version 3 \
       --out standard_10x_results.h5ad

Standard SmartSeq3
==================

For standard SmartSeq3 data with one BAM per cell:

.. code-block:: bash

   utrcalling run \
       genome_annotations.gtf.gz \
       --alignments cell1.bam cell2.bam cell3.bam \
       --annotations_file annotations.csv \
       --protocol_platform standard_smartseq \
       --protocol_version 3 \
       --out smartseq3_results.h5ad

Step-by-step execution
======================

The pipeline can also be run in two stages for large datasets or debugging:

.. code-block:: bash

   # Stage 1 & 2: Build UTR reference and map reads
   utrcalling prepare-reference-and-map-reads \
       genome_annotations.gtf.gz \
       --alignments sample.bam \
       --protocol_platform 10x \
       --protocol_version 3

   # Stage 3: Call peaks from intermediate files
   utrcalling call-peaks \
       intermediate/utr_reference.parquet \
       --utr_sizes_counts intermediate/mapped_reads_*.parquet \
       --annotations_file annotations.csv \
       --protocol_platform 10x \
       --protocol_version 3 \
       --out peaks.h5ad

Key options
===========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--feature / -f``
     - UTR type: ``three_prime_utr`` (default) or ``five_prime_utr``.
   * - ``--summarization_level / -s``
     - Aggregate at ``UTR``, ``transcript``, or ``gene`` level.
   * - ``--raw_sizes / -r``
     - Output both raw UTR sizes and KDE peaks.
   * - ``--kde_bandwidth``
     - KDE bandwidth for peak calling (default: 12).
   * - ``--batch_merge_method / -m``
     - How to merge batches: ``union`` or ``intersection``.
   * - ``--store_intermediate / -i``
     - Save intermediate files for resumable workflows.
   * - ``--label_batches / -L``
     - Append batch index to cell barcodes.

Downstream processing
=====================

After running the main pipeline, several utility commands are available for
further analysis:

.. code-block:: bash

   # Remove low-count peaks
   utrcalling remove-peaks-below-threshold results.h5ad filtered.h5ad \
       --threshold 0.05 --minimum_counts 10

   # Keep only genes with multiple UTR isoforms (APA/ATI events)
   utrcalling select-alternative-transcription-events filtered.h5ad apa.h5ad

   # Generate human-readable UTR site names
   utrcalling generate-names apa.h5ad named.h5ad

   # Compute mean UTR sizes per gene
   utrcalling compute-mean-utr-sizes named.h5ad mean_sizes.h5ad

   # Convert counts to percentage usage per gene
   utrcalling counts-to-percentages named.h5ad percentages.h5ad

   # Concatenate experiments
   utrcalling concatenate-experiments \
       -i experiment1.h5ad experiment2.h5ad \
       -o combined.h5ad

   # Estimate optimal KDE bandwidth
   utrcalling estimate-best-bandwidth raw_sizes.h5ad bandwidth_scores.joblib \
       --bandwidths 1 7 12 15

   # Validate peaks against known polyadenylation signals
   utrcalling check-distance-to-pas results.h5ad distances.joblib genome.fa

-------
Testing
-------

The test suite contains 359 tests covering all protocols, edge cases, and the
full pipeline end-to-end.

.. code-block:: bash

   # Run all tests via the CLI entry point
   utrcalling run-tests

   # Or via unittest directly
   python -m unittest discover -s utrcalling/tests -p "test_*.py"

   # Run a single test module
   python -m unittest utrcalling.tests.test_core.test_pipeline

   # Run a single test class
   python -m unittest utrcalling.tests.test_core.test_pipeline.TestCallPeaksSmartseq3LabelMinimal

-----------------
Project structure
-----------------

::

   utrcalling/
   +-- utrcalling/
   |   +-- runtime.py              CLI entry point
   |   +-- core/
   |   |   +-- core.py             Pipeline orchestrator
   |   |   +-- protocol.py         Sequencing protocol definitions
   |   |   +-- utr_mapping.py      Read-to-UTR mapping (pysam)
   |   |   +-- utr_reference.py    GTF parsing, UTR extraction
   |   |   +-- count_matrix.py     Sparse count matrix assembly
   |   |   +-- barcode_filtering.py
   |   |   +-- process_mapped_reads.py
   |   |   +-- storage.py
   |   |   +-- ...
   |   +-- tools/                   CLI subcommand implementations
   |   +-- tests/                   359 unit and integration tests
   |   +-- notebooks/               Example notebooks
   +-- setup.py
   +-- environment.yml
   +-- README.rst

------------
Dependencies
------------

Core Python dependencies (installed automatically via ``pip``):

- **pysam** (>= 0.20.0) --- BAM file reading
- **pybedtools** (>= 0.8.2) --- genomic interval operations (requires BEDTools)
- **anndata** (>= 0.12.0) / **scanpy** (>= 1.11.0) --- single-cell data structures
- **pandas** (>= 1.3.0) / **numpy** (>= 1.22.3) / **scipy** (>= 1.6.2)
- **joblib** (>= 1.0.1) --- parallelism
- **gtfparse** (>= 1.2.1, < 2) --- GTF file parsing
- **rapidfuzz** (>= 2.0.0) --- fuzzy barcode matching
- **scikit-learn** (>= 0.23.2) --- used by KDE peak calling
- **pyarrow** --- Parquet I/O

System dependency:

- **BEDTools** (>= 2.30.0) --- install via ``conda install -c bioconda bedtools``

-------
License
-------

BSD-3-Clause

------
Author
------

Andre Lopes Martins Macedo (`andre.lopes.macedo@gmail.com <mailto:andre.lopes.macedo@gmail.com>`_)

Source code: https://github.com/AndreMacedo88/utrcalling
