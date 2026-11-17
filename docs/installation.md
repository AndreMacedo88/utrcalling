# Installation

utrcalling is a Python package with a command line, `utrcalling`. It needs
Python 3.11 or later, on Linux or macOS, and nothing outside Python.

## In a conda environment (recommended)

Make an environment for utrcalling, and install it from PyPI:

```bash
conda create -n utrcalling -c conda-forge python=3.12
conda activate utrcalling
pip install utrcalling
utrcalling --version
```

A dedicated environment keeps utrcalling's dependencies from clashing with
those of your other tools. Install the packages of your analyses, such as
scanpy, into it too, or read utrcalling's `.h5ad` files from another
environment.

## Into an existing environment

utrcalling declares compatible ranges of its dependencies, so it can share an
environment with other tools:

```bash
pip install utrcalling
```

## From the source code

To use changes that are not released yet, install the repository in editable
mode, with the latest compatible versions of the dependencies:

```bash
git clone https://github.com/AndreMacedo88/utrcalling.git
cd utrcalling
conda env create -f environment.yml
conda activate utrcalling
```

## A frozen environment, for reproducibility

`envs/utrcalling-frozen.yml` pins every package to the versions the test suite
passed with. Use it if the installs above fail or behave differently, for
example after breaking releases of dependencies years from now, or to repeat
an analysis exactly:

```bash
conda env create -f envs/utrcalling-frozen.yml
conda activate utrcalling
pip install --group test   # pytest, to check the installation
pytest
```

## Checking the installation

`utrcalling --version` prints the version, and `utrcalling --help` the
commands. The test suite runs from a clone of the repository: `pip install
--group test`, then `pytest` (a few minutes).

## Dependencies

pip installs them. Their compatible ranges are in `pyproject.toml`, and the
versions the tests passed with in `envs/utrcalling-frozen.yml`. The main ones:

- **pysam** --- reading BAM and genome FASTA files
- **anndata** --- the count matrices
- **pandas**, **numpy** and **scipy** --- tables, GTF files and the merging of
  UTR regions
- **scikit-learn** --- the KDE of peak calling
- **joblib** --- parallel work
- **rapidfuzz** --- fuzzy barcode matching
- **pyarrow** --- Parquet files
- **cyclopts** and **rich** --- the command line and its messages

Before version 5.1.0, utrcalling also needed BEDTools, pybedtools and
gtfparse.
