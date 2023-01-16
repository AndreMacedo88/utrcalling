import setuptools
import re
from pathlib import Path


def readme():
    """
    Opens and reads the README.rst file.
    """
    with open("README.rst", "r") as readme_file:
        return readme_file.read()


metadata_file = "utrcalling/_metadata.py"
with open(metadata_file, "rt") as file:
    metadata = file.read()
    version_expression = r"^__version__ = ['\"]([^'\"]*)['\"]"
    author_expression = r"^__author__ = ['\"]([^'\"]*)['\"]"
    version_search = re.search(version_expression, metadata, re.M)
    author_search = re.search(author_expression, metadata, re.M)
    if version_search:
        version = version_search.group(1)
    else:
        raise RuntimeError("Unable to find version string in {}.".format(metadata))
    if author_search:
        author = author_search.group(1)
    else:
        raise RuntimeError("Unable to find author string in {}.".format(metadata))

setuptools.setup(
    name='utrcalling',
    version=version,
    description='Package with tools to calculate molecule UTR sizes from RNA sequencing reads.',
    long_description=readme(),
    author=author,
    author_email='andre.lopes.macedo@gmail.com',
    url='https://github.com/AndreMacedo88/utrcalling',
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.22.3",
        "pandas>=1.3.0",
        "scipy",
        "scikit-learn",
        "regex",
        "rapidfuzz>=2.0.0",
        "joblib",
        "anndata",
        "scanpy",
        "gtfparse",
        "htseq",
        "pybedtools",
        "tqdm",
        "psutil",
        "pympler",
        "coloredlogs",
        "verboselogs",
        "colorama"
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    scripts=[x.as_posix() for x in Path("./tools/").glob("*.py")],
    entry_points={
        "console_scripts": [
            # command = package.module:function
            "utrcalling-run=utrcalling.tools.run:main",
            "utrcalling-call_peaks=utrcalling.tools.call_peaks:main",
            "utrcalling-prepare-reference-and-map-reads=utrcalling.tools.prepare_reference_and_map_reads:main",
            "utrcalling-runtests=utrcalling.tests.test_case:main",
            "compute-mean-utr-sizes=utrcalling.tools.compute_mean_utr_sizes:main",
            "remove-peaks-below-count-threshold=utrcalling.tools.remove_peaks_below_count_threshold:main",
            "select-alternative-transcription-events=utrcalling.tools.select_alternative_transcription_events:main",
            "generate-names=utrcalling.tools.generate_utr_names:main",
            "counts-to-percentages=utrcalling.tools.counts_to_percentages:main",
            "concatenate-experiments=utrcalling.tools.concatenate_experiments:main",
            "subset-experiments=utrcalling.tools.subset_experiments:main",
            "split-train-test=utrcalling.tools.split_train_test:main",
            "estimate-best-bandwidth=utrcalling.tools.estimate_best_bandwidth:main",
            "check-distance-to-pas=utrcalling.tools.check_distance_to_pas:main",
            "call-peaks-different-bandwidths=utrcalling.tools.call_peaks_different_bandwidths:main",
        ],
    }
)
