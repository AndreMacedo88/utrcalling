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
    description='Package to calculate UTR sizes from sequencing reads.',
    long_description=readme(),
    author=author,
    author_email='andre.lopes.macedo@gmail.com',
    url='https://github.com/AndreMacedo88/utrcalling',
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.22.3",
        "pandas",
        "scipy",
        "scikit-learn",
        "regex",
        "rapidfuzz",
        "joblib",
        "anndata",
        "scanpy",
        "gtfparse",
        "htseq",
        "pybedtools",
        "tqdm",
        "psutil",
        "pympler",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research"
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    scripts=[x.as_posix() for x in Path("./tools/").glob("*.py")] + [
        "utrcall"
    ],
    entry_points={
        "console_scripts": [
            # command = package.module:function
            "generate-names=utrcalling.tools.generate_utr_names:main",
            "subset-experiments=utrcalling.tools.subset_experiments:main",
            "concatenate-experiments=utrcalling.tools.concatenate_experiments:main",
            "utrcalling-runtests=utrcalling.tests.test_case:run_all_tests"
            ],
    }
)
