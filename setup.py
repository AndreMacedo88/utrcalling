import setuptools
import re


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
    name="utrcalling",
    version=version,
    description="Package with tools to calculate molecule UTR sizes from RNA sequencing reads.",
    long_description=readme(),
    author=author,
    author_email="andre.lopes.macedo@gmail.com",
    url="https://github.com/AndreMacedo88/utrcalling",
    packages=setuptools.find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.22.3",
        "pandas>=1.3.0",
        "scipy>=1.6.2",
        "scikit-learn>=0.23.2",
        "regex>=2022.04.24",
        "rapidfuzz>=2.0.0",
        "joblib>=1.0.1",
        "anndata>=0.12.0",
        "scanpy>=1.11.0",
        "gtfparse>=1.2.1, <2",
        "pyarrow",
        "pysam>=0.20.0",
        "pybedtools>=0.8.2",
        "tqdm>=4.59.0",
        "bidict>=0.22.0",
        "psutil",
        "pympler",
        "coloredlogs",
        "verboselogs",
        "colorama",
    ],
    include_package_data=True,
    license="BSD License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            # command = package.module:function
            "utrcalling=utrcalling.runtime:main",
        ],
    },
)
