utrcalling
##########

Requisites
----------

Should download GTF and fasta files from http://ftp.ensembl.org/pub/

Installing
----------

Requirements
____________

Before using this module, you must have the following software installed in your environment:

- Python (tested on 3.9 but should work for later versions)
- BEDTools for generating the UTR reference regions pkl file. To install, visit https://github.com/arq5x/bedtools2. Alternatively, if using Anaconda, run ``conda install -c bioconda bedtools`` to install BEDTools in your current environment. Finally, make sure BEDTools is callable on the command-line.

Using pip
_________

This module is currently not available from PyPi.

From source
___________

1. Clone the git repository hosted at: https://github.com/AndreMacedo88/utrcalling
2. Go to the cloned directory in your local machine.
3. Run $ ``pip install .``

Development
___________

1. Clone the git repository hosted at: https://github.com/AndreMacedo88/utrcalling
2. Go to the cloned directory in your local machine.
3. Run $ ``pip install -e . --use-feature=in-tree-build``
   
This will install the package locally and in editable mode, where changes to the 
local files will update the package in real-time.

Run Tests
---------

After installation, you can:

- run the individual test files (example for one test file): 

``python -m unittest tests.test_core``

- Run all test files at once:

``utrcalling run_tests``