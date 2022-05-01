utrcalling
##########

Requisites
----------

Should download GTF and fasta files from http://ftp.ensembl.org/pub/

Installing
----------

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