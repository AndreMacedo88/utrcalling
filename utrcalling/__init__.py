"""
utrcalling is a package to calculate the number of molecules that map to UTR regions 
and the size of the UTR for those molecules.

For the source see: https://github.com/AndreMacedo88/utrcalling

For detailed documentation see: TODO
"""

from utrcalling._metadata import __version__, __author__, __license__

from utrcalling import main_module
from utrcalling import tests
from utrcalling import tools

from utrcalling.main_module import protocols
from utrcalling.main_module import storage