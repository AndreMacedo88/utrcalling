"""
utrcalling is a package to calculate the number of molecules that map to UTR regions 
and the size of the UTR for those molecules.

For the source see: https://github.com/AndreMacedo88/utrcalling

For detailed documentation see: TODO
"""

import psutil
from colorama import init, Fore, Style

from utrcalling._metadata import __version__, __author__, __license__

from utrcalling import core
from utrcalling import tests
from utrcalling import tools

from utrcalling.core import protocols


def show_logo():
    print(f"\n{Style.BRIGHT}{Fore.MAGENTA}/{Fore.RED}/{Fore.YELLOW}/" +
          f" {Fore.WHITE}utrcalling {Fore.YELLOW}\\{Fore.RED}\\{Fore.MAGENTA}\\" +
          f"{Style.RESET_ALL} 🧬 v{__version__} 🧬\n")


# Print a header/logo to show that the utrcalling tool has started
# First, prevent joblib instances from printing the header

caller_process = psutil.Process().cmdline()

if any("joblib" in element for element in caller_process):
    pass
else:
    init()
    show_logo()
