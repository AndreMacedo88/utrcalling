"""
utrcalling is a package to calculate the number of molecules that map to UTR regions 
and the size of the UTR for those molecules.

For the source see: https://github.com/AndreMacedo88/utrcalling

For detailed documentation see: TODO
"""

import os
import psutil
from colorama import init, Fore, Style


from utrcalling._metadata import __version__, __author__, __license__

# from utrcalling import core
# from utrcalling import tests
# from utrcalling import tools

# from utrcalling.core import protocols


def show_logo():
    print(f"\n{Style.BRIGHT}{Fore.MAGENTA}/{Fore.RED}/{Fore.YELLOW}/" +
          f" {Fore.WHITE}utrcalling {Fore.YELLOW}\\{Fore.RED}\\{Fore.MAGENTA}\\" +
          f"{Style.RESET_ALL} 🧬 v{__version__} 🧬\n")


# Print a header/logo to show that the utrcalling tool has started
# But prevent joblib and ray instances from printing the header
caller_process = psutil.Process().cmdline()
initialize = True
for element in caller_process:
    if ("ray::IDLE" in element) or ("joblib" in element):
        initialize = False
        break
if initialize:
    init()
    show_logo()
