"""
utrcalling is a package to calculate the number of molecules that map to UTR regions
and the size of the UTR for those molecules.

For the source see: https://github.com/AndreMacedo88/utrcalling

For detailed documentation see: https://github.com/AndreMacedo88/utrcalling
"""

# Re-exported as package attributes (utrcalling.__version__, ...).
from utrcalling._metadata import __author__ as __author__
from utrcalling._metadata import __license__ as __license__
from utrcalling._metadata import __version__ as __version__


def show_logo():
    import psutil
    from colorama import Fore, Style, init

    # Prevent joblib worker processes from printing the header
    caller_process = psutil.Process().cmdline()
    for element in caller_process:
        if "joblib" in element:
            return

    init()
    print(
        f"\n{Style.BRIGHT}{Fore.MAGENTA}/{Fore.RED}/{Fore.YELLOW}/"
        + f" {Fore.WHITE}utrcalling {Fore.YELLOW}\\{Fore.RED}\\{Fore.MAGENTA}\\"
        + f"{Style.RESET_ALL} 🧬 v{__version__} 🧬\n"
    )
