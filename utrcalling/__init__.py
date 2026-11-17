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


def show_logo() -> None:
    """Print utrcalling's name and version, in colour in a terminal."""
    from rich.console import Console
    from rich.text import Text

    slashes = [("/", "bold magenta"), ("/", "bold red"), ("/", "bold yellow")]
    backslashes = [("\\", "bold yellow"), ("\\", "bold red"), ("\\", "bold magenta")]
    logo = Text.assemble(
        *slashes, " ", ("utrcalling", "bold"), " ", *backslashes, f" 🧬 v{__version__} 🧬"
    )
    Console(highlight=False).print(Text("\n") + logo + Text("\n"))
