import os
from datetime import datetime
import logging
import coloredlogs
from colorama import Fore, Style
import verboselogs

# TODO: module_logger cannot handle storing to a file anymore


def module_logger(disable=False, store_log=False, level='info', reinstall=False):

    # create logger with the name of the module
    verboselogs.install()
    module_logger = logging.getLogger("utrcalling")
    module_logger.propagate = False

    if disable:
        return module_logger

    if not module_logger.handlers:
        now = datetime.now()

        if level.lower() == 'info':
            module_logger.setLevel(logging.INFO)
        elif level.lower() == 'debug':
            module_logger.setLevel(logging.DEBUG)

        # create formatter
        format_log = Fore.YELLOW + '║' + Style.RESET_ALL +\
            ' %(asctime)s %(name)s %(programname)s » %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_log)

        if store_log:
            # create file handler which logs even debug messages
            fh = MakeFileHandler(
                f'logs/{now.strftime("%Y%m%d-%Hh%Mm")}-utrcalling.log')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            # add the handler to the logger
            module_logger.addHandler(fh)

        # create console handler which logs even debug messages
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        # add the handler to the logger
        module_logger.addHandler(ch)

        # Make logs prettier
        coloredlogs.install(level=module_logger.getEffectiveLevel(), logger=module_logger,
                            fmt=format_log, field_styles=field_styles,
                            level_styles=level_styles)

    if reinstall:
        format_log = Fore.YELLOW + '║' + Style.RESET_ALL +\
            ' %(asctime)s %(name)s %(programname)s » %(levelname)s - %(message)s'

        coloredlogs.install(level=module_logger.getEffectiveLevel(), logger=module_logger,
                            fmt=format_log, field_styles=field_styles,
                            level_styles=level_styles)

    return module_logger


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False, errors=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.FileHandler.__init__(
            self, filename, mode, encoding, delay, errors)


field_styles = {'asctime': {'color': 'yellow'}, 'hostname': {'color': 'cyan'},
                'levelname': {'bold': True, 'color': 'black'}, 'name': {'color': 'magenta'},
                'programname': {'color': 'blue'}, 'username': {'color': 'green'}}

level_styles = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'},
                'error': {'color': 'red'}, 'info': {}, 'notice': {'color': 'magenta'},
                'spam': {'color': 'green', 'faint': True},
                'success': {'bold': True, 'color': 'green'},
                'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}


def log_end_run(message):
    logger = module_logger(reinstall=True)
    logger.success(f"Job ended: {message}")
