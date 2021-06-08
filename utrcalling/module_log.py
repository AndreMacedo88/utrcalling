import logging
from datetime import datetime

def module_logger(enable=False):
    if not enable:
        return
    # create logger with the name of the module, or whatever
    module_logger = logging.getLogger("utrcallingtool")
    if not len(module_logger.handlers):
        now = datetime.now()
        module_logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(f'{now.strftime("%Y%m%d-%Hh%Mm")}-utrcalling.log')
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add the handler to the logger
        module_logger.addHandler(fh)
    return module_logger
