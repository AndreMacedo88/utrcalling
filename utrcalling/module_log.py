import logging
import os
from datetime import datetime

def module_logger(disable=False, store_log=False):
    # create logger with the name of the module, or whatever
    module_logger = logging.getLogger("utrcallingtool")
    module_logger.propagate = False
    
    if disable:
        return module_logger

    if not module_logger.handlers:
        pass
        now = datetime.now()
        module_logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if store_log:
            # create file handler which logs even debug messages
            fh = MakeFileHandler(f'logs/{now.strftime("%Y%m%d-%Hh%Mm")}-utrcalling.log')
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
        
    return module_logger

class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False, errors=None):            
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay, errors)