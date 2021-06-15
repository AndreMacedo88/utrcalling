import logging
from datetime import datetime

def module_logger(store_log=False):
    # create logger with the name of the module, or whatever
    module_logger = logging.getLogger("utrcallingtool")
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not len(module_logger.handlers):
        now = datetime.now()
        module_logger.setLevel(logging.DEBUG)
        if store_log:
            # create file handler which logs even debug messages
            fh = logging.FileHandler(f'{now.strftime("%Y%m%d-%Hh%Mm")}-utrcalling.log')
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
