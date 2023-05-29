import logging


def getLogger(name = None, root = True):
    '''
    Get normal loggers or file loggers.

    Args:
    name: The name of a generated logger
    file: print all logs into the file if set.
    '''

    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d]: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger