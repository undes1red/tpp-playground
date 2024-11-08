import sys
import logging


def get_logger(name = None, root = True, mode = 'logging'):
    if mode == 'logging':
        return get_logger_logging(name, root)
    elif mode == 'loguru':
        return get_logger_loguru(name, root)


class LogFormat(logging.Formatter):
    
    green = "\x1b[38;5;2m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"
    
    debug = "\x1b[38;5;243m"
    info = ""
    warning = "\x1b[38;5;214m"
    error = "\x1b[38;5;196m"
    critical = '\x1b[48;5;196m'
    
    format = f"{green}%(asctime)s{reset}{cyan} [Line %(lineno)d of %(filename)s]{reset}: {{}}%(message)s{reset}"

    FORMATS = {
        logging.DEBUG: format.format(debug),
        logging.INFO: format.format(info),
        logging.WARNING: format.format(warning),
        logging.ERROR: format.format(error),
        logging.CRITICAL: format.format(critical),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt = '%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger_logging(name = None, root = True):
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
    ch = logging.StreamHandler(sys.stdout)
    # add formatter to ch
    ch.setFormatter(LogFormat())
    # add ch to logger
    logger.addHandler(ch)

    return logger


def get_logger_loguru(name, root = True):
    from loguru import logger
    
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


if __name__ == "__main__":
    logger = get_logger(name = f'{__name__}', root = True, mode = 'logging')
    logger.debug('DEBUG.')
    logger.info('INFO.')
    logger.warning('WARNING.')
    logger.error('ERROR.')
    logger.critical('ERROR.')