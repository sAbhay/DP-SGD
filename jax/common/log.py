import logging
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(filename='log.txt',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    logging.disable_existing_loggers=False
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
