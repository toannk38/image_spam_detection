import logging 
import sys

def create_logger(filename="logs.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Create a file handler and set the level to debug
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    if len(logger.handlers) == 2:
        return logger

    # Create a console handler and set the level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger 