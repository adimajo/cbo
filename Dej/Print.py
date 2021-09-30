# -*- coding: utf-8 -*-
#======================================== ========  Import statements   ===============================================#
import logging
from logging.handlers import RotatingFileHandler
# ======================================================================================================================


#======================================= Logger ========================================================================
# Instanciate base package
LEVEL = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(LEVEL)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

# add Redirection Steam
steam_handler = logging.StreamHandler()
steam_handler.setLevel(LEVEL)
steam_handler.setFormatter(formatter)
logger.addHandler(steam_handler)


class MyPrinter(object):

    def __init__(self, path_log):
        # Redirection fichier
        file_handler = RotatingFileHandler(path_log, 'a', 1000000, 1)
        file_handler.setLevel(LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    @staticmethod
    def info(msg, importance=0, tab=0):
        """
        Custom function to print a fancy message.

        :param msg: message to print   
        :type msg: string

        :param importance: whether to display special caracters around the message or not
        :type importance: int

        :param tab: Number of tabs to add
        :type tab: int 

        """
        if importance == 1:
            logger.info("\t" * tab + "#=======================================================================")

        logger.info("#{}{}".format("\t" * tab, msg))
        if importance == 1:
            logger.info("\t" * tab + "#=======================================================================")

    @staticmethod
    def error(msg):
        logger.exception(msg)

    @staticmethod
    def warning(msg):
        logger.warning(msg)

    @staticmethod
    def set_level(level):
        logger.setLevel(level)

    @staticmethod
    def get_level():
        return logger.getEffectiveLevel()

    @staticmethod
    def debug(msg):
        logger.debug(msg)

    @staticmethod
    def critical(msg):
        logger.critical(msg)



